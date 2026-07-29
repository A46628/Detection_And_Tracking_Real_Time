import cv2
import numpy as np
import xir
import vart
import socket
import struct
import supervision as sv
import time
import traceback

XMODEL_PATH = "yolo_tiny.xmodel"
IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45
HOST, PORT = '10.64.10.18', 5000
VIDEO_PATH = "teste2.mp4"
CLASS_NAMES = ["camouflage_soldier", "weapon", "military_tank", "military_truck",
               "military_vehicle", "civilian", "soldier", "civilian_vehicle",
               "military_artillery", "trench", "military_aircraft", "military_warship"]
ANCHORS = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def preprocess(img, fix_point):
    # Primeiro reduz o tamanho (menos píxeis para converter cor)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = (2**fix_point) / 255.0
    return np.expand_dims((img.astype(np.float32) * scale).astype(np.int8), axis=0)

def main():
    graph = xir.Graph.deserialize(XMODEL_PATH)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device").upper()=="DPU"][0]
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    in_fix = runner.get_input_tensors()[0].get_attr("fix_point")
    output_tensors = runner.get_output_tensors()
    output_buffers = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    cap = cv2.VideoCapture(VIDEO_PATH)
    thresh_logit = -np.log(1/CONF_THRESH - 1)
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            h_orig, w_orig = frame.shape[:2]

            # --- 1. TEMPO DE PRÉ-PROCESSAMENTO ---
            start_preprocess = time.time()
            input_data = preprocess(frame, in_fix)
            end_preprocess = time.time()
            preprocess_time = (end_preprocess - start_preprocess) * 1000

            # --- 2. TEMPO DE INFERÊNCIA (DPU) ---
            start_inference = time.time()
            job_id = runner.execute_async([input_data], output_buffers)
            runner.wait(job_id)
            end_inference = time.time()
            inference_time = (end_inference - start_inference) * 1000

            # --- 3. TEMPO DE PÓS-PROCESSAMENTO ---
            start_postprocess = time.time()
            all_boxes, all_scores, all_class_ids = [], [], []
            for i, tensor in enumerate(output_tensors):
                grid = tensor.dims[1]
                ofix = tensor.get_attr("fix_point")
                data = output_buffers[i][0].reshape(grid, grid, 3, 5 + len(CLASS_NAMES))
                mask = data[..., 4] > (thresh_logit * (2**ofix))
                if not np.any(mask): continue
                valid = sigmoid(data[mask].astype(np.float32) / (2**ofix))
                scores = valid[:, 4] * np.max(valid[:, 5:], axis=1)
                score_mask = scores > CONF_THRESH
                if not np.any(score_mask): continue
                
                yv, xv = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
                grid_xy = np.repeat(np.expand_dims(np.stack((xv, yv), axis=2), axis=2), 3, axis=2)[mask][score_mask]
                stride = IMG_SIZE // grid
                cx, cy = (grid_xy[:, 0] + valid[score_mask, 0]) * stride, (grid_xy[:, 1] + valid[score_mask, 1]) * stride
                anchors_tile = np.tile(np.array(ANCHORS[i]).reshape(3, 2), (grid, grid, 1, 1))[mask][score_mask]
                bw, bh = (valid[score_mask, 2] * 2)**2 * anchors_tile[:, 0], (valid[score_mask, 3] * 2)**2 * anchors_tile[:, 1]
                
                all_boxes.extend(np.stack([(cx - bw/2)*w_orig/IMG_SIZE, (cy - bh/2)*h_orig/IMG_SIZE, 
                                           (cx + bw/2)*w_orig/IMG_SIZE, (cy + bh/2)*h_orig/IMG_SIZE], axis=1).astype(int).tolist())
                all_scores.extend(scores[score_mask].tolist())
                all_class_ids.extend(np.argmax(valid[score_mask, 5:], axis=1).tolist())
            end_postprocess = time.time()
            postprocess_time = (end_postprocess - start_postprocess) * 1000

            # --- 4. TEMPO DE TRACKING ---
            tracking_time = 0.0
            if len(all_boxes) > 0:
                detections = sv.Detections(
                    xyxy=np.array(all_boxes),
                    confidence=np.array(all_scores),
                    class_id=np.array(all_class_ids)
                )
                detections = detections.with_nms(threshold=NMS_THRESH)
                
                start_tracking = time.time()
                detections = tracker.update_with_detections(detections)
                end_tracking = time.time()
                tracking_time = (end_tracking - start_tracking) * 1000

                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    labels = [f"ID {tid}" for tid in detections.tracker_id]
                    frame = box_annotator.annotate(frame, detections)
                    frame = label_annotator.annotate(frame, detections, labels=labels)

            total_time = preprocess_time + inference_time + postprocess_time + tracking_time

            print(f"--- Frame {frame_idx:04d} ---")
            print(f"  Pre-processamento: {preprocess_time:.2f} ms")
            print(f"  Inferencia (DPU): {inference_time:.2f} ms")
            print(f"  Pos-processamento: {postprocess_time:.2f} ms")
            print(f"  Tracking:          {tracking_time:.2f} ms")
            print(f"  TEMPO TOTAL:       {total_time:.2f} ms\n")

            # Compressão e Envio por Socket
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = jpeg.tobytes()
            s.sendall(struct.pack(">L", len(data)) + data)

    except Exception as e:
        traceback.print_exc()
    finally:
        cap.release()
        s.close()
        del runner

if __name__ == "__main__":
    main()