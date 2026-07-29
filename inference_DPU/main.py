import argparse
import cv2
import numpy as np
import xir
import vart
import socket
import struct
import time
import traceback

from trackers import ByteTrackAdapter, DeepSortAdapter, SortAdapter

CLASS_NAMES = [
    "camouflage_soldier", "weapon", "military_tank", "military_truck",
    "military_vehicle", "civilian", "soldier", "civilian_vehicle",
    "military_artillery", "trench", "military_aircraft", "military_warship"
]
ANCHORS = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def preprocess(img, fix_point, img_size):
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = (2**fix_point) / 255.0
    return np.expand_dims((img.astype(np.float32) * scale).astype(np.int8), axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="DPU YOLO + Multi-Tracker Pipeline (ByteTrack / DeepSORT / SORT)")
    
    parser.add_argument("--xmodel", type=str, default="yolo_tiny.xmodel", help="Path to the .xmodel file")
    parser.add_argument("--video", type=str, default="teste2.mp4", help="Path to the input video file")
    parser.add_argument("--host", type=str, default="10.64.10.18", help="Socket server IP address")
    parser.add_argument("--port", type=int, default=5000, help="Socket server port")
    parser.add_argument("--img-size", type=int, default=640, help="Input size for the neural network")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="NMS threshold")
    
    parser.add_argument("--save-txt", type=str, default="results_mot.txt", help="File path to save MOT format metrics (.txt)")

    parser.add_argument("--tracker", type=str, choices=["bytetrack", "deepsort", "sort"], default="bytetrack", 
                        help="Choose tracking algorithm: bytetrack, deepsort, or sort")

    parser.add_argument("--track-buffer", type=int, default=30, help="[ByteTrack] Frames to keep lost tracks")
    parser.add_argument("--max-age", type=int, default=30, help="[DeepSORT/SORT] Maximum frames to keep a track without detections")
    parser.add_argument("--n-init", type=int, default=3, help="[DeepSORT] Minimum consecutive detections to confirm a track")
    parser.add_argument("--min-hits", type=int, default=3, help="[SORT] Minimum detections required before activating a track")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="[SORT] IoU threshold for data association")

    return parser.parse_args()


def main():
    args = parse_args()

    # Tracker selection
    if args.tracker == "bytetrack":
        tracker_manager = ByteTrackAdapter(
            track_thresh=args.conf_thresh,
            track_buffer=args.track_buffer,
            nms_thresh=args.nms_thresh
        )
    elif args.tracker == "deepsort":
        tracker_manager = DeepSortAdapter(
            max_age=args.max_age,
            n_init=args.n_init,
            nms_max_overlap=args.nms_thresh
        )
    elif args.tracker == "sort":
        tracker_manager = SortAdapter(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_thresh
        )

    # Vitis AI / DPU Initialization
    graph = xir.Graph.deserialize(args.xmodel)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"][0]
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    in_fix = runner.get_input_tensors()[0].get_attr("fix_point")
    output_tensors = runner.get_output_tensors()
    output_buffers = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

    # Socket & Video Initialization
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    cap = cv2.VideoCapture(args.video)
    
    thresh_logit = -np.log(1 / args.conf_thresh - 1)
    frame_idx = 0

    # Open metrics file if specified
    txt_file = open(args.save_txt, "w") if args.save_txt else None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            frame_idx += 1
            h_orig, w_orig = frame.shape[:2]

            # 1. Preprocessing
            t0 = time.time()
            input_data = preprocess(frame, in_fix, args.img_size)
            preprocess_time = (time.time() - t0) * 1000

            # 2. DPU Inference
            t0 = time.time()
            job_id = runner.execute_async([input_data], output_buffers)
            runner.wait(job_id)
            inference_time = (time.time() - t0) * 1000

            # 3. Postprocessing
            t0 = time.time()
            all_boxes, all_scores, all_class_ids = [], [], []
            
            for i, tensor in enumerate(output_tensors):
                grid = tensor.dims[1]
                ofix = tensor.get_attr("fix_point")
                data = output_buffers[i][0].reshape(grid, grid, 3, 5 + len(CLASS_NAMES))
                mask = data[..., 4] > (thresh_logit * (2**ofix))
                
                if not np.any(mask): 
                    continue
                
                valid = sigmoid(data[mask].astype(np.float32) / (2**ofix))
                scores = valid[:, 4] * np.max(valid[:, 5:], axis=1)
                score_mask = scores > args.conf_thresh
                
                if not np.any(score_mask): 
                    continue
                
                yv, xv = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
                grid_xy = np.repeat(np.expand_dims(np.stack((xv, yv), axis=2), axis=2), 3, axis=2)[mask][score_mask]
                stride = args.img_size // grid
                
                cx, cy = (grid_xy[:, 0] + valid[score_mask, 0]) * stride, (grid_xy[:, 1] + valid[score_mask, 1]) * stride
                anchors_tile = np.tile(np.array(ANCHORS[i]).reshape(3, 2), (grid, grid, 1, 1))[mask][score_mask]
                bw, bh = (valid[score_mask, 2] * 2)**2 * anchors_tile[:, 0], (valid[score_mask, 3] * 2)**2 * anchors_tile[:, 1]
                
                all_boxes.extend(np.stack([
                    (cx - bw/2) * w_orig / args.img_size, 
                    (cy - bh/2) * h_orig / args.img_size, 
                    (cx + bw/2) * w_orig / args.img_size, 
                    (cy + bh/2) * h_orig / args.img_size
                ], axis=1).astype(int).tolist())
                
                all_scores.extend(scores[score_mask].tolist())
                all_class_ids.extend(np.argmax(valid[score_mask, 5:], axis=1).tolist())
                
            postprocess_time = (time.time() - t0) * 1000

            # 4. Tracking
            frame, tracking_time, active_tracks = tracker_manager.update_and_annotate(
                frame, all_boxes, all_scores, all_class_ids, class_names=CLASS_NAMES
            )

            # 5. Save MOT metrics to file
            if txt_file and active_tracks:
                for track in active_tracks:
                    x1, y1, w, h, track_id, class_id = track
                    # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    line = f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{class_id + 1},-1,-1\n"
                    txt_file.write(line)

            total_time = preprocess_time + inference_time + postprocess_time + tracking_time

            print(f"--- Frame {frame_idx:04d} [{args.tracker.upper()}] ---")
            print(f"  Tracks Active:     {len(active_tracks)}")
            print(f"  Total Time:        {total_time:.2f} ms\n")

            # 6. Stream via Socket
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = jpeg.tobytes()
            s.sendall(struct.pack(">L", len(data)) + data)

    except Exception as e:
        traceback.print_exc()
    finally:
        if txt_file:
            txt_file.close()
            print(f"[INFO] MOT metrics successfully saved to {args.save_txt}")
        cap.release()
        s.close()
        del runner

if __name__ == "__main__":
    main()
