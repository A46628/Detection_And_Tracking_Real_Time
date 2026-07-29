import os
import time
import cv2
import numpy as np
import vart
import xir

# ============================================================
# CONFIGURATION vff
# ============================================================

XMODEL_PATH = "yolo_tiny.xmodel"
IMAGE_PATH = "teste_final_ok.jpg"
OUTPUT_PATH = "teste_image.jpg"

IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45

CLASS_NAMES = [
    "camouflage_soldier", "weapon", "military_tank", "military_truck",
    "military_vehicle", "civilian", "soldier", "civilian_vehicle",
    "military_artillery", "trench", "military_aircraft", "military_warship"
]

ANCHORS = [
    [12, 16, 19, 36, 40, 28],       # Small scale (80x80)
    [36, 75, 76, 55, 72, 146],      # Medium scale (40x40)
    [142, 110, 192, 243, 459, 401]  # Large scale (20x20)
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function to normalize values between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def get_dpu_subgraph(graph):
    """Extracts the DPU subgraph from the compiled XMODEL."""
    root = graph.get_root_subgraph()
    subgraphs = root.toposort_child_subgraph()
    for s in subgraphs:
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
            return s
    raise RuntimeError("No DPU subgraph found in the XMODEL.")


def init_dpu(model_path: str):
    """Initializes the Vitis AI DPU runner and retrieves tensor info."""
    print("[INFO] Loading XMODEL...")
    graph = xir.Graph.deserialize(model_path)
    dpu_subgraph = get_dpu_subgraph(graph)
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    input_tensor = runner.get_input_tensors()[0]
    output_tensors = runner.get_output_tensors()

    return runner, input_tensor, output_tensors


def preprocess(img: np.ndarray, fix_point: int) -> np.ndarray:
    """Prepares image for DPU inference (resize, normalization, quantization)."""
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm_img = rgb_img.astype(np.float32) / 255.0
    scaled_img = norm_img * (2 ** fix_point)
    quantized_img = scaled_img.astype(np.int8)
    return np.expand_dims(quantized_img, axis=0)


def nms(boxes: list, scores: list) -> list:
    """Applies OpenCV Non-Max Suppression to filter overlapping boxes."""
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    return idxs.flatten() if len(idxs) > 0 else []


def postprocess(output_data: list, output_tensors: list, img_orig_shape: tuple):
    """Decodes raw DPU outputs into bounding boxes, scores, and class IDs."""
    h_orig, w_orig = img_orig_shape[:2]
    boxes, scores, class_ids = [], [], []

    for i, tensor in enumerate(output_tensors):
        grid = tensor.dims[1]
        stride = IMG_SIZE // grid
        anchors = ANCHORS[i]
        ofix = tensor.get_attr("fix_point")

        # Dequantize output tensor from int8 to float32
        data = output_data[i][0].astype(np.float32) / (2 ** ofix)

        for a in range(3):
            offset = a * (5 + len(CLASS_NAMES))

            for y in range(grid):
                for x in range(grid):
                    # Objectness score check
                    obj = sigmoid(data[y, x, offset + 4])
                    if obj < CONF_THRESH:
                        continue

                    # Class probabilities check
                    cls_scores = sigmoid(
                        data[y, x, offset + 5: offset + 5 + len(CLASS_NAMES)]
                    )
                    cls_id = int(np.argmax(cls_scores))
                    score = float(obj * cls_scores[cls_id])

                    if score < CONF_THRESH:
                        continue

                    # Box offsets decoding
                    tx = sigmoid(data[y, x, offset + 0])
                    ty = sigmoid(data[y, x, offset + 1])
                    tw = sigmoid(data[y, x, offset + 2])
                    th = sigmoid(data[y, x, offset + 3])

                    # Decode center, width and height
                    cx = (x + tx) * stride
                    cy = (y + ty) * stride
                    bw = (tw * 2) ** 2 * anchors[a * 2]
                    bh = (th * 2) ** 2 * anchors[a * 2 + 1]

                    # Scale coordinates to original image size
                    x1 = int((cx - bw / 2) * w_orig / IMG_SIZE)
                    y1 = int((cy - bh / 2) * h_orig / IMG_SIZE)
                    w = int(bw * w_orig / IMG_SIZE)
                    h = int(bh * h_orig / IMG_SIZE)

                    boxes.append([x1, y1, w, h])
                    scores.append(score)
                    class_ids.append(cls_id)

    keep = nms(boxes, scores)
    return boxes, scores, class_ids, keep


def draw_detections(img: np.ndarray, boxes: list, scores: list, class_ids: list, keep: list):
    """Draws bounding boxes and labels on the image."""
    for i in keep:
        x, y, w, h = boxes[i]
        label = f"{CLASS_NAMES[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return img


def print_benchmark_report(t_prep: float, t_infer: float, t_post: float):
    """Displays timing performance summary."""
    t_total = t_prep + t_infer + t_post
    fps_hw = 1000 / t_infer if t_infer > 0 else 0
    fps_total = 1000 / t_total if t_total > 0 else 0

    print("\n" + "=" * 50)
    print("         EXECUTION TIME BENCHMARK REPORT")
    print("=" * 50)
    print(f"Pre-processing (CPU):  {t_prep:6.2f} ms")
    print(f"Inference Pure (DPU):  {t_infer:6.2f} ms  (--> Hardware FPS: {fps_hw:.1f})")
    print(f"Post-processing (CPU): {t_post:6.2f} ms")
    print("-" * 50)
    print(f"Total Pipeline Time:   {t_total:6.2f} ms")
    print(f"Real System Frame Rate:{fps_total:6.2f} FPS")
    print("=" * 50 + "\n")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Input image '{IMAGE_PATH}' not found.")
        return

    # 1. Initialize DPU Runner
    runner, input_tensor, output_tensors = init_dpu(XMODEL_PATH)
    in_fix = input_tensor.get_attr("fix_point")

    # 2. Load Input Image
    img_orig = cv2.imread(IMAGE_PATH)

    # 3. Preprocessing (TIMED)
    t0 = time.perf_counter()
    input_data = preprocess(img_orig, in_fix)
    output_data = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]
    t_prep = (time.perf_counter() - t0) * 1000

    # 4. DPU Inference (TIMED)
    print("[INFO] Running DPU inference...")
    t0 = time.perf_counter()
    job_id = runner.execute_async([input_data], output_data)
    runner.wait(job_id)
    t_infer = (time.perf_counter() - t0) * 1000

    # 5. Post-Processing (TIMED)
    print("[INFO] Post-processing outputs...")
    t0 = time.perf_counter()
    boxes, scores, class_ids, keep = postprocess(output_data, output_tensors, img_orig.shape)
    t_post = (time.perf_counter() - t0) * 1000

    # 6. Draw & Save Detections
    output_img = draw_detections(img_orig, boxes, scores, class_ids, keep)
    cv2.imwrite(OUTPUT_PATH, output_img)

    # 7. Print Benchmark Results
    print_benchmark_report(t_prep, t_infer, t_post)
    print(f"[OK] Output saved to {OUTPUT_PATH}")
    print(f"[OK] Total detections: {len(keep)}")

    # Clean up hardware runner
    del runner


if __name__ == "__main__":
    main()
