

import cv2
import numpy as np
import xir
import vart
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Path to compiled DPU model
XMODEL_PATH = "Yolooooov7.xmodel"

# Input image path
IMAGE_PATH  = "011164.jpg"

# Output image path
OUTPUT_PATH = "resultado_final.jpg"

# Network input resolution (YOLO trained at 640x640)
IMG_SIZE = 640

# Confidence threshold for objectness * class score
CONF_THRESH = 0.25

# IoU threshold for Non-Max Suppression
NMS_THRESH  = 0.45

# Class labels (must match training)
CLASS_NAMES = [
    "camouflage_soldier", "weapon", "military_tank", "military_truck",
    "military_vehicle", "civilian", "soldier", "civilian_vehicle",
    "military_artillery", "trench", "military_aircraft", "military_warship"
]

# YOLO anchors (3 anchors per scale)
ANCHORS = [
    [12,16, 19,36, 40,28],          # Small scale (e.g. 80x80)
    [36,75, 76,55, 72,146],         # Medium scale (40x40)
    [142,110, 192,243, 459,401]     # Large scale (20x20)
]



def sigmoid(x):
    """
    Numerically stable sigmoid.
    Used to decode YOLO outputs.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def get_dpu_subgraph(graph):
    """
    Extracts the DPU subgraph from the compiled XMODEL.
    Usually there is only one DPU subgraph.
    """
    root = graph.get_root_subgraph()
    subgraphs = root.toposort_child_subgraph()
    return [
        s for s in subgraphs
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ][0]


def preprocess(img, fix_point):
    """
    Prepares image for DPU inference:
    - Resize to 640x640
    - BGR -> RGB
    - Normalize to [0,1]
    - Scale using DPU fix_point
    - Convert to INT8
    - Add batch dimension
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img * (2 ** fix_point)
    img = img.astype(np.int8)
    return np.expand_dims(img, axis=0)


def nms(boxes, scores):
    """
    Applies OpenCV Non-Max Suppression
    to remove duplicated bounding boxes.
    """
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    return idxs.flatten() if len(idxs) > 0 else []

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load DPU model
    # --------------------------------------------------------
    print("[INFO] Loading XMODEL...")
    graph = xir.Graph.deserialize(XMODEL_PATH)
    dpu_subgraph = get_dpu_subgraph(graph)
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    # Get input/output tensor information
    input_tensor = runner.get_input_tensors()[0]
    output_tensors = runner.get_output_tensors()

    # Fix-point used for input quantization
    in_fix = input_tensor.get_attr("fix_point")

    # --------------------------------------------------------
    # Load input image
    # --------------------------------------------------------
    print("[INFO] Loading image...")
    img_orig = cv2.imread(IMAGE_PATH)
    h_orig, w_orig = img_orig.shape[:2]

    # Preprocess image for DPU
    input_data = preprocess(img_orig, in_fix)

    # Allocate output buffers
    output_data = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

    # --------------------------------------------------------
    # Run inference on DPU
    # --------------------------------------------------------
    print("[INFO] Running DPU inference...")
    job_id = runner.execute_async([input_data], output_data)
    runner.wait(job_id)

    # Lists to store final detections
    boxes, scores, class_ids = [], [], []

    # --------------------------------------------------------
    # POST-PROCESSING (CPU)
    # --------------------------------------------------------
    print("[INFO] Post-processing outputs...")

    # Loop over each output scale (e.g. 80x80, 40x40, 20x20)
    for i, tensor in enumerate(output_tensors):

        # Grid size for this output
        grid = tensor.dims[1]

        # Stride relative to input image
        stride = IMG_SIZE // grid

        # Anchors for this scale
        anchors = ANCHORS[i]

        # Fix-point used for output quantization
        ofix = tensor.get_attr("fix_point")

        # Dequantize output tensor
        data = output_data[i][0].astype(np.float32) / (2 ** ofix)

        # Loop over the 3 anchors
        for a in range(3):

            # Offset in the channel dimension
            # (tx, ty, tw, th, obj + num_classes)
            offset = a * (5 + len(CLASS_NAMES))

            # Loop over grid cells
            for y in range(grid):
                for x in range(grid):

                    # Objectness score
                    obj = sigmoid(data[y, x, offset + 4])
                    if obj < CONF_THRESH:
                        continue

                    # Class probabilities
                    cls_scores = sigmoid(
                        data[y, x, offset + 5 : offset + 5 + len(CLASS_NAMES)]
                    )

                    cls_id = np.argmax(cls_scores)
                    score = obj * cls_scores[cls_id]

                    if score < CONF_THRESH:
                        continue

                    # Bounding box regression
                    tx = sigmoid(data[y, x, offset + 0])
                    ty = sigmoid(data[y, x, offset + 1])
                    tw = sigmoid(data[y, x, offset + 2])
                    th = sigmoid(data[y, x, offset + 3])

                    # Decode center coordinates
                    cx = (x + tx) * stride
                    cy = (y + ty) * stride

                    # Decode width and height
                    bw = (tw * 2) ** 2 * anchors[a*2]
                    bh = (th * 2) ** 2 * anchors[a*2+1]

                    # Convert to original image scale
                    x1 = int((cx - bw / 2) * w_orig / IMG_SIZE)
                    y1 = int((cy - bh / 2) * h_orig / IMG_SIZE)
                    w  = int(bw * w_orig / IMG_SIZE)
                    h  = int(bh * h_orig / IMG_SIZE)

                    boxes.append([x1, y1, w, h])
                    scores.append(float(score))
                    class_ids.append(cls_id)

    # --------------------------------------------------------
    # Apply Non-Max Suppression
    # --------------------------------------------------------
    keep = nms(boxes, scores)

    # Draw final detections
    for i in keep:
        x, y, w, h = boxes[i]
        label = f"{CLASS_NAMES[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(img_orig, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_orig, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Save output image
    cv2.imwrite(OUTPUT_PATH, img_orig)

    print(f"[OK] Output saved to {OUTPUT_PATH}")
    print(f"[OK] Total detections: {len(keep)}")

    del runner


if __name__ == "__main__":
    main()
