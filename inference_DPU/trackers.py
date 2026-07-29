import time
import numpy as np
import cv2
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from sort import Sort

class ByteTrackAdapter:
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, nms_thresh: float = 0.45):
        """
        Inicializa o ByteTrack (Supervision).
        """
        self.nms_thresh = nms_thresh
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        tracking_time = 0.0

        if len(all_boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(all_boxes),
                confidence=np.array(all_scores),
                class_id=np.array(all_class_ids)
            )
            detections = detections.with_nms(threshold=self.nms_thresh)

            t0 = time.time()
            detections = self.tracker.update_with_detections(detections)
            tracking_time = (time.time() - t0) * 1000

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                labels = []
                for tid, cid in zip(detections.tracker_id, detections.class_id):
                    c_name = class_names[cid] if (class_names and cid < len(class_names)) else f"cls:{cid}"
                    labels.append(f"ID {tid} | {c_name}")

                frame = self.box_annotator.annotate(frame, detections)
                frame = self.label_annotator.annotate(frame, detections, labels=labels)

        return frame, tracking_time


class DeepSortAdapter:
    def __init__(self, max_age: int = 30, n_init: int = 3, nms_max_overlap: float = 1.0):
        """
        Inicializa o DeepSORT (deep-sort-realtime).
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        t0 = time.time()
        
        raw_detections = []
        for box, score, class_id in zip(all_boxes, all_scores, all_class_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], score, class_id))

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        tracking_time = (time.time() - t0) * 1000

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            class_id = track.get_det_class()
            if class_names and class_id is not None and int(class_id) < len(class_names):
                cls_name = class_names[int(class_id)]
            else:
                cls_name = f"Cls {class_id}" if class_id is not None else ""

            color = [int(c) for c in self.colors[int(track_id) % len(self.colors)]]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} | {cls_name}" if cls_name else f"ID {track_id}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, tracking_time



class SortAdapter:
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Inicializa o SORT puro (Filtro de Kalman + IoU).
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        t0 = time.time()
        
        if len(all_boxes) > 0:
            dets = np.column_stack((all_boxes, all_scores))
        else:
            dets = np.empty((0, 5))

        trackers = self.tracker.update(dets)
        tracking_time = (time.time() - t0) * 1000

        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d)
            color = [int(c) for c in self.colors[track_id % len(self.colors)]]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, tracking_time
