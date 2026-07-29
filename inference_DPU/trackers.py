# trackers.py
import time
import numpy as np
import cv2
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from sort import Sort

class ByteTrackAdapter:
    def __init__(self, track_thresh=0.25, track_buffer=30, nms_thresh=0.45):
        self.nms_thresh = nms_thresh
        self.tracker = sv.ByteTrack(track_activation_threshold=track_thresh, lost_track_buffer=track_buffer)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        tracking_time = 0.0
        active_tracks = [] # Formato: [x1, y1, w, h, track_id, class_id]

        if len(all_boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(all_boxes),
                confidence=np.array(all_scores),
                class_id=np.array(all_class_ids)
            ).with_nms(threshold=self.nms_thresh)

            t0 = time.time()
            detections = self.tracker.update_with_detections(detections)
            tracking_time = (time.time() - t0) * 1000

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                for xyxy, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    active_tracks.append([x1, y1, w, h, tid, cid])

                labels = [f"ID {tid} | {class_names[cid]}" if (class_names and cid < len(class_names)) else f"ID {tid}" 
                          for tid, cid in zip(detections.tracker_id, detections.class_id)]
                frame = self.box_annotator.annotate(frame, detections)
                frame = self.label_annotator.annotate(frame, detections, labels=labels)

        return frame, tracking_time, active_tracks


class DeepSortAdapter:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap, embedder="mobilenet", half=True, bgr=True)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        t0 = time.time()
        raw_detections = [([x1, y1, x2-x1, y2-y1], score, cid) for (x1, y1, x2, y2), score, cid in zip(all_boxes, all_scores, all_class_ids)]
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        tracking_time = (time.time() - t0) * 1000

        active_tracks = []
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            w, h = x2 - x1, y2 - y1
            cid = track.get_det_class()
            cid_int = int(cid) if cid is not None else 0
            
            active_tracks.append([x1, y1, w, h, int(track_id), cid_int])

            c_name = class_names[cid_int] if (class_names and cid_int < len(class_names)) else ""
            color = [int(c) for c in self.colors[int(track_id) % len(self.colors)]]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} | {c_name}" if c_name else f"ID {track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, tracking_time, active_tracks


class SortAdapter:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    def update_and_annotate(self, frame, all_boxes, all_scores, all_class_ids, class_names=None):
        t0 = time.time()
        dets = np.column_stack((all_boxes, all_scores)) if len(all_boxes) > 0 else np.empty((0, 5))
        trackers = self.tracker.update(dets)
        tracking_time = (time.time() - t0) * 1000

        active_tracks = []
        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d)
            w, h = x2 - x1, y2 - y1
            active_tracks.append([x1, y1, w, h, int(track_id), 0])

            color = [int(c) for c in self.colors[track_id % len(self.colors)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, tracking_time, active_tracks
