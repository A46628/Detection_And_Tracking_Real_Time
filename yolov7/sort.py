import time
from pathlib import Path
import os
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized

# =====================================================================
# CONFIGURAÇÕES FIXAS (HARDCODED)
# =====================================================================
VIDEO_PATH = "teste3.mp4"            # Se o vídeo estiver noutra pasta, usa o caminho completo!
MODEL_PATH = "yolov7O.pt"            
IMG_SIZE = 640                       
CONF_THRES = 0.25                    
IOU_THRES = 0.45                     
OUTPUT_DIR = "sort_tracking_results" 
DEVICE = ""                          
AUGMENT = False                      
CLASSES = None                       
AGNOSTIC_NMS = False                 
# =====================================================================

def calcula_iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, cls_id, conf_val):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.x[:4] = np.array([bbox[0]+w/2., bbox[1]+h/2., w*h, w/float(h)]).reshape((4, 1))
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = cls_id
        self.conf_val = conf_val

    def update(self, bbox, cls_id, conf_val):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.update(np.array([bbox[0]+w/2., bbox[1]+h/2., w*h, w/float(h)]).reshape((4, 1)))
        self.cls_id = cls_id
        self.conf_val = conf_val

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        w = np.sqrt(self.kf.x[2] * self.kf.x[3])
        h = self.kf.x[2] / w
        return np.array([self.kf.x[0]-w/2., self.kf.x[1]-h/2., self.kf.x[0]+w/2., self.kf.x[1]+h/2.]).reshape((1, 4))

    def get_state(self):
        w = np.sqrt(self.kf.x[2] * self.kf.x[3])
        h = self.kf.x[2] / w
        return np.array([self.kf.x[0]-w/2., self.kf.x[1]-h/2., self.kf.x[0]+w/2., self.kf.x[1]+h/2.]).reshape((1, 4))

class SortTracker:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = calcula_iou(det[:4], trk)

        matched_indices = []
        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append([r, c])
        matched_indices = np.array(matched_indices) if len(matched_indices) > 0 else np.empty((0, 2), dtype=int)

        unmatched_detections = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]

        for m in matched_indices:
            self.trackers[m[1]].update(dets[m[0]][:4], int(dets[m[0]][5]), dets[m[0]][4])

        for i in unmatched_detections:
            trk = KalmanBoxTracker(dets[i][:4], int(dets[i][5]), dets[i][4])
            self.trackers.append(trk)

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id, trk.cls_id, trk.conf_val])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

# Inicializar o tracker globalmente
tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)

def detect():
    print("-> Inicializando o script SORT Puro...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_logging()
    device = select_device(DEVICE)
    half = device.type != 'cpu'

    print(f"-> Carregando modelo YOLOv7 de: {MODEL_PATH}")
    model = attempt_load(MODEL_PATH, map_location=device)  
    stride = int(model.stride.max())  
    imgsz = check_img_size(IMG_SIZE, s=stride)  

    if half:
        model.half()  

    # Verificar caminhos de forma absoluta para evitar falsos vazios
    abs_video_path = os.path.abspath(VIDEO_PATH)
    print(f"-> Tentando abrir o vídeo em: {abs_video_path}")
    if not os.path.exists(abs_video_path):
        print(f"[ERRO CRÍTICO] Ficheiro de vídeo não encontrado no caminho: {abs_video_path}")
        return
        
    dataset = LoadImages(abs_video_path, img_size=imgsz, stride=stride)
    print(f"-> Dataset de imagens carregado com sucesso.")

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  

    t0 = time.time()
    frame_processado = False

    for path, img, im0s, vid_cap in dataset:
        frame_processado = True
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():   
            pred = model(img, augment=AUGMENT)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
        t3 = time_synchronized()

        for i, det in enumerate(pred):  
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            altura_original, largura_original = im0.shape[0], im0.shape[1]
            linhas_frame_yolo = []
            
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                dets_to_sort = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(float, xyxy)
                    dets_to_sort.append([x1, y1, x2, y2, float(conf), int(cls)])
                
                dets_to_sort = np.array(dets_to_sort)
                tracked_objects = tracker.update(dets_to_sort)

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id, cls_id, conf_val = obj
                    w = x2 - x1
                    h = y2 - y1

                    x_centro_pixels = x1 + (w / 2.0)
                    y_centro_pixels = y1 + (h / 2.0)
                    
                    x_center_yolo = x_centro_pixels / largura_original
                    y_center_yolo = y_centro_pixels / altura_original
                    w_yolo = w / largura_original
                    h_yolo = h / altura_original

                    linha = f"{int(cls_id)} {int(track_id)} {x_center_yolo:.6f} {y_center_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f} {conf_val:.4f}\n"
                    linhas_frame_yolo.append(linha)

                if len(linhas_frame_yolo) > 0:
                    txt_filename = os.path.join(OUTPUT_DIR, f"frame_{frame:06d}.txt")
                    with open(txt_filename, "w", encoding="utf-8") as f:
                        f.writelines(linhas_frame_yolo)

            print(f"Frame {frame:05d} [SORT Puro] | Inferência: {(1E3 * (t2 - t1)):.1f}ms | NMS: {(1E3 * (t3 - t2)):.1f}ms")

    if not frame_processado:
        print("[AVISO] O loop de frames não chegou a iniciar. O arquivo de vídeo pode estar corrompido ou vazio.")
    else:
        print(f"\n[SUCESSO] Processamento concluído com SORT Puro!")
        print(f"Resultados guardados na pasta: {os.path.abspath(OUTPUT_DIR)}")

# Esta linha garante a execução direta via comando terminal
if __name__ == '__main__':
    detect()