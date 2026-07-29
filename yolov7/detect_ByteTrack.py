import os
import cv2
import torch
import numpy as np
import supervision as sv

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ==========================================
# Configurações de Caminhos e Parâmetros
# ==========================================
VIDEO_PATH = "test4.mp4"            # Teu vídeo de teste
MODEL_PATH = "yolov7O.pt"            # Teu modelo treinado YOLOv7
IMG_SIZE = 640                       # Resolução de entrada do modelo

# --- AJUSTE: Subimos ligeiramente o NMS para evitar falsos positivos rápidos ---
CONF_THRES = 0.25                    # Evita detetar "ruído" que gera IDs falsos altos no início
IOU_THRES = 0.45                     # Threshold de IoU para o NMS
OUTPUT_DIR = "yolo_tracking_results" # Pasta de saída para as labels .txt

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Inicialização do Modelo, Tracker e Anotadores
# ==========================================
device = select_device('')
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

# Inicializa o tracker ByteTrack
tracker = sv.ByteTrack()

# --- AJUSTE CRÍTICO: Configuração Robusta para evitar ID Switching ---
if hasattr(tracker, 'track_activation_threshold'):
    tracker.track_activation_threshold = 0.35  # Confiança mínima para iniciar um ID novo (evita IDs altos à toa)
    tracker.minimum_matching_threshold = 0.20  # Mais tolerante na associação por proximidade/IoU
elif hasattr(tracker, 'track_thresh'):
    tracker.track_thresh = 0.35
    tracker.match_thresh = 0.20

if hasattr(tracker, 'track_buffer'):
    tracker.track_buffer = 150                 # Aumentado para 150 frames: mantém o ID guardado por mais tempo se o tanque sumir
elif hasattr(tracker, 'frame_rate'):
    tracker.frame_rate = 30

# Anotadores com linhas mais espessas para melhor visualização
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)

# ==========================================
# Captura de Vídeo e Configuração de FPS
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo {VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
largura_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Se o OpenCV não conseguir ler os FPS, assumimos 30 por padrão
if video_fps <= 0:
    video_fps = 30

# Calcular o delay necessário em milissegundos para o vídeo correr no tempo real original
wait_time = max(1, int(1000 / video_fps))

print(f"Vídeo aberto. Resolução: {largura_video}x{altura_video} | FPS Original: {video_fps}")
print(f"A sincronizar exibição para {wait_time}ms por frame.")

tempos_tracking_puro = []

# ==========================================
# Loop Principal de Processamento
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if frame_idx % 100 == 0 or frame_idx == 1:
        print(f"Progresso: Frame {frame_idx}/{total_frames}")

    # Pré-processamento
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]
    detections = sv.Detections.empty()

    if pred is not None and len(pred):
        pred_scaled = pred.clone()
        pred_scaled[:, :4] = scale_coords(img.shape[2:], pred_scaled[:, :4], frame.shape).round()

        boxes = pred_scaled[:, :4].cpu().numpy()
        scores = pred_scaled[:, 4].cpu().numpy()
        classes = pred_scaled[:, 5].cpu().numpy().astype(int)

        # Filtro de Classe (2 = Tanques)
        indices_tanques = (classes == 2)
        boxes_tanques = boxes[indices_tanques]
        scores_tanques = scores[indices_tanques]
        classes_tanques = classes[indices_tanques]

        if len(boxes_tanques) > 0:
            detections = sv.Detections(
                xyxy=boxes_tanques,
                confidence=scores_tanques,
                class_id=classes_tanques
            )

    # Cronometragem do tracking
    start_track = cv2.getTickCount()
    detections = tracker.update_with_detections(detections)
    end_track = cv2.getTickCount()
    
    tempo_this_track = (end_track - start_track) / cv2.getTickFrequency()
    tempos_tracking_puro.append(tempo_this_track)

    # Desenhar os IDs estáveis
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        labels = [f"Tanque ID: {track_id}" for track_id in detections.tracker_id]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Mostrar a Janela de Vídeo
    cv2.imshow("Tracking em Tempo Real (Estabilizado)", frame)

    # Gravar as labels .txt
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        linhas_frame_yolo = []
        for track_id, tlbr, conf in zip(detections.tracker_id, detections.xyxy, detections.confidence):
            x_min, y_min, x_max, y_max = tlbr[0], tlbr[1], tlbr[2], tlbr[3]
            w_pixels = x_max - x_min
            h_pixels = y_max - y_min
            
            x_center_yolo = (x_min + (w_pixels / 2.0)) / largura_video
            y_center_yolo = (y_min + (h_pixels / 2.0)) / altura_video
            w_yolo = w_pixels / largura_video
            h_yolo = h_pixels / altura_video
            
            linha = f"{2} {track_id} {x_center_yolo:.6f} {y_center_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f} {conf:.4f}\n"
            linhas_frame_yolo.append(linha)
        
        txt_filename = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.writelines(linhas_frame_yolo)

    # --- AJUSTE: wait_time dinâmico controla a velocidade real do vídeo ---
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        print("Processamento interrompido.")
        break

cap.release()
cv2.destroyAllWindows()