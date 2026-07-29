import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
from deep_sort_realtime.deepsort_tracker import DeepSort

# =====================================================================
# CONFIGURAÇÕES FIXAS (HARDCODED)
# =====================================================================
VIDEO_PATH = "teste2.mp4"            # Teu vídeo de teste
MODEL_PATH = "yolovx.pt"            # Teu modelo treinado YOLOv7
IMG_SIZE = 640                       # Resolução de entrada do modelo
CONF_THRES = 0.25                    # Threshold de confiança da deteção (Mais alto = menos IDs fantasmas)
IOU_THRES = 0.45                     # Threshold de IoU para o NMS
OUTPUT_DIR = "yolo_tracking_results" # Pasta onde vão ser guardados os ficheiros .txt
DEVICE = ""                          # Deixa vazio para auto-select (CUDA se disponível, senão CPU)
AUGMENT = False                      # Ativar Augment Inference se necessário
CLASSES = None                       # Filtrar por classes específicas (ex: [0]), None apanha todas
AGNOSTIC_NMS = False                 # NMS agnóstico à classe
# =====================================================================

# Inicializar o DeepSort Tracker (Ajustado para maior estabilidade de ID)
tracker = DeepSort(
    max_age=30,             # Mantém o ID guardado por até 30 frames se o objeto sumir
    n_init=3,               # O objeto precisa de ser detetado 3 frames seguidos para ganhar um ID (evita IDs falsos altos no início)
    max_cosine_distance=0.5,
    nn_budget=200
)

def detect():
    # Garantir que a pasta de output existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Inicializar logs e dispositivo de hardware
    set_logging()
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # FP16 apenas suportado em ambientes GPU CUDA

    # Carregar o modelo YOLOv7
    model = attempt_load(MODEL_PATH, map_location=device)  
    stride = int(model.stride.max())  
    imgsz = check_img_size(IMG_SIZE, s=stride)  

    if half:
        model.half()  

    # Configurar o Dataloader para ler o vídeo
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Vídeo '{VIDEO_PATH}' não foi encontrado. Verifica o caminho.")
        
    dataset = LoadImages(VIDEO_PATH, img_size=imgsz, stride=stride)

    # --- NOVO: Extrair FPS original para sincronizar a velocidade de reprodução ---
    vid_cap_temp = cv2.VideoCapture(VIDEO_PATH)
    video_fps = int(vid_cap_temp.get(cv2.CAP_PROP_FPS))
    vid_cap_temp.release()
    if video_fps <= 0:
        video_fps = 30
    wait_time = max(1, int(1000 / video_fps))

    # Executar uma inferência fantasma (Warmup)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 para fp16/32
        img /= 255.0  # Normalizar 0-255 para 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup dinâmico caso o tamanho mude
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=AUGMENT)[0]

        # Inferência do Modelo
        t1 = time_synchronized()
        with torch.no_grad():   
            pred = model(img, augment=AUGMENT)[0]
        t2 = time_synchronized()

        # Aplicar o Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
        t3 = time_synchronized()

        # Processar as deteções obtidas
        for i, det in enumerate(pred):  
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Criamos uma cópia limpa para desenhar e não corromper os dados originais
            annotated_frame = im0.copy()

            altura_original, largura_original = im0.shape[0], im0.shape[1]
            linhas_frame_yolo = []
            
            if len(det):
                # Reescalar as caixas do tamanho de inferência (640) para o tamanho nativo do vídeo
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                detections = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], float(conf), int(cls)))

                # Atualizar o Filtro de Kalman e ID no DeepSort
                tracks = tracker.update_tracks(detections, frame=im0)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    l, t, w, h = track.to_ltrb()
                    
                    x1_b, y1_b, x2_b, y2_b = map(int, [l, t, w, h])

                    # --- NOVO: Desenhar a bounding box e o ID no frame ---
                    cor_caixa = (0, 255, 0) # Verde
                    cv2.rectangle(annotated_frame, (x1_b, y1_b), (x2_b, y2_b), cor_caixa, 2)
                    
                    texto = f"ID: {track_id}"
                    cv2.putText(annotated_frame, texto, (x1_b, max(15, y1_b - 8)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_caixa, 2)

                    cls_id = track.get_det_class()
                    if cls_id is None:
                        cls_id = 0  
                    
                    conf_val = track.get_det_conf()
                    if conf_val is None:
                        conf_val = 1.0

                    # Converter as coordenadas absolutas (pixels) para formato YOLO normalizado (0 a 1)
                    x_centro_pixels = x1_b + ((x2_b - x1_b) / 2.0)
                    y_centro_pixels = y1_b + ((y2_b - y1_b) / 2.0)
                    w_pixels = x2_b - x1_b
                    h_pixels = y2_b - y1_b
                    
                    x_center_yolo = x_centro_pixels / largura_original
                    y_center_yolo = y_centro_pixels / altura_original
                    w_yolo = w_pixels / largura_original
                    h_yolo = h_pixels / altura_original

                    # Construir a linha do ficheiro
                    linha = f"{int(cls_id)} {track_id} {x_center_yolo:.6f} {y_center_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f} {conf_val:.4f}\n"
                    linhas_frame_yolo.append(linha)

                # Guardar o ficheiro se existirem alvos válidos a ser seguidos no frame atual
                if len(linhas_frame_yolo) > 0:
                    txt_filename = os.path.join(OUTPUT_DIR, f"frame_{frame:06d}.txt")
                    with open(txt_filename, "w", encoding="utf-8") as f:
                        f.writelines(linhas_frame_yolo)

            # --- NOVO: Mostrar a janela gráfica sincronizada ---
            cv2.imshow("DeepSort Tracking em Tempo Real", annotated_frame)

            # Log simples no terminal para veres o progresso em tempo real
            print(f"Frame {frame:05d} | Inferência: {(1E3 * (t2 - t1)):.1f}ms | NMS: {(1E3 * (t3 - t2)):.1f}ms")

        # --- NOVO: Escuta o teclado com base no wait_time correto. Pressionar 'q' cancela ---
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            print("\n[AVISO] Processamento interrompido pelo utilizador.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print(f"\n[SUCESSO] Processamento do vídeo concluído!")
    print(f"Resultados guardados na pasta: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Tempo total de execução: {time.time() - t0:.2f} segundos.")


if __name__ == '__main__':
    detect()