import os
import glob
import re

# Mapeamento de classes 
classes = [
    "camouflage_soldier", "weapon", "military_tank", "military_truck", 
    "military_vehicle", "civilian", "soldier", "civilian_vehicle", 
    "military_artillery", "trench", "military_aircraft", "military_warship"
]

MAX_FRAME = 5513 

def calcular_iou(box1, box2):
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    int_xtl, int_ytl = max(x1_tl, x2_tl), max(y1_tl, y2_tl)
    int_xbr, int_ybr = min(x1_br, x2_br), min(y1_br, y2_br)
    int_area = max(0, int_xbr - int_xtl) * max(0, int_ybr - int_ytl)
    area1 = (x1_br - x1_tl) * (y1_br - y1_tl)
    area2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    uni_area = area1 + area2 - int_area
    return int_area / uni_area if uni_area > 0 else 0

linhas_unificadas = {}
next_id = 1 
objetos_no_frame_anterior = {}

# CORREÇÃO 1: Ordenação robusta baseada estritamente no número do frame_XXXXXX.txt
txt_files = glob.glob("yolo_tracking_results_/frame_*.txt")
txt_files = sorted(txt_files, key=lambda x: int(re.search(r'frame_(\d+)', os.path.basename(x)).group(1)))

# SE O TEU VÍDEO NÃO FOR 1920x1080, AJUSTA ESTES DOIS VALORES ABAIXO
LARGURA_VIDEO = 1920 
ALTURA_VIDEO = 1080

for file_path in txt_files:
    base_name = os.path.basename(file_path)
    
    # CORREÇÃO 2: Capturar corretamente o número do frame único do nome do ficheiro
    match = re.search(r'frame_(\d+)', base_name)
    if not match: 
        continue
    frame_num = int(match.group(1))
    
    if frame_num > MAX_FRAME: 
        continue
    
    objetos_atuais = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            
            # CORREÇÃO 3: Adaptado para ler as 7 colunas geradas pelo script de tracking anterior
            if len(parts) >= 6:
                cls_id = int(parts[0])
                # Ignoramos parts[1] (que era o track_id interno do byteTrack) se quiseres refazer o cálculo de IDs via IoU
                # ou podes usá-lo diretamente. Mantendo a tua lógica de IoU:
                x_c, y_c, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) == 7 else 1.0
                
                label = classes[cls_id]
                
                # Converter coordenadas YOLO normais para píxeis absolutos (xtl, ytl, xbr, ybr)
                xtl = (x_c - w/2) * LARGURA_VIDEO
                ytl = (y_c - h/2) * ALTURA_VIDEO
                xbr = (x_c + w/2) * LARGURA_VIDEO
                ybr = (y_c + h/2) * ALTURA_VIDEO
                
                # Guardamos também a largura e altura absolutas para o formato MOT
                w_abs = w * LARGURA_VIDEO
                h_abs = h * ALTURA_VIDEO
                
                objetos_atuais.append(([xtl, ytl, xbr, ybr], w_abs, h_abs, cls_id, label, conf))
                
    objetos_associados_neste_frame = {}
    
    for box_atual, w_abs, h_abs, cls_id, label_atual, conf_atual in objetos_atuais:
        melhor_id = None
        melhor_iou = 0.3
        
        for id_antigo, (box_antiga, label_antiga) in list(objetos_no_frame_anterior.items()):
            if label_atual == label_antiga:
                iou = calcular_iou(box_atual, box_antiga)
                if iou > melhor_iou:
                    melhor_iou = iou
                    melhor_id = id_antigo
                    
        # Mantém a consistência com o index do teu frame nativo
        mot_frame = frame_num 
        bb_left = box_atual[0]
        bb_top = box_atual[1]
        
        # Forçar classe a 1 se for para o benchmark padrão de Pedestrian do TrackEval
        classe_mot = 1 
        
        if melhor_id is not None:
            linhas_unificadas[melhor_id].append([mot_frame, melhor_id, bb_left, bb_top, w_abs, h_abs, conf_atual, classe_mot])
            objetos_associados_neste_frame[melhor_id] = (box_atual, label_atual)
            if melhor_id in objetos_no_frame_anterior:
                del objetos_no_frame_anterior[melhor_id]
        else:
            linhas_unificadas[next_id] = [[mot_frame, next_id, bb_left, bb_top, w_abs, h_abs, conf_atual, classe_mot]]
            objetos_associados_neste_frame[next_id] = (box_atual, label_atual)
            next_id += 1
            
    objetos_no_frame_anterior = objetos_associados_neste_frame

# Juntar todas as linhas para ordenar por Frame
linhas_finais_mot = []
for t_id, boxes in linhas_unificadas.items():
    for box in boxes:
        linhas_finais_mot.append(box)

# Ordenar primeiro pelo número do frame, depois pelo ID do objeto
linhas_finais_mot.sort(key=lambda x: (x[0], x[1]))

# Escrever o ficheiro de texto final no formato padrão MOT Challenge
with open("det.txt", "w", encoding="utf-8") as f:
    for linha in linhas_finais_mot:
        f.write(f"{linha[0]},{linha[1]},{linha[2]:.2f},{linha[3]:.2f},{linha[4]:.2f},{linha[5]:.2f},{linha[6]:.4f},{linha[7]},-1,-1\n")

print("Ficheiro 'det.txt' gerado com sucesso a partir dos frames estruturados!")