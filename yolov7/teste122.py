import cv2
import os

# 1. Configuração dos caminhos
video_path = "teste2.mp4"  # Coloca aqui o nome do teu vídeo
output_folder = "sequencia_frames"       # Nome da pasta onde vão ficar as imagens

# Criar a pasta de saída se ela não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Abrir o ficheiro de vídeo
cap = cv2.getStructuringElement if not cv2.VideoCapture(video_path) else cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

frame_idx = 1

print("A extrair frames... Por favor, aguarda.")

# 3. Loop para ler frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # O vídeo terminou
    
    # Define o nome do ficheiro (ex: sequencia_frames/000001.jpg)
    # O :06d garante que o nome tem sempre 6 dígitos com zeros à esquerda
    frame_name = f"{frame_idx:06d}.jpg"
    frame_path = os.path.join(output_folder, frame_name)
    
    # Guardar a imagem no disco
    cv2.imwrite(frame_path, frame)
    
    frame_idx += 1

# Fechar o vídeo
cap.release()
print(f"Concluído! {frame_idx - 1} frames extraídos e guardados em '{output_folder}'.")