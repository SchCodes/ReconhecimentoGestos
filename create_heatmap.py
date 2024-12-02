import numpy as np
import cv2
import os

# Função para criar um mapa de calor a partir dos landmarks
def create_heatmap(landmarks, height=224, width=224, sigma=1.0):
    """
    Cria um mapa de calor a partir de landmarks com distribuição gaussiana.
    """
    # Criação de uma grade 2D com coordenadas x e y
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Inicializar o mapa de calor
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Para cada ponto, adicionar a contribuição gaussiana ao mapa de calor
    for (x, y, _) in landmarks:
        # Normalizar as coordenadas para o tamanho da imagem
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        
        # Garantir que as coordenadas estejam dentro do limite
        x_pixel = np.clip(x_pixel, 0, width - 1)
        y_pixel = np.clip(y_pixel, 0, height - 1)
        
        # Cálculo da distribuição gaussiana em uma matriz inteira
        dist_squared = (x_coords - x_pixel)**2 + (y_coords - y_pixel)**2
        heatmap += np.exp(-dist_squared / (2 * sigma**2))
    
    # Normalizar o mapa de calor para a faixa [0, 1]
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

# Função para salvar mapas de calor em uma pasta
def save_heatmaps(heatmaps, output_dir, prefix="heatmap"):
    """
    Salva uma lista de mapas de calor como imagens PNG.
    """
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório, se não existir
    for i, heatmap in enumerate(heatmaps):
        file_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        cv2.imwrite(file_path, (heatmap * 255).astype(np.uint8))

# Parâmetros do dataset
landmarks_file = "data/raw/landmarks/positive/positive_landmarks.npy"
output_dir = "data/processed/categorias/positive"
nome_classe = "positive"

# Carregar os dados de landmarks
landmarks = np.load(landmarks_file)

# Lista para armazenar os mapas de calor
heatmaps = []

# Iterar sobre os frames de landmarks
for idx, frame_landmarks in enumerate(landmarks):
    # Gerar o mapa de calor para o frame atual
    heatmap = create_heatmap(frame_landmarks, height=224, width=224, sigma=2.5)
    heatmaps.append(heatmap)  # Adicionar o mapa de calor à lista
    
    # Salvar o mapa de calor diretamente
    save_heatmaps([heatmap], output_dir, prefix=f"{nome_classe}_{idx}")
