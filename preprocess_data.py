
import cv2
import os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

def augment_image(image):
    flipped = cv2.flip(image, 1)
    return [image, flipped]

def preprocess_and_augment(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
            image = preprocess_image(file_path)
            augmented_images = augment_image(image)
            for i, img in enumerate(augmented_images):
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{i}.jpg")
                cv2.imwrite(output_file, (img * 255).astype(np.uint8))

def preprocess_all_categories(input_base_dir, output_base_dir):
    for class_name in os.listdir(input_base_dir):
        input_dir = os.path.join(input_base_dir, class_name)
        output_dir = os.path.join(output_base_dir, class_name)
        if os.path.isdir(input_dir):
            preprocess_and_augment(input_dir, output_dir)

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

    #flipar o heatmap
    heatmap_fliped = cv2.flip(heatmap, 1)

    return heatmap, heatmap_fliped

# Função para salvar mapas de calor em uma pasta
def save_heatmaps(heatmaps, output_dir, prefix="heatmap"):
    """
    Salva uma lista de mapas de calor como imagens pjg.
    """
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório, se não existir
    for i, heatmap in enumerate(heatmaps):
        file_path = os.path.join(output_dir, f"{prefix}_{i}.jpg")
        cv2.imwrite(file_path, (heatmap * 255).astype(np.uint8))

def preprocess_landmarks(input_base_dir, output_base_dir):
    for class_name in os.listdir(input_base_dir):
        input_dir = os.path.join(input_base_dir, class_name)
        output_dir = os.path.join(output_base_dir, class_name)
        if os.path.isdir(input_dir):
            landmarks_file = os.path.join(input_dir, f"{class_name}_landmarks.npy")
            if os.path.exists(landmarks_file):
                landmarks = np.load(landmarks_file)
                heatmaps = []
                heatmaps_fliped = []
                for idx, frame_landmarks in enumerate(landmarks):
                    heatmap, heatmap_fliped = create_heatmap(frame_landmarks, height=224, width=224, sigma=5)
                    heatmaps.append(heatmap)
                    heatmaps_fliped.append(heatmap_fliped)
                    save_heatmaps([heatmap], output_dir, prefix=f"{class_name}_{idx}")
                    save_heatmaps([heatmap_fliped], output_dir, prefix=f"{class_name}_{idx}_fliped")

# Exemplo de uso
preprocess_all_categories("data/raw/categorias", "data/processed/categorias")
preprocess_landmarks("data/raw/landmarks", "data/processed/categorias")
