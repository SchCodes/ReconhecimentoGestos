# Reconhecimento de Gestos (Hand Gesture Recognition)

Pipeline completo para:
1. Coleta de dados (imagens + landmarks da mão via MediaPipe)
2. Geração de heatmaps a partir dos landmarks
3. Pré-processamento e augmentação de imagens
4. Treinamento de uma CNN simples em TensorFlow/Keras
5. Detecção em tempo real via webcam

## Sumário
- Visão Geral
- Arquitetura do Projeto
- Estrutura de Pastas
- Dependências e Ambiente
- Fluxo de Execução
- Scripts em Detalhe
- Treinamento do Modelo
- Execução em Tempo Real
- Qualidade dos Dados e Boas Prática
- Contribuição

---

## Visão Geral

O objetivo deste repositório é construir um classificador de gestos de mão usando:
- MediaPipe para extração de landmarks
- CNN para classificação baseada em imagens (originais + heatmaps)
- Pipeline simples de coleta → processamento → treinamento → inferência

Categorias atuais que o código de detecção em tempo real espera:
```
['positivo', 'indicador', 'shaka', 'punho', 'rock']
```
(Adapte se acrescentar novas classes; é fundamental manter consistência entre nomes de diretórios e lista usada na inferência.)

---

## Arquitetura do Projeto

Fluxo simplificado:

```
Webcam --> collect_data.py
    |--> Imagens (data/raw/categorias/<classe>/*.jpg)
    |--> Landmarks (data/raw/landmarks/<classe>/<classe>_landmarks.npy)

Landmarks --> preprocess_data.py (create_heatmap + save_heatmaps)
Imagens   --> preprocess_data.py (resize + normalização + flip)
         --> data/processed/categorias/<classe>/*.jpg

Processed Images --> train_model.py --> models/cnn_model.keras

models/cnn_model.keras + Webcam --> real_time_detection.py
```

---

## Estrutura de Pastas (esperada)

Após executar coleta e pré-processamento:

```
ReconhecimentoGestos/
  collect_data.py
  preprocess_data.py
  train_model.py
  real_time_detection.py
  requirements.txt
  README.md
  data/
    raw/
      categorias/
        positivo/
          positivo_0.jpg
          ...
        indicador/
        shaka/
        punho/
        rock/
      landmarks/
        positivo/
          positivo_landmarks.npy
        indicador/
        ...
    processed/
      categorias/
        positivo/
          positivo_0.jpg
          positivo_0_fliped.jpg
          positivo_0_heatmap.jpg (se optar integrar no pipeline)
        indicador/
        ...
  models/
    cnn_model.keras
```

Observação: Atualmente os heatmaps são salvos junto de cada classe em data/processed/categorias/<classe> (prefixos <classe>_idx e <classe>_idx_fliped).

---

## Dependências e Ambiente

```
opencv-python==4.10.0.84
mediapipe==0.10.18
tensorflow==2.18.0
numpy==1.26.4
```

Criação de ambiente:
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Fluxo de Execução (Passo a Passo)

1. Definir classes de gestos (ex.: positivo, indicador, shaka, punho, rock).
2. Coletar dados (imagens + landmarks) com collect_data.py.
3. Pré-processar:
   - Imagens: resize + normalização + flip.
   - Landmarks: gerar heatmaps.
4. Treinar CNN com train_model.py.
5. Testar detecção em tempo real com real_time_detection.py.
6. Ajustar hiperparâmetros / aumentar dataset / melhorar modelo.

---

## Scripts em Detalhe

### collect_data.py
- Abre webcam (cv2.VideoCapture(0)).
- Usa MediaPipe Hands (max_num_hands=1).
- Captura imagens em intervalos definidos por photos_per_second.
- Salva:
  - Imagens JPEG em data/raw/categorias/<classe>
  - Landmarks como arquivo .npy em data/raw/landmarks/<classe>/<classe>_landmarks.npy

### preprocess_data.py
Partes:
- preprocess_image: resize para (224,224) + normalização [0,1].
- augment_image: flip horizontal.
- preprocess_and_augment: aplica a todas as imagens de uma pasta.
- create_heatmap: gera mapa de calor gaussiano (sigma configurável) a partir de landmarks (x,y normalizados).
- preprocess_landmarks: lê .npy e gera heatmaps + versão flipada.

### train_model.py
- Cria uma CNN simples:
  - Conv2D(32) → MaxPool
  - Conv2D(64) → MaxPool
  - Conv2D(128) → MaxPool
  - Flatten → Dense(128) → Dense(num_classes, softmax)
- Usa image_dataset_from_directory (sem split de validação).
- Treina por 1 época (epochs=1; mude para maior).
- Salva modelo em models/cnn_model.keras.

Execução:
```bash
python train_model.py
# Ou editar final para mudar diretório ou epochs
```
Sugestão para validação:
```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,
    image_size=(224,224),
    validation_split=0.2,
    subset="training",
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,
    image_size=(224,224),
    validation_split=0.2,
    subset="validation",
    seed=42
)
```

### real_time_detection.py
- Carrega modelo salvo.
- Lista fixa de categorias (ajuste se mudar nomes).
- Captura da webcam + MediaPipe.
- Se houver mão detectada:
  - Desenha landmarks.
  - Usa frame inteiro (não recorta mão) → preprocess_image → inferência.
  - Exibe texto: Prediction: <classe>.

Execução:
```bash
python real_time_detection.py
# Pressione 'q' para sair
```

---

## Treinamento do Modelo (Recomendações)

Ajuste:
- epochs: 10–50 (dependendo do dataset).
- data augmentation adicional:
  - brightness/contrast
  - pequena rotação
  - zoom leve

---

## Execução em Tempo Real (Sugestões)

1. Verifique iluminação consistente.
2. Fundo neutro melhora a distinção.
3. Ajuste lista de categorias no topo de real_time_detection.py para refletir diretórios em data/processed/categorias.
4. Para menor latência, reduza resolução da captura:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## Qualidade dos Dados e Boas Práticas

- Variabilidade: colete gestos em diferentes ângulos, distâncias e mãos (direita/esquerda).
- Balanceamento: número semelhante de imagens por classe.
- Consistência: defina exatamente o formato do gesto (ex.: “positivo” = polegar para cima).
- Evitar overfitting: não treinar só com seu próprio fundo/mesa.
- Sementes para reprodutibilidade:
```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```
---

## Contribuição

1. Abra uma issue descrevendo sugestão ou bug.
2. Faça fork.
3. Crie branch: feature/nome-do-recurso.
4. Commit mensagens claras (Português ou Inglês).
5. Pull Request explicando motivação e teste manual.

Padrão de código:
- PEP8
- Comentários em português consistentes
- Docstrings curtas nas funções principais

---

## Aviso de Uso de Imagem

Se for coletar gestos de outras pessoas, informe a finalidade dos dados e não publique imagens sem consentimento.

---

## Referências Úteis

- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- TensorFlow ImageDataset: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
- Data augmentation (tf.image): https://www.tensorflow.org/api_docs/python/tf/image

---

## Contato

Dúvidas, ideias ou melhorias: abra uma issue ou envie uma mensagem.