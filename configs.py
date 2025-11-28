import os
import torch
import warnings
warnings.filterwarnings('ignore')
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATASET_PATH = '../datasets/gi/Gastrovision 4 class/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/gi/results/'
ENSEMBLE_PATH = '../santosh_lab/shared/KagoziA/gi/results/ensemble_results/'
CLASSES = sorted(os.listdir(f'{DATASET_PATH}/'))
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 16
NUM_EPOCHS = 30

# Define model configurations
configs = [
    {'model': 'ResNet50', 'name': 'ResNet50-CE', 'loss': 'ce'},
    {'model': 'ResNet50', 'name': 'ResNet50-Focal', 'loss': 'focal'},
    {'model': 'EfficientNet', 'name': 'EfficientNet-Focal', 'loss': 'focal'},
    {'model': 'EfficientNet', 'name': 'EfficientNet-CE', 'loss': 'ce'},
    {'model': 'ViT', 'name': 'ViT-CE', 'loss': 'ce'},
    {'model': 'ViT', 'name': 'ViT-Focal', 'loss': 'focal'},
    {'model': 'SwinTransformer', 'name': 'Swin-CE', 'loss': 'ce'},
    {'model': 'SwinTransformer', 'name': 'Swin-Focal', 'loss': 'focal'},
]