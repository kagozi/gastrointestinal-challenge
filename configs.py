import os
import torch
import warnings
warnings.filterwarnings('ignore')
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATASET_PATH = '../datasets/ECG/gastrovision-4/Gastrovision 4 class/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/gi/results/'
ENSEMBLE_PATH = '../santosh_lab/shared/KagoziA/gi/results/ensemble_results/'
CLASSES = sorted(os.listdir(f'{DATASET_PATH}/'))
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32