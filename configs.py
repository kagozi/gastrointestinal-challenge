import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, auc
)

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATASET_PATH = '/kaggle/input/gastrovision-4/Gastrovision 4 class/'
ENSEMBLE_PATH = 'ensemble_results'
CLASSES = sorted(os.listdir(f'{DATASET_PATH}/'))
NUM_CLASSES = len(CLASSES)
print(f"Classes:{CLASSES}, Number of classes: {NUM_CLASSES}")