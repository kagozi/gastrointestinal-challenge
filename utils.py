from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from configs import CLASSES, NUM_CLASSES
# ========================
# Dataset Class
# ========================
class GIDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.labels = []
        
        # Load images from each class folder
        for idx, class_name in enumerate(CLASSES):
            class_path = self.root_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append(str(img_path))
                        self.labels.append(idx)
        
        print(f"{mode} set: {len(self.samples)} images across {NUM_CLASSES} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# ========================
# Data Transforms
# ========================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_class_weights(dataset, num_classes):
    """Compute class weights for imbalanced datasets"""
    class_counts = np.bincount(dataset.labels, minlength=num_classes)
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights)


# ========================
# VISUALIZATION FUNCTIONS WITH MODEL NAME
# ========================
def plot_confusion_matrix(cm, classes, model_name='', save_path='confusion_matrix.png'):
    """Plot and save confusion matrix with model name"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    title = f'Confusion Matrix - {model_name}' if model_name else 'Confusion Matrix'
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curves(y_true, y_probs, classes, model_name='', save_path='roc_curves.png'):
    """Plot ROC curves with model name"""
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for i, color in zip(range(NUM_CLASSES), colors):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{classes[i]} (AUC = {roc_auc:.3f})')
        except:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = f'ROC Curves - {model_name}' if model_name else 'ROC Curves - Per Class'
    plt.title(title, fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")

def plot_training_history(history, model_name='', save_path='training_history.png'):
    """Plot training history with model name"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    title = f'Loss - {model_name}' if model_name else 'Training and Validation Loss'
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    title = f'Accuracy - {model_name}' if model_name else 'Training and Validation Accuracy'
    ax2.set_title(title, fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")
    
    
def compute_all_metrics(y_true, y_pred, y_scores):
    """Compute comprehensive metrics"""
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    # AUROC per class
    auroc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auroc = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
            auroc_scores.append(auroc)
        except:
            auroc_scores.append(0.0)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'macro_auc': np.mean(auroc_scores),
        'auroc_per_class': auroc_scores,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics