import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from configs import CLASSES, NUM_CLASSES

def train_epoch(model, dataloader, criterion, optimizer, device, model_name=''):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Training {model_name}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(dataloader), 'acc': 100.*correct/total})
    
    return running_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device, model_name=''):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Validation {model_name}'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (running_loss/len(dataloader), 100.*correct/total, 
            np.array(all_preds), np.array(all_labels), np.array(all_probs))
    
# ========================
# Evaluation Metrics
# ========================
def evaluate_model(y_true, y_pred, y_probs):
    """Comprehensive evaluation with all metrics"""
    
    # Use argmax for predictions (standard multi-class approach)
    y_pred = y_probs.argmax(axis=1)
    
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # 2. Macro F1-Score
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\nMACRO F1-SCORE: {macro_f1:.4f}")
    
    # 3. Per-class Precision & Recall
    print(f"\n{'='*60}")
    print("PER-CLASS PRECISION & RECALL")
    print(f"{'='*60}")
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)
    
    # 4. AUROC per class
    print(f"\n{'='*60}")
    print("AUROC PER CLASS")
    print(f"{'='*60}")
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    auroc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auroc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            auroc_scores.append(auroc)
            print(f"{CLASSES[i]}: {auroc:.4f}")
        except:
            auroc_scores.append(0.0)
            print(f"{CLASSES[i]}: N/A (insufficient samples)")
    
    print(f"\nMean AUROC: {np.mean(auroc_scores):.4f}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auroc_scores': auroc_scores,
        'confusion_matrix': cm,
        'classification_report': report
    }