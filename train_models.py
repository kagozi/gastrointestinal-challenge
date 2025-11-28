import numpy as np
import torch
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from config import CLASSES, NUM_CLASSES, RESULTS_PATH, NUM_EPOCHS
from models import GIClassifier
from focal_loss import FocalLoss
from utils import (
    plot_confusion_matrix, plot_roc_curves, 
    compute_all_metrics, compute_class_weights, plot_training_history
)

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
    
    # Close progress bar explicitly
    pbar.close()
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
# ========================
# 6. SINGLE MODEL TRAINING FUNCTION
# ========================
def train_single_model(config, train_loader, val_loader, device, val_labels):
    """Train a single model with given configuration"""
    model_name = config['name']
    model_type = config['model']
    loss_type = config['loss']
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Model: {model_type} | Loss: {loss_type}")
    print(f"{'='*60}")
    
    # Model-specific parameters
    if 'resnet' in model_type.lower():
        timm_name = 'resnet50'
        lr = 0.001
    elif 'efficientnet' in model_type.lower():
        timm_name = 'efficientnet_b3'
        lr = 0.0005
    elif 'vit' in model_type.lower():
        timm_name = 'vit_base_patch16_224'
        lr = 0.0001
    elif 'swin' in model_type.lower():
        timm_name = 'swin_base_patch4_window7_224'
        lr = 0.0001
    else:
        timm_name = 'resnet50'
        lr = 0.001
    
    # Initialize model
    model = GIClassifier(model_name=timm_name, num_classes=NUM_CLASSES, pretrained=True).to(device)
    
   # Loss function
    if loss_type == 'focal':
        # Check class distribution
        class_counts = np.bincount(train_loader.dataset.labels, minlength=NUM_CLASSES)
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        print(f"Class distribution: {class_counts}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Use class weights only if severe imbalance (>3:1 ratio)
        if imbalance_ratio > 3.0:
            class_weights = compute_class_weights(train_loader.dataset, NUM_CLASSES).to(device)
            criterion = FocalLoss(alpha=class_weights, gamma=2)
            print(f"Using Focal Loss WITH class weights (severe imbalance detected)")
            print(f"Class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = FocalLoss(alpha=None, gamma=2)
            print(f"Using Focal Loss WITHOUT class weights (balanced enough)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = f'best_{model_name}.pth'
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, model_name)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels_batch, val_probs = validate(
            model, val_loader, criterion, device, model_name
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved: {val_acc:.2f}%")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    _, _, val_preds, _, val_probs = validate(model, val_loader, criterion, device, model_name)
        
    final_preds = val_probs.argmax(axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(val_labels, final_preds, val_probs)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    
    # Save visualizations
    output_dir = f'{RESULTS_PATH}results_{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_confusion_matrix(metrics['confusion_matrix'], CLASSES, 
                         model_name=model_name,
                         save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    plot_roc_curves(val_labels, val_probs, CLASSES,
                   model_name=model_name,
                   save_path=os.path.join(output_dir, 'roc_curves.png'))
    
    plot_training_history(history, model_name=model_name,
                         save_path=os.path.join(output_dir, 'training_history.png'))
    
    return {
        'model': model,
        'metrics': metrics,
        'y_scores': val_probs,
        'y_pred': final_preds,
        'history': history
    }