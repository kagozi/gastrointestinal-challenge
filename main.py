import os
import torch
import numpy as np
import json
from train_models import train_single_model
from torch.utils.data import DataLoader
from config import ENSEMBLE_PATH, CLASSES, DATASET_PATH, RESULTS_PATH, BATCH_SIZE, device, configs
from utils import GIDataset, train_transform, val_transform
from tests_ensemble import create_ensemble
def main():
    """Main function to train multiple models and create ensembles"""
    os.makedirs(ENSEMBLE_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    # Reduce num_workers if experiencing hangs
    num_workers = 2 if torch.cuda.is_available() else 0
    train_dataset = GIDataset(DATASET_PATH, transform=train_transform, mode='train')
    val_dataset = GIDataset(DATASET_PATH, transform=val_transform, mode='val')
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, 
                             persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True if num_workers > 0 else False)

    # Get true labels for ensemble
    val_labels = np.array(val_dataset.labels)
    
    # Train all models
    model_results = {}
    for cfg in configs:
        result = train_single_model(cfg, train_loader, val_loader, device, val_labels)
        model_results[cfg['name']] = result
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Macro F1':<12} {'Macro AUC':<12}")
    print("-"*80)
    for name, result in model_results.items():
        m = result['metrics']
        print(f"{name:<25} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} {m['macro_auc']:<12.4f}")
    print("="*80)
    
    # Create ensembles
    ensemble_configs = [
        {'method': 'average', 'top_k': None},
        {'method': 'weighted', 'top_k': None},
        {'method': 'weighted', 'top_k': 3},
    ]
    
    ensemble_results = {}
    for ens_config in ensemble_configs:
        ens_result = create_ensemble(model_results, val_labels, CLASSES, 
                                     method=ens_config['method'], 
                                     top_k=ens_config['top_k'])
        ensemble_results[ens_result['name']] = ens_result
    
    # Final comparison with ensembles
    print("\n" + "="*80)
    print("FINAL COMPARISON (Including Ensembles)")
    print("="*80)
    print(f"{'Model/Ensemble':<30} {'Accuracy':<12} {'Macro F1':<12} {'Macro AUC':<12}")
    print("-"*80)
    
    all_results = {**{k: v for k, v in model_results.items()}, 
                   **{k: v for k, v in ensemble_results.items()}}
    
    for name, result in sorted(all_results.items(), 
                              key=lambda x: x[1]['metrics']['macro_auc'], 
                              reverse=True):
        m = result['metrics']
        print(f"{name:<30} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} {m['macro_auc']:<12.4f}")
    print("="*80)
    
    # Save results to JSON
    results_summary = {
        name: {
            'accuracy': float(result['metrics']['accuracy']),
            'f1_macro': float(result['metrics']['f1_macro']),
            'macro_auc': float(result['metrics']['macro_auc']),
        }
        for name, result in all_results.items()
    }
    
    with open(f'{RESULTS_PATH}model_comparison_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to model_comparison_results.json")
    
    return model_results, ensemble_results


if __name__ == "__main__":
    model_results, ensemble_results = main()