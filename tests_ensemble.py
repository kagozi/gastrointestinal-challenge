import os
import numpy as np
from configs import ENSEMBLE_PATH
from utils import plot_confusion_matrix, plot_roc_curves, compute_all_metrics

def create_ensemble(model_results, y_true, classes, method='average', top_k=None):
    """
    Create ensemble from multiple models
    Args:
        model_results: dict with model results
        y_true: true labels
        classes: class names
        method: 'average', 'weighted', or 'max'
        top_k: use only top k models (by macro_auc)
    """
    print(f"\n{'='*60}")
    print(f"Creating Ensemble (method={method}, top_k={top_k})")
    print(f"{'='*60}")
    
    if top_k:
        sorted_models = sorted(model_results.items(), 
                             key=lambda x: x[1]['metrics']['macro_auc'], 
                             reverse=True)[:top_k]
        selected_names = [n for n, _ in sorted_models]
        print(f"Top {top_k} models selected:")
        for name, result in sorted_models:
            print(f"  - {name}: AUC={result['metrics']['macro_auc']:.4f}")
    else:
        selected_names = list(model_results.keys())
        print(f"Using all {len(selected_names)} models")
    
    # Collect predictions and scores
    all_scores = [model_results[n]['y_scores'] for n in selected_names]
    
    # Ensemble scoring
    if method == 'average':
        ensemble_scores = np.mean(all_scores, axis=0)
    elif method == 'weighted':
        weights = np.array([model_results[n]['metrics']['macro_auc'] for n in selected_names])
        weights /= weights.sum()
        print(f"Model weights: {dict(zip(selected_names, weights))}")
        ensemble_scores = np.average(all_scores, axis=0, weights=weights)
    elif method == 'max':
        ensemble_scores = np.max(all_scores, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Make predictions
    ensemble_pred = ensemble_scores.argmax(axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, ensemble_pred, ensemble_scores)
    
    print(f"\nEnsemble Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    
    # Save visualizations
    ensemble_name = f"Ensemble-{method}" + (f"-Top{top_k}" if top_k else "-All")
    
    cm_path = os.path.join(ENSEMBLE_PATH, f"confusion_{ensemble_name}.png")
    plot_confusion_matrix(metrics['confusion_matrix'], classes, 
                         model_name=ensemble_name, save_path=cm_path)
    
    roc_path = os.path.join(ENSEMBLE_PATH, f"roc_{ensemble_name}.png")
    plot_roc_curves(y_true, ensemble_scores, classes, 
                   model_name=ensemble_name, save_path=roc_path)
    
    return {
        'metrics': metrics,
        'name': ensemble_name,
        'y_scores': ensemble_scores,
        'y_pred': ensemble_pred,
    }