"""
Model Evaluation and Inference Script
Children's Stories Multi-Task Classification

This script provides comprehensive evaluation metrics and inference capabilities
for the trained multi-task classification model.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_recall_fscore_support, roc_curve, auc
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from children_stories_classification import MultiTaskBERT, StoriesDataset, Config

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path, encoders_path, config):
        self.config = config
        self.device = config.DEVICE
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Initialize and load model
        self.model = MultiTaskBERT(
            model_name=config.MODEL_NAME,
            num_age_groups=len(self.encoders['age_group'].classes_),
            num_severities=len(self.encoders['safety_severity'].classes_),
            num_safety_types=len(self.encoders['safety_type_mlb'].classes_),
            num_bias_types=len(self.encoders['bias_type'].classes_)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict_batch(self, data_loader):
        """Make predictions on a batch of data"""
        all_predictions = {
            'safety_binary': [],
            'safety_severity': [],
            'safety_type': [],
            'bias_binary': [],
            'bias_type': [],
            'age_group': []
        }
        all_targets = {
            'safety_binary': [],
            'safety_severity': [],
            'safety_type': [],
            'bias_binary': [],
            'bias_type': [],
            'age_group': []
        }
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Collect predictions
                all_predictions['safety_binary'].extend(
                    outputs['safety_binary'].cpu().numpy()
                )
                all_predictions['safety_severity'].extend(
                    torch.argmax(outputs['safety_severity'], dim=1).cpu().numpy()
                )
                all_predictions['safety_type'].extend(
                    outputs['safety_type'].cpu().numpy()
                )
                all_predictions['bias_binary'].extend(
                    outputs['bias_binary'].cpu().numpy()
                )
                all_predictions['bias_type'].extend(
                    torch.argmax(outputs['bias_type'], dim=1).cpu().numpy()
                )
                all_predictions['age_group'].extend(
                    torch.argmax(outputs['age_group'], dim=1).cpu().numpy()
                )
                
                # Collect targets
                for labels in batch['labels']:
                    all_targets['safety_binary'].append(labels['safety_binary'])
                    all_targets['safety_severity'].append(labels['safety_severity'])
                    all_targets['safety_type'].append(labels['safety_type'])
                    all_targets['bias_binary'].append(labels['bias_binary'])
                    all_targets['bias_type'].append(labels['bias_type'])
                    all_targets['age_group'].append(labels['age_group'])
        
        return all_predictions, all_targets
    
    def evaluate_binary_classification(self, predictions, targets, task_name):
        """Evaluate binary classification task"""
        pred_binary = (np.array(predictions) > 0.5).astype(int)
        target_binary = np.array(targets).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(target_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary'
        )
        
        try:
            auc_score = roc_auc_score(target_binary, predictions)
        except:
            auc_score = 0.0
        
        print(f"\n{task_name} - Binary Classification:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(target_binary, pred_binary)
        print(f"Confusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm
        }
    
    def evaluate_multiclass_classification(self, predictions, targets, task_name, encoder):
        """Evaluate multi-class classification task"""
        pred_classes = np.array(predictions).astype(int)
        target_classes = np.array(targets).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(target_classes, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_classes, pred_classes, average='weighted'
        )
        
        print(f"\n{task_name} - Multi-class Classification:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Classification report
        class_names = encoder.classes_
        print(f"\nClassification Report:")
        print(classification_report(
            target_classes, pred_classes, 
            target_names=class_names, zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(target_classes, pred_classes)
        print(f"Confusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_names': class_names
        }
    
    def evaluate_multilabel_classification(self, predictions, targets, task_name):
        """Evaluate multi-label classification task"""
        pred_binary = (np.array(predictions) > 0.5).astype(int)
        target_binary = np.array(targets).astype(int)
        
        # Calculate metrics for each label
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='micro'
        )
        
        print(f"\n{task_name} - Multi-label Classification:")
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"Micro F1-Score: {f1:.4f}")
        
        # Per-label metrics
        label_names = self.encoders['safety_type_mlb'].classes_
        for i, label in enumerate(label_names):
            if target_binary[:, i].sum() > 0:  # Only evaluate labels that exist
                label_precision = precision_recall_fscore_support(
                    target_binary[:, i], pred_binary[:, i], average='binary'
                )[0]
                label_recall = precision_recall_fscore_support(
                    target_binary[:, i], pred_binary[:, i], average='binary'
                )[1]
                label_f1 = precision_recall_fscore_support(
                    target_binary[:, i], pred_binary[:, i], average='binary'
                )[2]
                print(f"  {label}: P={label_precision:.3f}, R={label_recall:.3f}, F1={label_f1:.3f}")
        
        return {
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1
        }
    
    def evaluate_all_tasks(self, data_loader):
        """Evaluate all tasks comprehensively"""
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        predictions, targets = self.predict_batch(data_loader)
        
        results = {}
        
        # Safety Binary Classification
        results['safety_binary'] = self.evaluate_binary_classification(
            predictions['safety_binary'], targets['safety_binary'], "Safety Violation Detection"
        )
        
        # Bias Binary Classification
        results['bias_binary'] = self.evaluate_binary_classification(
            predictions['bias_binary'], targets['bias_binary'], "Bias Detection"
        )
        
        # Safety Severity Classification
        results['safety_severity'] = self.evaluate_multiclass_classification(
            predictions['safety_severity'], targets['safety_severity'], 
            "Safety Severity Classification", self.encoders['safety_severity']
        )
        
        # Bias Type Classification
        results['bias_type'] = self.evaluate_multiclass_classification(
            predictions['bias_type'], targets['bias_type'], 
            "Bias Type Classification", self.encoders['bias_type']
        )
        
        # Age Group Classification
        results['age_group'] = self.evaluate_multiclass_classification(
            predictions['age_group'], targets['age_group'], 
            "Age Group Classification", self.encoders['age_group']
        )
        
        # Safety Type Multi-label Classification
        results['safety_type'] = self.evaluate_multilabel_classification(
            predictions['safety_type'], targets['safety_type'], 
            "Safety Type Classification"
        )
        
        return results
    
    def plot_confusion_matrices(self, results, save_path="confusion_matrices.png"):
        """Plot confusion matrices for all classification tasks"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices for All Tasks', fontsize=16)
        
        # Safety Binary
        sns.heatmap(results['safety_binary']['confusion_matrix'], 
                   annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Safety Violation Detection')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Bias Binary
        sns.heatmap(results['bias_binary']['confusion_matrix'], 
                   annot=True, fmt='d', ax=axes[0,1], cmap='Blues')
        axes[0,1].set_title('Bias Detection')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # Safety Severity
        sns.heatmap(results['safety_severity']['confusion_matrix'], 
                   annot=True, fmt='d', ax=axes[0,2], cmap='Blues')
        axes[0,2].set_title('Safety Severity')
        axes[0,2].set_xlabel('Predicted')
        axes[0,2].set_ylabel('Actual')
        
        # Bias Type
        sns.heatmap(results['bias_type']['confusion_matrix'], 
                   annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
        axes[1,0].set_title('Bias Type')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # Age Group
        sns.heatmap(results['age_group']['confusion_matrix'], 
                   annot=True, fmt='d', ax=axes[1,1], cmap='Blues')
        axes[1,1].set_title('Age Group')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        # Remove the last subplot
        axes[1,2].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_story(self, story_text):
        """Make predictions for a single story"""
        # Tokenize
        encoding = self.tokenizer(
            story_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Process outputs
        results = {}
        
        # Safety binary
        safety_prob = outputs['safety_binary'].cpu().numpy()[0][0]
        results['safety_violation'] = {
            'present': safety_prob > 0.5,
            'confidence': float(safety_prob)
        }
        
        # Safety severity
        severity_probs = torch.softmax(outputs['safety_severity'], dim=1).cpu().numpy()[0]
        severity_idx = np.argmax(severity_probs)
        results['safety_severity'] = {
            'predicted': self.encoders['safety_severity'].classes_[severity_idx],
            'confidence': float(severity_probs[severity_idx]),
            'all_probabilities': {
                self.encoders['safety_severity'].classes_[i]: float(prob) 
                for i, prob in enumerate(severity_probs)
            }
        }
        
        # Safety type (multi-label)
        safety_type_probs = outputs['safety_type'].cpu().numpy()[0]
        predicted_types = []
        for i, prob in enumerate(safety_type_probs):
            if prob > 0.5:
                predicted_types.append({
                    'type': self.encoders['safety_type_mlb'].classes_[i],
                    'confidence': float(prob)
                })
        results['safety_types'] = predicted_types
        
        # Bias binary
        bias_prob = outputs['bias_binary'].cpu().numpy()[0][0]
        results['bias'] = {
            'present': bias_prob > 0.5,
            'confidence': float(bias_prob)
        }
        
        # Bias type
        bias_type_probs = torch.softmax(outputs['bias_type'], dim=1).cpu().numpy()[0]
        bias_type_idx = np.argmax(bias_type_probs)
        results['bias_type'] = {
            'predicted': self.encoders['bias_type'].classes_[bias_type_idx],
            'confidence': float(bias_type_probs[bias_type_idx]),
            'all_probabilities': {
                self.encoders['bias_type'].classes_[i]: float(prob) 
                for i, prob in enumerate(bias_type_probs)
            }
        }
        
        # Age group
        age_probs = torch.softmax(outputs['age_group'], dim=1).cpu().numpy()[0]
        age_idx = np.argmax(age_probs)
        results['age_group'] = {
            'predicted': self.encoders['age_group'].classes_[age_idx],
            'confidence': float(age_probs[age_idx]),
            'all_probabilities': {
                self.encoders['age_group'].classes_[i]: float(prob) 
                for i, prob in enumerate(age_probs)
            }
        }
        
        return results

def analyze_misclassifications(evaluator, data_loader, df, num_examples=5):
    """Analyze misclassified examples for insights"""
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    predictions, targets = evaluator.predict_batch(data_loader)
    
    # Safety violation misclassifications
    safety_pred = (np.array(predictions['safety_binary']) > 0.5).astype(int)
    safety_target = np.array(targets['safety_binary']).astype(int)
    safety_misclass = np.where(safety_pred != safety_target)[0]
    
    print(f"\nSafety Violation Misclassifications: {len(safety_misclass)}")
    if len(safety_misclass) > 0:
        print("Examples of misclassified stories:")
        for i, idx in enumerate(safety_misclass[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {'Violation' if safety_pred[idx] else 'No Violation'}")
            print(f"Actual: {'Violation' if safety_target[idx] else 'No Violation'}")
            # Note: You'd need to map back to original stories for full analysis
    
    # Age group misclassifications
    age_pred = np.array(predictions['age_group']).astype(int)
    age_target = np.array(targets['age_group']).astype(int)
    age_misclass = np.where(age_pred != age_target)[0]
    
    print(f"\nAge Group Misclassifications: {len(age_misclass)}")
    if len(age_misclass) > 0:
        age_encoder = evaluator.encoders['age_group']
        print("Examples of misclassified age groups:")
        for i, idx in enumerate(age_misclass[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {age_encoder.classes_[age_pred[idx]]}")
            print(f"Actual: {age_encoder.classes_[age_target[idx]]}")

def create_evaluation_report(results, save_path="evaluation_report.txt"):
    """Create a comprehensive evaluation report"""
    with open(save_path, 'w') as f:
        f.write("CHILDREN'S STORIES CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary metrics
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Safety Violation Detection F1: {results['safety_binary']['f1']:.4f}\n")
        f.write(f"Safety Severity Classification Accuracy: {results['safety_severity']['accuracy']:.4f}\n")
        f.write(f"Bias Detection F1: {results['bias_binary']['f1']:.4f}\n")
        f.write(f"Bias Type Classification Accuracy: {results['bias_type']['accuracy']:.4f}\n")
        f.write(f"Age Group Classification Accuracy: {results['age_group']['accuracy']:.4f}\n")
        f.write(f"Safety Type Multi-label F1: {results['safety_type']['micro_f1']:.4f}\n")
        
        f.write("\n\nDETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        for task, metrics in results.items():
            f.write(f"\n{task.upper()}:\n")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix' and metric != 'class_names':
                    f.write(f"  {metric}: {value}\n")

def main():
    """Main evaluation pipeline"""
    config = Config()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path="children_stories_model.pth",
        encoders_path="encoders.pkl",
        config=config
    )
    
    # Load test data (you'll need to create this from your data preparation)
    # For now, let's create a simple example
    print("Model loaded successfully!")
    
    # Example single story prediction
    example_story = """
    Max and Tommy were playing in the sandbox. Tommy pushed Max and they started fighting.
    The teacher had to separate them. Emma said she didn't want to play because 
    'girls shouldn't get dirty in the sandbox.'
    """
    
    print("\nExample Prediction:")
    print("Story:", example_story)
    print("\nPredictions:")
    
    results = evaluator.predict_single_story(example_story)
    
    print(f"Safety Violation: {results['safety_violation']}")
    print(f"Safety Severity: {results['safety_severity']}")
    print(f"Safety Types: {results['safety_types']}")
    print(f"Bias Present: {results['bias']}")
    print(f"Bias Type: {results['bias_type']}")
    print(f"Age Group: {results['age_group']}")

if __name__ == "__main__":
    main()
