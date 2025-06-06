import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class Config:
    """Configuration class for model parameters"""
    # Model parameters
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    WARMUP_STEPS = 0.1
    
    # Data parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    DATA_PATH = "/kaggle/input/ai-generated-stories/AI.json"
    MODEL_SAVE_PATH = "/kaggle/working/children_stories_model.pth"

def load_and_explore_data(file_path):
    """Load and explore the dataset"""
    print("=" * 50)
    print("PHASE 1: DATA LOADING AND EXPLORATION")
    print("=" * 50)
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Basic statistics
    print(f"\nTotal stories: {len(df)}")
    print(f"Stories with complete data: {df.dropna(subset=['story']).shape[0]}")
    
    return df

def analyze_data_structure(df):
    """Analyze the structure of safety violations and stereotypes"""
    print("\n" + "=" * 50)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Analyze safety violations
    safety_data = []
    for idx, row in df.iterrows():
        if pd.notna(row.get('safety_violations')):
            if isinstance(row['safety_violations'], dict):
                safety_data.append({
                    'id': row.get('id'),
                    'present': row['safety_violations'].get('present', False),
                    'severity': row['safety_violations'].get('severity'),
                    'type': row['safety_violations'].get('type'),
                    'description': row['safety_violations'].get('description')
                })
    
    safety_df = pd.DataFrame(safety_data)
    
    # Analyze stereotypes/biases
    bias_data = []
    for idx, row in df.iterrows():
        if pd.notna(row.get('stereotypes_biases')):
            if isinstance(row['stereotypes_biases'], dict):
                bias_data.append({
                    'id': row.get('id'),
                    'present': row['stereotypes_biases'].get('present', False),
                    'type': row['stereotypes_biases'].get('type'),
                    'description': row['stereotypes_biases'].get('description')
                })
    
    bias_df = pd.DataFrame(bias_data)
    
    print("SAFETY VIOLATIONS:")
    print(f"Total records with safety data: {len(safety_df)}")
    if len(safety_df) > 0:
        print(f"Safety violations present: {safety_df['present'].sum()}")
        print(f"Severity distribution:")
        print(safety_df['severity'].value_counts())
        print(f"Type distribution:")
        print(safety_df['type'].value_counts())
    
    print("\nSTEREOTYPES/BIASES:")
    print(f"Total records with bias data: {len(bias_df)}")
    if len(bias_df) > 0:
        print(f"Biases present: {bias_df['present'].sum()}")
        print(f"Type distribution:")
        print(bias_df['type'].value_counts())
    
    print("\nAGE GROUPS:")
    print(df['age_group'].value_counts())
    
    return safety_df, bias_df

def clean_and_prepare_data(df):
    """Clean and prepare the dataset for modeling"""
    print("\n" + "=" * 50)
    print("PHASE 2: DATA CLEANING AND PREPARATION")
    print("=" * 50)
    
    # Filter out rows without stories
    df_clean = df[df['story'].notna() & (df['story'] != '')].copy()
    print(f"Stories after removing empty: {len(df_clean)}")
    
    # Extract safety violation features
    df_clean['safety_present'] = df_clean['safety_violations'].apply(
        lambda x: x.get('present', False) if isinstance(x, dict) else False
    )
    
    df_clean['safety_severity'] = df_clean['safety_violations'].apply(
        lambda x: x.get('severity', 'None') if isinstance(x, dict) else 'None'
    )
    
    df_clean['safety_type'] = df_clean['safety_violations'].apply(
        lambda x: x.get('type', 'none') if isinstance(x, dict) else 'none'
    )
    
    # Extract stereotype/bias features
    df_clean['bias_present'] = df_clean['stereotypes_biases'].apply(
        lambda x: x.get('present', False) if isinstance(x, dict) else False
    )
    
    df_clean['bias_type'] = df_clean['stereotypes_biases'].apply(
        lambda x: x.get('type', 'none') if isinstance(x, dict) else 'none'
    )
    
    # Clean and standardize labels
    df_clean['safety_severity'] = df_clean['safety_severity'].fillna('None').str.lower()
    df_clean['safety_type'] = df_clean['safety_type'].fillna('none').str.lower()
    df_clean['bias_type'] = df_clean['bias_type'].fillna('none').str.lower()
    
    # Handle age groups
    df_clean['age_group'] = df_clean['age_group'].fillna('unknown')
    
    print("Label distributions after cleaning:")
    print(f"Safety present: {df_clean['safety_present'].value_counts()}")
    print(f"Safety severity: {df_clean['safety_severity'].value_counts()}")
    print(f"Safety type: {df_clean['safety_type'].value_counts()}")
    print(f"Bias present: {df_clean['bias_present'].value_counts()}")
    print(f"Bias type: {df_clean['bias_type'].value_counts()}")
    print(f"Age group: {df_clean['age_group'].value_counts()}")
    
    return df_clean

def encode_labels(df):
    """Encode categorical labels for training"""
    print("\n" + "=" * 50)
    print("LABEL ENCODING")
    print("=" * 50)
    
    # Create label encoders
    encoders = {}
    
    # Age group encoding (multi-class)
    age_encoder = LabelEncoder()
    df['age_group_encoded'] = age_encoder.fit_transform(df['age_group'])
    encoders['age_group'] = age_encoder
    
    # Safety severity encoding (multi-class)
    severity_encoder = LabelEncoder()
    df['safety_severity_encoded'] = severity_encoder.fit_transform(df['safety_severity'])
    encoders['safety_severity'] = severity_encoder
    
    # Handle multi-label safety types (some entries have multiple types)
    def parse_safety_types(type_str):
        if pd.isna(type_str) or type_str in ['none', 'null', '']:
            return ['none']
        # Split by comma and clean
        types = [t.strip().lower() for t in str(type_str).split(',')]
        return [t for t in types if t]
    
    df['safety_types_list'] = df['safety_type'].apply(parse_safety_types)
    
    # Multi-label binarization for safety types
    safety_mlb = MultiLabelBinarizer()
    safety_type_encoded = safety_mlb.fit_transform(df['safety_types_list'])
    encoders['safety_type_mlb'] = safety_mlb
    
    # Bias type encoding (multi-class, treating as single label for now)
    bias_encoder = LabelEncoder()
    df['bias_type_encoded'] = bias_encoder.fit_transform(df['bias_type'])
    encoders['bias_type'] = bias_encoder
    
    print("Encoding summary:")
    print(f"Age groups: {list(age_encoder.classes_)}")
    print(f"Safety severities: {list(severity_encoder.classes_)}")
    print(f"Safety types: {list(safety_mlb.classes_)}")
    print(f"Bias types: {list(bias_encoder.classes_)}")
    
    return df, encoders, safety_type_encoded

class StoriesDataset(Dataset):
    """Custom dataset for children's stories"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert label dictionary to individual tensors
        label_dict = self.labels[idx]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'safety_binary': torch.tensor([label_dict['safety_binary']], dtype=torch.float),
            'safety_severity': torch.tensor(label_dict['safety_severity'], dtype=torch.long),
            'safety_type': torch.tensor(label_dict['safety_type'], dtype=torch.float),
            'bias_binary': torch.tensor([label_dict['bias_binary']], dtype=torch.float),
            'bias_type': torch.tensor(label_dict['bias_type'], dtype=torch.long),
            'age_group': torch.tensor(label_dict['age_group'], dtype=torch.long)
        }

class MultiTaskBERT(nn.Module):
    """Multi-task BERT model for story classification"""
    
    def __init__(self, model_name, num_age_groups, num_severities, num_safety_types, num_bias_types):
        super(MultiTaskBERT, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Task-specific heads
        # Task 1: Safety violation detection (binary)
        self.safety_binary_head = nn.Linear(hidden_size, 1)
        
        # Task 2: Safety severity classification (multi-class)
        self.safety_severity_head = nn.Linear(hidden_size, num_severities)
        
        # Task 3: Safety type classification (multi-label)
        self.safety_type_head = nn.Linear(hidden_size, num_safety_types)
        
        # Task 4: Bias detection (binary)
        self.bias_binary_head = nn.Linear(hidden_size, 1)
        
        # Task 5: Bias type classification (multi-class)
        self.bias_type_head = nn.Linear(hidden_size, num_bias_types)
        
        # Task 6: Age group classification (multi-class)
        self.age_group_head = nn.Linear(hidden_size, num_age_groups)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token) from last hidden state
        # DistilBERT doesn't have pooler_output, so we use the [CLS] token manually
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]  # [CLS] token is at position 0
        pooled_output = self.dropout(pooled_output)
        
        # Task predictions
        safety_binary = torch.sigmoid(self.safety_binary_head(pooled_output))
        safety_severity = self.safety_severity_head(pooled_output)
        safety_type = torch.sigmoid(self.safety_type_head(pooled_output))
        bias_binary = torch.sigmoid(self.bias_binary_head(pooled_output))
        bias_type = self.bias_type_head(pooled_output)
        age_group = self.age_group_head(pooled_output)
        
        return {
            'safety_binary': safety_binary,
            'safety_severity': safety_severity,
            'safety_type': safety_type,
            'bias_binary': bias_binary,
            'bias_type': bias_type,
            'age_group': age_group
        }

def prepare_labels(df, safety_type_encoded, encoders):
    """Prepare labels for multi-task learning"""
    labels = []
    
    for idx, row in df.iterrows():
        label_dict = {
            'safety_binary': float(row['safety_present']),
            'safety_severity': float(row['safety_severity_encoded']),
            'safety_type': safety_type_encoded[idx].astype(float),
            'bias_binary': float(row['bias_present']),
            'bias_type': float(row['bias_type_encoded']),
            'age_group': float(row['age_group_encoded'])
        }
        labels.append(label_dict)
    
    return labels

def create_data_loaders(df, labels, tokenizer, config):
    """Create train, validation, and test data loaders"""
    print("\n" + "=" * 50)
    print("CREATING DATA LOADERS")
    print("=" * 50)
    
    # First split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['story'], labels, test_size=config.TEST_SIZE, random_state=42, 
        stratify=df['age_group_encoded']
    )
    
    # Second split: train vs val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), 
        random_state=42, stratify=train_val_texts.map(df.set_index('story')['age_group_encoded'])
    )
    
    print(f"Train size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    # Create datasets
    train_dataset = StoriesDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = StoriesDataset(val_texts, val_labels, tokenizer, config.MAX_LENGTH)
    test_dataset = StoriesDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def calculate_class_weights(df, encoders):
    """Calculate class weights for imbalanced datasets"""
    weights = {}
    
    # Safety severity weights
    safety_severity_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(df['safety_severity_encoded']), 
        y=df['safety_severity_encoded']
    )
    weights['safety_severity'] = torch.FloatTensor(safety_severity_weights)
    
    # Bias type weights
    bias_type_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(df['bias_type_encoded']), 
        y=df['bias_type_encoded']
    )
    weights['bias_type'] = torch.FloatTensor(bias_type_weights)
    
    # Age group weights
    age_group_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(df['age_group_encoded']), 
        y=df['age_group_encoded']
    )
    weights['age_group'] = torch.FloatTensor(age_group_weights)
    
    return weights

def train_model(model, train_loader, val_loader, config, class_weights):
    """Train the multi-task model"""
    print("\n" + "=" * 50)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 50)
    
    # Move class weights to device
    for key in class_weights:
        class_weights[key] = class_weights[key].to(config.DEVICE)
    
    # Loss functions
    bce_loss = nn.BCELoss()
    ce_loss_severity = nn.CrossEntropyLoss(weight=class_weights['safety_severity'])
    ce_loss_bias = nn.CrossEntropyLoss(weight=class_weights['bias_type'])
    ce_loss_age = nn.CrossEntropyLoss(weight=class_weights['age_group'])
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(config.WARMUP_STEPS * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            # Move targets to device
            safety_binary_targets = batch['safety_binary'].to(config.DEVICE)
            safety_severity_targets = batch['safety_severity'].to(config.DEVICE)
            safety_type_targets = batch['safety_type'].to(config.DEVICE)
            bias_binary_targets = batch['bias_binary'].to(config.DEVICE)
            bias_type_targets = batch['bias_type'].to(config.DEVICE)
            age_group_targets = batch['age_group'].to(config.DEVICE)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate losses for each task
            loss_safety_binary = bce_loss(outputs['safety_binary'], safety_binary_targets)
            loss_safety_severity = ce_loss_severity(outputs['safety_severity'], safety_severity_targets)
            loss_safety_type = bce_loss(outputs['safety_type'], safety_type_targets)
            loss_bias_binary = bce_loss(outputs['bias_binary'], bias_binary_targets)
            loss_bias_type = ce_loss_bias(outputs['bias_type'], bias_type_targets)
            loss_age_group = ce_loss_age(outputs['age_group'], age_group_targets)
            
            # Combined loss (you can adjust weights)
            total_batch_loss = (
                loss_safety_binary + 
                loss_safety_severity + 
                loss_safety_type + 
                loss_bias_binary + 
                loss_bias_type + 
                loss_age_group
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, config, class_weights)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Model saved!")
    
    return model

def evaluate_model(model, data_loader, config, class_weights):
    """Evaluate the model"""
    model.eval()
    
    # Loss functions
    bce_loss = nn.BCELoss()
    ce_loss_severity = nn.CrossEntropyLoss(weight=class_weights['safety_severity'])
    ce_loss_bias = nn.CrossEntropyLoss(weight=class_weights['bias_type'])
    ce_loss_age = nn.CrossEntropyLoss(weight=class_weights['age_group'])
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            # Move targets to device
            safety_binary_targets = batch['safety_binary'].to(config.DEVICE)
            safety_severity_targets = batch['safety_severity'].to(config.DEVICE)
            safety_type_targets = batch['safety_type'].to(config.DEVICE)
            bias_binary_targets = batch['bias_binary'].to(config.DEVICE)
            bias_type_targets = batch['bias_type'].to(config.DEVICE)
            age_group_targets = batch['age_group'].to(config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            
            # Calculate losses
            loss_safety_binary = bce_loss(outputs['safety_binary'], safety_binary_targets)
            loss_safety_severity = ce_loss_severity(outputs['safety_severity'], safety_severity_targets)
            loss_safety_type = bce_loss(outputs['safety_type'], safety_type_targets)
            loss_bias_binary = bce_loss(outputs['bias_binary'], bias_binary_targets)
            loss_bias_type = ce_loss_bias(outputs['bias_type'], bias_type_targets)
            loss_age_group = ce_loss_age(outputs['age_group'], age_group_targets)
            
            batch_loss = (
                loss_safety_binary + 
                loss_safety_severity + 
                loss_safety_type + 
                loss_bias_binary + 
                loss_bias_type + 
                loss_age_group
            )
            
            total_loss += batch_loss.item()
    
    model.train()
    return total_loss / len(data_loader)

def main():
    """Main training pipeline"""
    print("Children's Stories Multi-Task Classification")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    print(f"Device: {config.DEVICE}")
    
    # Load and explore data
    df = load_and_explore_data(config.DATA_PATH)
    
    # Analyze data structure
    safety_df, bias_df = analyze_data_structure(df)
    
    # Clean and prepare data
    df_clean = clean_and_prepare_data(df)
    
    # Encode labels
    df_encoded, encoders, safety_type_encoded = encode_labels(df_clean)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Prepare labels for multi-task learning
    labels = prepare_labels(df_encoded, safety_type_encoded, encoders)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        df_encoded, labels, tokenizer, config
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(df_encoded, encoders)
    for key in class_weights:
        class_weights[key] = class_weights[key].to(config.DEVICE)
    
    # Initialize model
    model = MultiTaskBERT(
        model_name=config.MODEL_NAME,
        num_age_groups=len(encoders['age_group'].classes_),
        num_severities=len(encoders['safety_severity'].classes_),
        num_safety_types=len(encoders['safety_type_mlb'].classes_),
        num_bias_types=len(encoders['bias_type'].classes_)
    ).to(config.DEVICE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, config, class_weights)
    
    print("\nTraining completed!")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    
    # Save encoders and configuration
    import pickle
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print("Encoders saved to encoders.pkl")

if __name__ == "__main__":
    main()