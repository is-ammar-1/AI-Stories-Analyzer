"""
🧸 Children's Stories AI Safety Analyzer
Author: Muhammad Ammar
Date: June 2025

Interactive Streamlit app for analyzing children's stories using multi-task BERT classification
"""

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Children's Stories Safety Analyzer",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    
    .safe-story {
        border-left-color: #96CEB4 !important;
        background: linear-gradient(135deg, #f5fffe, #e8fffe);
    }
    
    .warning-story {
        border-left-color: #FFEAA7 !important;
        background: linear-gradient(135deg, #fffef5, #fffce8);
    }
    
    .danger-story {
        border-left-color: #FF6B6B !important;
        background: linear-gradient(135deg, #fff5f5, #ffe8e8);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .story-input {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .story-input:focus {
        border-color: #4ECDC4;
        box-shadow: 0 0 0 0.2rem rgba(78, 205, 196, 0.25);
    }
    
    .emoji-large {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(90deg, #FF6B6B, #FFEAA7, #96CEB4);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Classes (from original training script)
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
        self.safety_binary_head = nn.Linear(hidden_size, 1)
        self.safety_severity_head = nn.Linear(hidden_size, num_severities)
        self.safety_type_head = nn.Linear(hidden_size, num_safety_types)
        self.bias_binary_head = nn.Linear(hidden_size, 1)
        self.bias_type_head = nn.Linear(hidden_size, num_bias_types)
        self.age_group_head = nn.Linear(hidden_size, num_age_groups)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
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

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Get dimensions from encoders
        num_age_groups = len(encoders['age_group'].classes_)
        num_severities = len(encoders['safety_severity'].classes_)
        num_safety_types = len(encoders['safety_type_mlb'].classes_)
        num_bias_types = len(encoders['bias_type'].classes_)
        
        # Initialize model
        model = MultiTaskBERT(
            model_name="distilbert-base-uncased",
            num_age_groups=num_age_groups,
            num_severities=num_severities,
            num_safety_types=num_safety_types,
            num_bias_types=num_bias_types
        )
        
        # Load model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('children_stories_model.pth', map_location=device))
        model.to(device)
        model.eval()
        
        return model, tokenizer, encoders, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def predict_story(story_text, model, tokenizer, encoders, device):
    """Make predictions for a given story"""
    if not story_text.strip():
        return None
    
    # Tokenize
    encoding = tokenizer(
        story_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Process predictions
    predictions = {}
    
    # Safety binary
    safety_prob = outputs['safety_binary'].cpu().numpy()[0][0]
    predictions['safety_violation'] = {
        'probability': float(safety_prob),
        'prediction': safety_prob > 0.5
    }
    
    # Safety severity
    safety_severity_probs = torch.softmax(outputs['safety_severity'], dim=1).cpu().numpy()[0]
    severity_idx = np.argmax(safety_severity_probs)
    predictions['safety_severity'] = {
        'prediction': encoders['safety_severity'].classes_[severity_idx],
        'confidence': float(safety_severity_probs[severity_idx]),
        'all_probabilities': {
            cls: float(prob) for cls, prob in 
            zip(encoders['safety_severity'].classes_, safety_severity_probs)
        }
    }
    
    # Safety types (multi-label)
    safety_type_probs = outputs['safety_type'].cpu().numpy()[0]
    safety_types = []
    for i, prob in enumerate(safety_type_probs):
        if prob > 0.5:
            safety_types.append({
                'type': encoders['safety_type_mlb'].classes_[i],
                'confidence': float(prob)
            })
    predictions['safety_types'] = safety_types
    
    # Bias binary
    bias_prob = outputs['bias_binary'].cpu().numpy()[0][0]
    predictions['bias_detection'] = {
        'probability': float(bias_prob),
        'prediction': bias_prob > 0.5
    }
    
    # Bias type
    bias_type_probs = torch.softmax(outputs['bias_type'], dim=1).cpu().numpy()[0]
    bias_type_idx = np.argmax(bias_type_probs)
    predictions['bias_type'] = {
        'prediction': encoders['bias_type'].classes_[bias_type_idx],
        'confidence': float(bias_type_probs[bias_type_idx]),
        'all_probabilities': {
            cls: float(prob) for cls, prob in 
            zip(encoders['bias_type'].classes_, bias_type_probs)
        }
    }
    
    # Age group
    age_group_probs = torch.softmax(outputs['age_group'], dim=1).cpu().numpy()[0]
    age_group_idx = np.argmax(age_group_probs)
    predictions['age_group'] = {
        'prediction': encoders['age_group'].classes_[age_group_idx],
        'confidence': float(age_group_probs[age_group_idx]),
        'all_probabilities': {
            cls: float(prob) for cls, prob in 
            zip(encoders['age_group'].classes_, age_group_probs)
        }
    }
    
    return predictions

def create_safety_visualization(predictions):
    """Create safety visualization charts"""
    # Safety Overview Gauge
    safety_score = 1 - predictions['safety_violation']['probability']
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = safety_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Safety Score"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Safety Severity Distribution
    severity_data = predictions['safety_severity']['all_probabilities']
    fig_severity = px.bar(
        x=list(severity_data.keys()),
        y=list(severity_data.values()),
        title="Safety Severity Probabilities",
        color=list(severity_data.values()),
        color_continuous_scale="RdYlGn_r"
    )
    fig_severity.update_layout(showlegend=False, xaxis_title="Severity Level", yaxis_title="Probability")
    
    return fig_gauge, fig_severity

def create_bias_visualization(predictions):
    """Create bias visualization charts"""
    bias_data = predictions['bias_type']['all_probabilities']
    
    # Bias Type Distribution
    fig_bias = px.pie(
        values=list(bias_data.values()),
        names=list(bias_data.keys()),
        title="Bias Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig_bias

def create_age_group_visualization(predictions):
    """Create age group visualization"""
    age_data = predictions['age_group']['all_probabilities']
    
    # Age Group Radar Chart
    categories = list(age_data.keys())
    values = list(age_data.values())
    
    fig_age = go.Figure()
    
    fig_age.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Age Group Suitability',
        line_color='rgb(78, 205, 196)'
    ))
    
    fig_age.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=True,
        title="Age Group Appropriateness"
    )
    
    return fig_age

def display_story_analysis(predictions):
    """Display comprehensive story analysis"""
    
    # Determine overall safety level
    safety_prob = predictions['safety_violation']['probability']
    bias_prob = predictions['bias_detection']['probability']
    
    if safety_prob < 0.3 and bias_prob < 0.3:
        safety_level = "safe"
        safety_emoji = "✅"
        safety_color = "#96CEB4"
        card_class = "safe-story"
    elif safety_prob < 0.7 and bias_prob < 0.7:
        safety_level = "caution"
        safety_emoji = "⚠️"
        safety_color = "#FFEAA7"
        card_class = "warning-story"
    else:
        safety_level = "review"
        safety_emoji = "❌"
        safety_color = "#FF6B6B"
        card_class = "danger-story"
    
    # Main prediction card
    st.markdown(f"""
     <div class="prediction-card {card_class}" style="background: {safety_color}; color: #222; border-left: 8px solid {safety_color}; box-shadow: 0 5px 20px rgba(0,0,0,0.08);">
          <div style="text-align: center;">
               <div class="emoji-large">{safety_emoji}</div>
               <h2 style="color: #222;">Story Analysis Complete</h2>
               <h3 style="color: #222; background: rgba(255,255,255,0.5); display: inline-block; padding: 0.3em 1em; border-radius: 8px;">
                    Recommendation: {safety_level.upper()}
               </h3>
          </div>
     </div>
     """, unsafe_allow_html=True)
    
    # Create three columns for main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4 style="color: #222;">🛡️ Safety Violation</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(
            "Probability",
            f"{predictions['safety_violation']['probability']:.2%}",
            delta=f"{'High Risk' if predictions['safety_violation']['prediction'] else 'Safe'}"
        )
        
        if predictions['safety_violation']['prediction']:
            st.error(f"⚠️ Predicted Severity: **{predictions['safety_severity']['prediction'].title()}**")
            
            if predictions['safety_types']:
                st.write("**Detected Safety Issues:**")
                for safety_type in predictions['safety_types']:
                    st.write(f"• {safety_type['type'].title()} ({safety_type['confidence']:.2%} confidence)")
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4 style="color: #222;">🎭 Bias Detection</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(
            "Probability",
            f"{predictions['bias_detection']['probability']:.2%}",
            delta=f"{'Bias Detected' if predictions['bias_detection']['prediction'] else 'No Bias'}"
        )
        
        if predictions['bias_detection']['prediction']:
            st.warning(f"📊 Predicted Bias Type: **{predictions['bias_type']['prediction'].title()}**")
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4 style="color: #222;">👶 Age Appropriateness</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(
            "Best Fit",
            predictions['age_group']['prediction'].replace('_', '-').title(),
            delta=f"{predictions['age_group']['confidence']:.2%} confidence"
        )
    
    # Detailed visualizations
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🛡️ Safety Analysis", "🎭 Bias Analysis", "👶 Age Group Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge, fig_severity = create_safety_visualization(predictions)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Safety types table
        if predictions['safety_types']:
            st.subheader("🚨 Detected Safety Issues")
            safety_df = pd.DataFrame(predictions['safety_types'])
            safety_df['confidence'] = safety_df['confidence'].apply(lambda x: f"{x:.2%}")
            st.dataframe(safety_df, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_bias = create_bias_visualization(predictions)
            st.plotly_chart(fig_bias, use_container_width=True)
        
        with col2:
            st.subheader("📈 Bias Detection Details")
            st.metric("Detection Probability", f"{predictions['bias_detection']['probability']:.2%}")
            st.metric("Most Likely Bias Type", predictions['bias_type']['prediction'].title())
            st.metric("Type Confidence", f"{predictions['bias_type']['confidence']:.2%}")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_age = create_age_group_visualization(predictions)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Age Group Recommendations")
            age_probs = predictions['age_group']['all_probabilities']
            
            # Sort by probability
            sorted_ages = sorted(age_probs.items(), key=lambda x: x[1], reverse=True)
            
            for i, (age_group, prob) in enumerate(sorted_ages[:3]):
                if i == 0:
                    st.success(f"🥇 **{age_group.replace('_', '-').title()}**: {prob:.2%}")
                elif i == 1:
                    st.info(f"🥈 **{age_group.replace('_', '-').title()}**: {prob:.2%}")
                else:
                    st.warning(f"🥉 **{age_group.replace('_', '-').title()}**: {prob:.2%}")

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧸 Children's Stories Safety Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Advanced multi-task model to analyze children's stories for safety, bias, and age appropriateness
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model and encoders..."):
        model, tokenizer, encoders, device = load_model_and_encoders()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if 'children_stories_model.pth' and 'encoders.pkl' exist in the current directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🎯 About This Tool")
        st.markdown("""
        This AI model analyzes children's stories across multiple dimensions:
        
        **🛡️ Safety Analysis:**
        - Detects potential safety violations
        - Classifies severity levels
        - Identifies specific safety issue types
        
        **🎭 Bias Detection:**
        - Identifies potential stereotypes
        - Classifies bias types
        - Provides confidence scores
        
        **👶 Age Appropriateness:**
        - Recommends suitable age groups
        - Considers content complexity
        - Evaluates developmental appropriateness
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(f"""
        - **Model:** DistilBERT-based Multi-task Classifier  
        - **Training Data:** 2,866 AI-generated stories  
        - **Model Parameters:** 66 million  
        - **Device:** {device.type.upper()}  
        - **Tasks:** 6 concurrent classification tasks
     """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Enter complete stories for best results
        - The model works with stories of any length
        - Results show confidence scores for transparency
        - Green = Safe, Yellow = Caution, Red = Review needed
        """)
    
    # Main input area
    st.markdown("### 📝 Enter Your Story")
    
    # Sample stories for quick testing
    sample_stories = {
        "Safe Story": "Once upon a time, there was a little bunny who loved to explore the meadow. Every day, the bunny would hop around, smelling flowers and making friends with other woodland creatures. The bunny learned about kindness and sharing through adventures with friends.",
        
        "Potentially Concerning": "The story of a child who always gets what they want by throwing tantrums. Whenever the child screams and cries, parents immediately give in to all demands. Other children should learn to behave the same way to get their desires fulfilled.",
        
        "Age-Inappropriate": "In the dark laboratory, the scientist conducted experiments that would change humanity forever. The complex molecular structures and advanced quantum physics principles required deep understanding of thermodynamics and biochemical interactions."
    }
    
    # Quick sample selection
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📚 Load Safe Story", help="Load a sample safe story"):
            st.session_state.story_text = sample_stories["Safe Story"]
    with col2:
        if st.button("⚠️ Load Concerning Story", help="Load a potentially concerning story"):
            st.session_state.story_text = sample_stories["Potentially Concerning"]
    with col3:
        if st.button("🎓 Load Complex Story", help="Load an age-inappropriate complex story"):
            st.session_state.story_text = sample_stories["Age-Inappropriate"]
    
    # Text input
    story_text = st.text_area(
        "Story Content",
        value=st.session_state.get('story_text', ''),
        height=200,
        placeholder="Enter the children's story you want to analyze...",
        help="Paste or type the complete story text here. The model works best with complete stories."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Story", type="primary", use_container_width=True):
        if story_text.strip():
            with st.spinner("🤖 AI is analyzing your story..."):
                predictions = predict_story(story_text, model, tokenizer, encoders, device)
            
            if predictions:
                display_story_analysis(predictions)
            else:
                st.error("❌ Failed to analyze the story. Please try again.")
        else:
            st.warning("⚠️ Please enter a story to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧸 Children's Stories AI Safety Analyzer | Built with ❤️ using Streamlit & PyTorch</p>
        <p><em>Promoting safe and inclusive storytelling for children</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()