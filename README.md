# 📚 Children's Stories AI Safety Analyzer

A beautiful, interactive Streamlit web application that uses AI to analyze children's stories for safety concerns, stereotypes, biases, and age appropriateness.

## 🌟 Features

### Multi-Task Analysis
- **🛡️ Safety Violation Detection**: Identifies potential safety concerns like violence, scary content, unsupervised activities, and mature themes
- **👥 Bias & Stereotype Detection**: Detects gender, cultural, and racial biases or stereotypes
- **🎯 Age Appropriateness**: Classifies stories into age groups (4-6, 7-12, 13+)
- **📊 Comprehensive Scoring**: Provides overall quality assessment with recommendations

### Interactive Features
- **🎨 Beautiful UI**: Modern, colorful interface with custom CSS styling
- **📊 Rich Visualizations**: Interactive charts, radar plots, and confidence gauges using Plotly
- **⚡ Real-time Analysis**: Instant predictions with progress indicators
- **📝 Example Stories**: Pre-loaded examples for quick testing
- **📥 Export Results**: Download analysis reports in JSON format

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- The trained model files:
  - `children_stories_model.pth` (trained model weights)
  - `encoders.pkl` (label encoders)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:
   Make sure these files are in the same directory as `streamlit_app.py`:
   - `children_stories_model.pth`
   - `encoders.pkl`

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in browser**:
   The app will automatically open in your default browser, typically at `http://localhost:8501`

## 🎯 How to Use

### 1. Enter Your Story
- Type or paste a children's story in the text area
- Use the example stories from the sidebar for quick testing
- The app works best with stories between 50-1000 words

### 2. Analyze
- Click the "🔍 Analyze Story" button
- Wait for the AI to process your story (usually takes a few seconds)
- View the comprehensive analysis results

### 3. Explore Results
Navigate through different tabs to see:

#### 🛡️ Safety Analysis
- Overall safety violation probability
- Safety severity classification (none, mild, moderate, severe)
- Specific safety types (violence, scary content, etc.)

#### 👥 Bias Analysis  
- Bias detection confidence gauge
- Bias type classification (gender, cultural, racial, etc.)
- Detailed probability breakdowns

#### 🎯 Age Appropriateness
- Recommended age group
- Confidence levels for each age category
- Age-specific characteristics and guidelines

#### 📊 Overall Assessment
- Quality radar chart
- Overall score and recommendations
- Actionable improvement suggestions

### 4. Export Results
- Download comprehensive analysis reports
- JSON format for further processing
- Timestamped for record keeping

## 🧠 Model Details

### Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Type**: Multi-task transformer model
- **Parameters**: 66+ million parameters
- **Training Data**: 2,866 AI-generated children's stories

### Tasks
1. **Binary Classification**: Safety violation detection, bias detection
2. **Multi-class Classification**: Safety severity, bias type, age group
3. **Multi-label Classification**: Safety types (can detect multiple simultaneously)

### Performance
The model has been trained and validated on a diverse dataset of children's stories with comprehensive safety and bias annotations.

## 🎨 Interface Features

### Visual Design
- **Gradient backgrounds** and modern card layouts
- **Color-coded indicators** for different safety levels
- **Interactive charts** with hover effects and animations
- **Responsive design** that works on desktop and mobile

### User Experience
- **Real-time feedback** with character and word counts
- **Progress indicators** during analysis
- **Clear warnings** for concerning content
- **Intuitive navigation** with organized tabs

## 🔧 Technical Implementation

### Key Components
- **Streamlit**: Web framework for the interactive interface
- **PyTorch**: Deep learning framework for model inference
- **Transformers**: Hugging Face library for BERT tokenization
- **Plotly**: Interactive visualization library
- **Scikit-learn**: Label encoding and preprocessing

### Performance Optimizations
- **Model caching** with `@st.cache_resource` for faster loading
- **Efficient tokenization** with proper padding and truncation
- **GPU support** when available (automatically detected)

## 📊 Example Outputs

### Safe Story Example
```
Story: "Emma and her friends went to the park with their teacher..."
✅ Safe Content (95% confidence)
✅ Bias-Free (92% confidence)  
🎯 Age Group: 4-6 (88% confidence)
🌟 Overall Score: 92%
```

### Concerning Story Example
```
Story: "Tommy pushed his classmate and said girls can't play soccer..."
❌ Safety Concerns (78% confidence)
❌ Bias Detected (85% confidence)
🎯 Age Group: 7-12 (76% confidence)
⚠️ Overall Score: 45%
```

## 🛡️ Safety & Limitations

### Important Notes
- This tool provides AI-powered suggestions and should not replace human judgment
- Always have qualified adults review children's content
- The model's predictions are based on training data and may not catch all edge cases
- Consider cultural context and individual child sensitivity

### Recommended Workflow
1. Use AI analysis as initial screening
2. Review flagged content manually
3. Test with target age group when possible
4. Iterate and improve based on feedback

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment
The app can be deployed on:
- **Streamlit Cloud**: Easy one-click deployment
- **Heroku**: With appropriate buildpacks
- **AWS/GCP/Azure**: Using container services
- **Docker**: Containerized deployment

### Environment Variables
For cloud deployment, ensure:
- Adequate memory (minimum 4GB recommended)
- Python 3.8+ runtime
- All dependencies from requirements.txt

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional safety categories
- More sophisticated bias detection
- Multi-language support
- Enhanced visualizations
- Performance optimizations

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with applicable AI ethics guidelines and children's safety regulations.

## 📞 Support

For issues or questions:
1. Check the model files are properly loaded
2. Verify all dependencies are installed
3. Ensure sufficient system memory
4. Review error messages in the Streamlit interface

---

**Built with ❤️ for safer children's storytelling**
