# MedNLP - Clinical Entity Extraction & Risk Analytics

> **A professional-grade Streamlit application for extracting medical entities from clinical text and generating actionable risk insights with comprehensive visual analytics.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-FF4B4B.svg)

---

## 🎯 Overview

MedNLP is a clinical decision support tool designed to help healthcare professionals quickly extract structured medical information from unstructured clinical text. The application uses advanced NLP techniques to identify medical entities, assess risk factors, and provide clinician-friendly insights through an intuitive interface.

### ⚕️ Key Capabilities

- **Multi-Format Support**: Process clinical text from direct input or uploaded files (TXT, DOCX, PDF)
- **Intelligent Entity Recognition**: Extract diseases, symptoms, medications, procedures, anatomical terms, and lab values
- **Risk Stratification**: Automated risk assessment with confidence scoring and severity classification
- **Clinical Insights**: AI-generated summaries highlighting critical findings, risk factors, and red flags
- **Visual Analytics**: Interactive charts, risk gauges, and comorbidity network diagrams
- **Data Export**: Download results in CSV or JSON format for further analysis
- **OCR Support**: Extract text from image-based PDFs using Tesseract OCR

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

### Entity Extraction
- **Comprehensive Medical NER**: Identifies 6+ entity types including diseases, symptoms, medications, procedures, anatomical terms, and lab values
- **Confidence Scoring**: Each entity includes a confidence score for reliability assessment
- **Context Preservation**: Maintains entity relationships and contextual information
- **Custom Dictionaries**: Extensible medical terminology databases

### Clinical Risk Analysis
- **Risk Stratification**: Automatic classification into Low, Moderate, High, or Critical risk categories
- **Comorbidity Detection**: Identifies disease interactions and potential complications
- **Red Flag Identification**: Highlights critical findings requiring immediate attention
- **Temporal Analysis**: Tracks condition progression and acute vs. chronic indicators

### Visual Analytics
- **Risk Factor Charts**: Bar charts showing top risk contributors with severity scores
- **Condition Distribution**: Visual breakdown of identified medical conditions
- **Risk Gauge**: Intuitive gauge visualization of overall risk level
- **Comorbidity Networks**: Interactive graphs showing disease relationships (when applicable)

### User Experience
- **Modern UI**: Clean, medical-grade interface with professional styling
- **Responsive Design**: Works seamlessly on desktop and tablet devices
- **Real-time Processing**: Instant analysis with progress indicators
- **Filterable Results**: Interactive tables with entity type and confidence filtering
- **Export Options**: One-click download of results in multiple formats

---

## 🛠 Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **spaCy**: Natural language processing and entity recognition

### NLP & Machine Learning
- **scikit-learn**: Risk modeling and feature engineering
- **numpy & pandas**: Data manipulation and analysis
- **networkx**: Comorbidity network analysis

### Document Processing
- **PyMuPDF (fitz)**: PDF text extraction
- **PyPDF2**: Alternative PDF processing
- **python-docx**: Microsoft Word document handling
- **Pillow (PIL)**: Image processing
- **pytesseract**: OCR for image-based documents

### Visualization
- **matplotlib**: Chart generation and data visualization
- **Custom CSS**: Professional medical-grade styling

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Tesseract OCR for image-based PDF processing

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mednlp.git
cd mednlp
```

### Step 2: Create Virtual Environment

**Recommended** to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/MacOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Download spaCy Model

```bash
# Install small model (recommended for speed)
python -m spacy download en_core_web_sm

# Or install large model (recommended for accuracy)
python -m spacy download en_core_web_lg
```

### Step 5: (Optional) Install Tesseract OCR

Only needed if processing image-based PDFs:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

**MacOS:**
```bash
brew install tesseract
```

**Windows:**
Download installer from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 🚀 Quick Start

### Basic Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface:**
   - Open your browser to `http://localhost:8501`
   - The application will automatically open in your default browser

3. **Analyze clinical text:**
   - **Option A**: Paste text directly into the text area
   - **Option B**: Upload a file (TXT, DOCX, or PDF)
   - Click "🔍 Analyze Text" to process

4. **Review results:**
   - View extracted entities in the interactive table
   - Examine risk insights and clinical summaries
   - Explore visual analytics and charts
   - Export data using CSV or JSON download buttons

### Example Workflow

```python
# Example clinical text to analyze:
"""
Patient presents with chest pain and shortness of breath. 
History of hypertension and diabetes mellitus type 2. 
Current medications include metformin 1000mg BID and lisinopril 10mg daily.
Blood pressure elevated at 160/95 mmHg. 
ECG shows ST-segment changes suggestive of acute coronary syndrome.
"""
```

---

## 📖 Usage Guide

### Input Methods

#### 1. Direct Text Input
- Paste clinical notes, discharge summaries, or medical reports
- Supports any length of text (performance optimal under 10,000 words)
- Real-time character count displayed

#### 2. File Upload
- **Supported formats**: `.txt`, `.docx`, `.pdf`
- **Maximum file size**: 200MB (configurable)
- **OCR support**: Enable for scanned or image-based PDFs

### Understanding Results

#### Entity Table
- **Entity**: Extracted medical term
- **Type**: Classification (Disease, Symptom, Medication, etc.)
- **Confidence**: Reliability score (0.0 - 1.0)
- **Context**: Surrounding text for verification

#### Clinical Insights
- **Overall Risk**: Stratified risk level with rationale
- **Top Risk Factors**: Primary contributors ranked by severity
- **Common Conditions**: Most frequently mentioned diagnoses
- **Red Flags**: Critical findings requiring attention
- **Comorbidity Interactions**: Related conditions and complications
- **Recommendations**: Evidence-based next steps

#### Visual Analytics
- **Risk Factor Chart**: Bar chart of weighted risk contributors
- **Condition Distribution**: Pie/bar chart of disease categories
- **Risk Gauge**: Color-coded risk level indicator
- **Network Diagram**: Comorbidity relationship graph

### Filtering & Export

- **Filter by Entity Type**: Show only specific categories (e.g., Medications only)
- **Filter by Confidence**: Set minimum confidence threshold
- **Export CSV**: Tabular data for spreadsheet analysis
- **Export JSON**: Structured data for programmatic processing

---

## ⚙️ Configuration

### Application Settings

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#0F62FE"
backgroundColor = "#FAFAFA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#171717"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

### spaCy Model Selection

Default uses small model for speed. To use large model for better accuracy:

**Edit `app.py`:**
```python
# Change this line in load_extractor()
extractor = EnhancedMedicalEntityExtractor(use_large_model=True)
```

**Install large model:**
```bash
python -m spacy download en_core_web_lg
```

### Custom Medical Dictionary

Extend entity recognition in `medical_nlp.py`:

```python
# Add custom medical terms
CUSTOM_DISEASE_PATTERNS = [
    "your_custom_disease",
    "another_condition"
]

# Add to appropriate matcher in the extractor class
```

### Risk Scoring Thresholds

Modify risk calculation in `riskAnalytics.py`:

```python
# Adjust risk level thresholds
RISK_THRESHOLDS = {
    'low': (0, 30),
    'moderate': (30, 60),
    'high': (60, 85),
    'critical': (85, 100)
}
```

---

## 📁 Project Structure

```
mednlp/
├── app.py                      # Main Streamlit application
├── medical_nlp.py              # Entity extraction & NLP processing
├── riskAnalytics.py            # Risk analysis & insights generation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── custom_medical_ner/         # Optional custom spaCy model
│   ├── meta.json
│   ├── config.cfg
│   └── ...
└── assets/                     # (Optional) Images, icons, etc.
```

### Key Files

- **`app.py`**: UI orchestration, page routing, user interactions
- **`medical_nlp.py`**: Core NLP engine, entity extraction, file processing
- **`riskAnalytics.py`**: Risk stratification, insights, recommendations, visualizations
- **`requirements.txt`**: Complete list of Python package dependencies
- **`.streamlit/config.toml`**: UI theme and server configuration

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `externally-managed-environment` error when installing packages

**Solution**: Use a virtual environment (see Installation Step 2)

---

**Issue**: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution**: 
```bash
# Ensure virtual environment is activated, then:
python -m spacy download en_core_web_sm
```

---

**Issue**: `streamlit: command not found`

**Solution**:
1. Verify virtual environment is activated
2. Reinstall Streamlit: `pip install --upgrade streamlit`
3. Check PATH includes venv scripts directory

---

#### Runtime Issues

**Issue**: OCR not working for image-based PDFs

**Solution**:
1. Install system Tesseract: `sudo apt-get install tesseract-ocr`
2. Verify installation: `tesseract --version`
3. Ensure `pytesseract` is in requirements.txt

---

**Issue**: Upload fails or shows no text

**Solution**:
- Check file isn't corrupted or password-protected
- Try converting PDF to text using external tool first
- Enable OCR if PDF contains only images
- Verify file size under 200MB limit

---

**Issue**: Slow performance with large documents

**Solution**:
- Use small spaCy model (`en_core_web_sm`) instead of large
- Disable OCR if not needed
- Process documents in smaller chunks
- Consider upgrading server resources

---

**Issue**: Port 8501 already in use

**Solution**:
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or run on different port
streamlit run app.py --server.port 8502
```

---

### Getting Help

1. Check [Issues](https://github.com/yourusername/mednlp/issues) for similar problems
2. Review [spaCy documentation](https://spacy.io/usage)
3. Consult [Streamlit documentation](https://docs.streamlit.io)
4. Open a new issue with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/mednlp.git
cd mednlp

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add: your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update README if adding configuration options

### Areas for Contribution

- 🧪 Additional medical entity types
- 📊 New visualization options
- 🌐 Multi-language support
- 🔒 HIPAA compliance features
- 📱 Mobile responsiveness improvements
- 🧠 Custom model training pipelines

---

## ⚖️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This application is designed as a clinical decision support tool and is intended for **informational purposes only**. It is **NOT** a substitute for professional medical judgment, diagnosis, or treatment.

### Important Notes:

- ✅ Results should be **reviewed and validated** by qualified healthcare professionals
- ✅ **Always** consult clinical guidelines and evidence-based practices
- ✅ Use as a **supplementary tool** to enhance, not replace, clinical expertise
- ❌ **Do not** base critical medical decisions solely on this tool's output
- ❌ **Do not** use for emergency medical situations
- ❌ **Not FDA approved** or certified for clinical diagnosis

### Privacy & Security:

- Ensure compliance with **HIPAA** and local data protection regulations
- Do not upload **identifiable patient information** without proper safeguards
- Consider deploying in secure, compliant environments for production use
- Implement appropriate **access controls** and **audit logging**

### Liability:

The developers and contributors assume **no liability** for any consequences arising from the use or misuse of this application. Users assume full responsibility for validating results and ensuring appropriate clinical application.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **spaCy**: For powerful NLP capabilities
- **Streamlit**: For making beautiful web apps accessible
- **Medical community**: For domain knowledge and validation
- **Open source contributors**: For continuous improvement

