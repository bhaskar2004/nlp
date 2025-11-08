# MedNLP - Clinical Text Analysis

Extract medical entities and generate risk insights from clinical text.

## Quick Start

```bash
# Clone
git clone https://github.com/bhaskar2004/nlp.git
cd nlp

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run
streamlit run app.py
```

Open `http://localhost:8501`

## Features

- Extract diseases, symptoms, medications, procedures
- Generate risk assessments and insights
- Visual charts and analytics
- Upload TXT, DOCX, or PDF files
- Export to CSV/JSON

## Optional: OCR Support

For image-based PDFs:
```bash
sudo apt-get install tesseract-ocr  # Ubuntu
brew install tesseract              # MacOS
```

## Troubleshooting

**Model not found?**
```bash
python -m spacy download en_core_web_sm
```

**Port busy?**
```bash
streamlit run app.py --server.port 8502
```

## Note

For informational purposes only. Not a substitute for professional medical advice.
