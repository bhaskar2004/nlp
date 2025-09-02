# Medical NLP Entity Extraction System

This project is a web-based application for extracting medical entities from clinical text using NLP techniques. It supports custom text input, batch processing of multiple files, and CSV export of results.

## Features
- **Custom Text Tester**: Input any clinical text and extract medical entities instantly.
- **Batch Processor**: Upload and process multiple `.txt` files at once.
- **CSV Export**: Download extracted entities as a CSV file for further analysis.
- **Web Interface**: User-friendly web app built with Streamlit.

## Installation
1. Clone the repository or download the files.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

Open the provided URL in your browser to use the web interface.

## File Structure
- `medical_nlp.py`: Core NLP logic and entity extraction.
- `app.py`: Streamlit web application.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Notes
- Make sure to install the spaCy English model as shown above.
- For batch processing, upload plain text (`.txt`) files only.

---
Feel free to extend the dictionaries and patterns in `medical_nlp.py` for more comprehensive extraction! 