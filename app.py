import streamlit as st
import pandas as pd
from medical_nlp import EnhancedMedicalEntityExtractor
import spacy
import os
from collections import Counter

# --- Page Config ---
st.set_page_config(page_title="Medical NLP Entity Extractor", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
        .stApp { background-color: #f4f8fb; }
        .main-title { font-size: 2.5rem; font-weight: bold; color: #0056b3; margin-bottom: 0.5em; }
        .subtitle { font-size: 1.2rem; color: #222; margin-bottom: 1em; }

        .entity-highlight {
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 6px;
            margin: 2px 2px;
            display: inline-block;
            color: #fff !important; /* force white text */
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: all 0.2s ease;
        }
        .entity-highlight small {
            font-size: 0.7rem;
            font-weight: 400;
            margin-left: 6px;
            opacity: 0.9;
        }

        .main-text-box {
            background: #fff;
            border-radius: 12px;
            border: 2px solid #0056b344;
            padding: 20px 26px;
            margin-bottom: 1.5em;
            font-size: 1.2rem;
            color: #111;
            box-shadow: 0 4px 12px rgba(0, 86, 179, 0.15);
            line-height: 1.5;
        }
        .stButton>button {
            background-color: #0056b3;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 700;
            padding: 10px 18px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #003d80;
            cursor: pointer;
        }
        .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #0056b3;
            font-size: 1.1rem;
            padding: 10px;
            color: #111;
            background-color: #fff;
        }
        .stTextArea textarea:focus {
            outline: none;
            border-color: #003d80;
            box-shadow: 0 0 8px rgba(0, 61, 128, 0.5);
        }
        .stDataFrame {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">🩺 Medical NLP Entity Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extract and visualize medical entities from clinical text using advanced NLP techniques.</div>', unsafe_allow_html=True)

# --- Load Custom Model if available ---
CUSTOM_MODEL_PATH = "custom_medical_ner"
custom_nlp = None
if os.path.exists(CUSTOM_MODEL_PATH):
    custom_nlp = spacy.load(CUSTOM_MODEL_PATH)

extractor = EnhancedMedicalEntityExtractor()

# --- Vibrant Colors for Entities ---
ENTITY_COLORS = {
    "DISEASE": "#d9534f",    # red
    "SYMPTOM": "#f0ad4e",    # orange
    "MEDICATION": "#5bc0de", # blue
    "TEST": "#5cb85c",       # green
    "PROCEDURE": "#337ab7",  # dark blue
    "BODY_PART": "#9b59b6",  # purple
    "TIME": "#34495e",       # dark grey-blue
    "DOSAGE": "#e67e22",     # bright orange
    "AGE": "#2ecc71",        # light green
    "OTHER": "#95a5a6"       # grey
}

# --- Helper functions ---
def highlight_entities_vibrant(text, entities):
    """Return HTML with vibrant highlights for entities and visible base text."""
    if not entities:
        return f"<span style='color:#111;font-size:1.1rem;'>{text}</span>"

    sorted_ents = sorted(entities, key=lambda e: e['Start'])
    html = ""
    last_idx = 0
    for ent in sorted_ents:
        start, end, label = ent['Start'], ent['End'], ent['Label']
        color = ENTITY_COLORS.get(label, "#007bff")
        # Add the plain text before the entity
        html += f"<span style='color:#111;font-size:1.1rem;'>{text[last_idx:start]}</span>"
        # Add the highlighted entity
        html += f"<span class='entity-highlight' style='background-color:{color};'>{text[start:end]} <small>{label}</small></span>"
        last_idx = end
    # Add the remaining text after the last entity
    html += f"<span style='color:#111;font-size:1.1rem;'>{text[last_idx:]}</span>"
    return html


def analyze_text(text, use_custom_model=False):
    if use_custom_model and custom_nlp:
        doc = custom_nlp(text)
        data = [
            {
                'Text': ent.text,
                'Label': ent.label_,
                'Start': ent.start_char,
                'End': ent.end_char,
                'Confidence': 1.0,
                'Context': text[max(0, ent.start_char-50):ent.end_char+50]
            }
            for ent in doc.ents
        ]
        return pd.DataFrame(data), data
    else:
        entities = extractor.extract_entities(text)
        data = [
            {
                'Text': e.text,
                'Label': e.label,
                'Start': e.start,
                'End': e.end,
                'Confidence': e.confidence,
                'Context': e.context
            }
            for e in entities
        ]
        return pd.DataFrame(data), data

def batch_process(files, use_custom_model=False):
    all_results = []
    for file in files:
        text = file.read().decode('utf-8')
        df, _ = analyze_text(text, use_custom_model=use_custom_model)
        df['Source File'] = file.name
        all_results.append(df)
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Choose mode:", ["Custom Text", "Batch File Processing"])
    model_option = st.radio(
        "Entity Extraction Model:",
        ["Rule-based (default)", "Custom Trained Model" if custom_nlp else "Custom Trained Model (not found)"]
    )
    use_custom_model = model_option.startswith("Custom Trained") and custom_nlp is not None
    if use_custom_model:
        st.success("✅ Custom model loaded!")
    else:
        st.info("ℹ️ Using rule-based extractor.")
    st.markdown("---")
    st.subheader("Entity Color Legend")
    for label, color in ENTITY_COLORS.items():
        st.markdown(
            f"<span style='background-color:{color};color:#fff;padding:4px 10px;border-radius:6px;display:inline-block;margin:3px;'>{label}</span>",
            unsafe_allow_html=True
        )

# --- Main UI ---
if mode == "Custom Text":
    st.subheader("🔎 Custom Text Analysis")
    user_text = st.text_area("Paste or type clinical text below:", height=180)
    if user_text.strip():
        st.markdown(f"<div class='main-text-box'>{user_text}</div>", unsafe_allow_html=True)
    extract_btn = st.button("🚀 Extract Entities")
    if extract_btn and user_text.strip():
        with st.spinner("Extracting entities..."):
            df, results = analyze_text(user_text, use_custom_model=use_custom_model)
        st.success(f"Found {len(df)} entities in your document.")
        st.markdown(f"**Document Length:** {len(user_text)} characters")
        if len(df):
            cat_counts = Counter(df['Label'])
            st.markdown("**Category Breakdown:** " + ", ".join(f"{k}: {v}" for k, v in cat_counts.items()))
        st.markdown("**Entities Highlighted in Text:**", unsafe_allow_html=True)
        st.markdown(highlight_entities_vibrant(user_text, df.to_dict('records')), unsafe_allow_html=True)
        with st.expander("Show Detailed Entity Table"):
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "entities.csv", "text/csv")

elif mode == "Batch File Processing":
    st.subheader("📁 Batch File Processing")
    uploaded_files = st.file_uploader(
        "Upload one or more .txt files", type=["txt"], accept_multiple_files=True
    )
    process_btn = st.button("🚀 Process Files")
    if uploaded_files and process_btn:
        with st.spinner("Processing batch files..."):
            batch_df = batch_process(uploaded_files, use_custom_model=use_custom_model)
        st.success(f"Extracted {len(batch_df)} entities from {len(uploaded_files)} files.")
        with st.expander("Show Batch Entity Table"):
            st.dataframe(batch_df)
            if not batch_df.empty:
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Batch CSV", csv, "batch_entities.csv", "text/csv")
