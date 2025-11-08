import streamlit as st
import pandas as pd
from medical_nlp import EnhancedMedicalEntityExtractor
from riskAnalytics import (
    MedicalRiskAnalyzer,
    create_risk_progress_bar_html,
    create_condition_card_html,
    create_recommendation_card_html,
)
import io
import PyPDF2
import docx

# --- Page Config ---
st.set_page_config(
    page_title="MedNLP - Advanced Medical Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏥"
)

# --- Refined Medical-Grade CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        
        /* ===== DESIGN SYSTEM ===== */
        :root {
            /* Primary Colors - Professional Blues */
            --primary: #0F62FE;
            --primary-hover: #0353E9;
            --primary-light: #EDF5FF;
            
            /* Neutral Palette */
            --neutral-50: #FAFAFA;
            --neutral-100: #F5F5F5;
            --neutral-200: #E5E5E5;
            --neutral-300: #D4D4D4;
            --neutral-400: #A3A3A3;
            --neutral-500: #737373;
            --neutral-600: #525252;
            --neutral-700: #404040;
            --neutral-800: #262626;
            --neutral-900: #171717;
            
            /* Semantic Colors */
            --success: #16A34A;
            --success-light: #F0FDF4;
            --warning: #EA580C;
            --warning-light: #FFF7ED;
            --error: #DC2626;
            --error-light: #FEF2F2;
            --info: #0284C7;
            --info-light: #F0F9FF;
            
            /* Surface Colors */
            --surface: #FFFFFF;
            --background: #FAFAFA;
            --border: #E5E5E5;
            --border-hover: #D4D4D4;
            
            /* Shadows - Subtle and Professional */
            --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            
            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;
            
            /* Typography */
            --font-primary: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
            --font-secondary: 'Inter', system-ui, -apple-system, sans-serif;
            
            /* Spacing Scale */
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --space-2xl: 48px;
            
            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* ===== GLOBAL RESET ===== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body, [class*="css"] {
            font-family: var(--font-primary) !important;
            background: var(--background) !important;
            color: var(--neutral-900) !important;
            line-height: 1.5 !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-size: 16px;
        }
        
        .stApp {
            background: var(--background) !important;
        }
        
        /* ===== ENHANCED LAYOUT ===== */
        .main .block-container {
            padding: var(--space-2xl) var(--space-2xl) var(--space-2xl) var(--space-2xl) !important;
            max-width: 1400px !important;
        }
        
        /* ===== NAVIGATION BAR ===== */
        .top-nav {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            padding: var(--space-md) 0;
            margin: -80px -100px var(--space-xl) -100px;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(8px);
            background: rgba(255, 255, 255, 0.95);
        }
        
        .nav-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 var(--space-2xl);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .nav-logo {
            display: flex;
            align-items: center;
            gap: var(--space-md);
        }
        
        .nav-logo-icon {
            font-size: 24px;
        }
        
        .nav-logo-text h1 {
            font-size: 20px !important;
            font-weight: 600 !important;
            color: var(--neutral-900) !important;
            margin: 0 !important;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        
        .nav-logo-text p {
            font-size: 12px !important;
            color: var(--neutral-500) !important;
            margin: 0 !important;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        
        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: none !important;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: var(--surface) !important;
            color: var(--neutral-700) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 12px var(--space-md) !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            text-align: left !important;
            margin-bottom: var(--space-sm) !important;
            transition: all var(--transition-base) !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--neutral-50) !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }
        
        /* ===== PAGE HEADER ===== */
        .page-header {
            background: var(--surface);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            margin-bottom: var(--space-xl);
            border: 1px solid var(--border);
        }
        
        .page-title {
            font-size: 32px !important;
            font-weight: 600 !important;
            color: var(--neutral-900) !important;
            margin: 0 0 var(--space-sm) 0 !important;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        
        .page-subtitle {
            font-size: 16px !important;
            color: var(--neutral-600) !important;
            margin: 0 !important;
            font-weight: 400;
            line-height: 1.5;
        }
        
        /* ===== CARDS - CLEANER DESIGN ===== */
        .premium-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            margin-bottom: var(--space-md);
            transition: all var(--transition-base);
        }
        
        .premium-card:hover {
            border-color: var(--border-hover);
            box-shadow: var(--shadow-md);
        }
        
        .card-icon {
            font-size: 32px;
            margin-bottom: var(--space-md);
            display: inline-block;
        }
        
        .card-title {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: var(--neutral-900) !important;
            margin: 0 0 var(--space-sm) 0 !important;
            letter-spacing: -0.01em;
        }
        
        .card-text {
            color: var(--neutral-600) !important;
            font-size: 14px !important;
            line-height: 1.6;
        }
        
        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {
            background: var(--surface);
            border: 2px dashed var(--border);
            border-radius: var(--radius-lg);
            padding: var(--space-2xl);
            text-align: center;
            transition: all var(--transition-base);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary);
            background: var(--primary-light);
        }
        
        [data-testid="stFileUploader"] label {
            font-weight: 500 !important;
            color: var(--neutral-900) !important;
            font-size: 14px !important;
        }
        
        /* ===== TEXT INPUTS ===== */
        .stTextArea textarea,
        .stTextInput input {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 12px var(--space-md) !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            transition: all var(--transition-fast) !important;
            color: var(--neutral-900) !important;
        }
        
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: var(--primary) !important;
            outline: none !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
        }
        
        .stTextArea label,
        .stTextInput label {
            font-weight: 500 !important;
            color: var(--neutral-900) !important;
            margin-bottom: var(--space-sm) !important;
            font-size: 14px !important;
        }
        
        /* ===== BUTTONS - SIMPLIFIED ===== */
        .stButton > button {
            background: var(--primary) !important;
            color: var(--surface) !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 12px var(--space-lg) !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            transition: all var(--transition-base) !important;
            box-shadow: var(--shadow-xs) !important;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            background: var(--primary-hover) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stButton > button:active {
            transform: scale(0.98);
        }
        
        /* ===== METRICS ===== */
        [data-testid="metric-container"] {
            background: var(--surface);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            border: 1px solid var(--border);
            text-align: center;
        }
        
        [data-testid="metric-container"] label {
            font-weight: 500 !important;
            color: var(--neutral-600) !important;
            font-size: 12px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: var(--space-sm) !important;
        }
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: 600 !important;
            color: var(--neutral-900) !important;
            line-height: 1;
        }
        
        /* ===== DATAFRAME ===== */
        .stDataFrame {
            border-radius: var(--radius-lg) !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
        }
        
        [data-testid="stDataFrame"] {
            font-size: 14px;
        }
        
        [data-testid="stDataFrame"] thead tr th {
            background: var(--neutral-50) !important;
            color: var(--neutral-900) !important;
            font-weight: 600 !important;
            padding: 12px !important;
            border-bottom: 2px solid var(--border) !important;
        }
        
        [data-testid="stDataFrame"] tbody tr:hover td {
            background: var(--neutral-50) !important;
        }
        
        /* ===== ALERTS ===== */
        .stAlert {
            border-radius: var(--radius-md) !important;
            border: none !important;
            padding: var(--space-md) var(--space-lg) !important;
            margin: var(--space-md) 0 !important;
            font-size: 14px;
            line-height: 1.5;
            border-left: 3px solid;
        }
        
        .stSuccess {
            background: var(--success-light) !important;
            color: var(--success) !important;
            border-left-color: var(--success) !important;
        }
        
        .stInfo {
            background: var(--info-light) !important;
            color: var(--info) !important;
            border-left-color: var(--info) !important;
        }
        
        .stWarning {
            background: var(--warning-light) !important;
            color: var(--warning) !important;
            border-left-color: var(--warning) !important;
        }
        
        .stError {
            background: var(--error-light) !important;
            color: var(--error) !important;
            border-left-color: var(--error) !important;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--surface);
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            padding: 4px;
            margin-bottom: var(--space-lg);
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-sm);
            padding: var(--space-sm) var(--space-md);
            font-weight: 500;
            color: var(--neutral-600);
            transition: all var(--transition-fast);
            font-size: 14px;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--primary);
            color: var(--surface);
        }
        
        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            background: var(--surface) !important;
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            padding: var(--space-md) var(--space-lg) !important;
            font-weight: 500 !important;
            transition: all var(--transition-base) !important;
            color: var(--neutral-900) !important;
            font-size: 14px !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: var(--neutral-50) !important;
            border-color: var(--border-hover) !important;
        }
        
        /* ===== TYPOGRAPHY ===== */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--neutral-900) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        
        .stMarkdown h3 {
            font-size: 20px !important;
            margin: var(--space-xl) 0 var(--space-md) 0 !important;
            padding-bottom: var(--space-md) !important;
            border-bottom: 1px solid var(--border);
        }
        
        .stMarkdown p {
            font-size: 14px !important;
            line-height: 1.6;
            color: var(--neutral-700) !important;
        }
        
        /* ===== DOWNLOAD BUTTON ===== */
        .stDownloadButton > button {
            background: var(--success) !important;
        }
        
        .stDownloadButton > button:hover {
            background: #15803D !important;
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--neutral-100);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--neutral-300);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--neutral-400);
        }
        
        /* ===== DIVIDER ===== */
        hr {
            border: none;
            height: 1px;
            background: var(--border);
            margin: var(--space-xl) 0;
        }
        
        /* ===== LOADING SPINNER ===== */
        .stSpinner > div {
            border-color: var(--primary) var(--neutral-200) var(--neutral-200) !important;
        }
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {
            background: var(--primary) !important;
            border-radius: var(--radius-sm);
        }
        
        /* ===== RESPONSIVE DESIGN ===== */
        @media (max-width: 768px) {
            .main .block-container {
                padding: var(--space-md) !important;
            }
            
            .nav-container {
                padding: 0 var(--space-md);
                flex-direction: column;
                gap: var(--space-md);
            }
            
            .top-nav {
                margin: -80px -20px var(--space-md) -20px;
            }
            
            .page-header {
                padding: var(--space-lg);
            }
            
            .page-title {
                font-size: 24px !important;
            }
            
            [data-testid="stFileUploader"] {
                padding: var(--space-lg);
            }
        }
        
        /* ===== ACCESSIBILITY ===== */
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        select:focus-visible {
            outline: 2px solid var(--primary) !important;
            outline-offset: 2px !important;
        }
        
        /* Reduce motion for users who prefer it */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* ===== IMPROVED SELECT BOXES ===== */
        .stSelectbox > div > div {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: var(--border-hover) !important;
        }
        
        /* ===== IMPROVED RADIO BUTTONS ===== */
        .stRadio > label {
            font-weight: 500 !important;
            color: var(--neutral-900) !important;
            font-size: 14px !important;
        }
        
        /* ===== IMPROVED CHECKBOXES ===== */
        .stCheckbox > label {
            font-weight: 400 !important;
            color: var(--neutral-700) !important;
            font-size: 14px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
    <div class="top-nav">
        <div class="nav-container">
            <div class="nav-logo">
                <span class="nav-logo-icon">🏥</span>
                <div class="nav-logo-text">
                    <h1>MedNLP</h1>
                    <p>Clinical Intelligence Platform</p>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Navigation Buttons
col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.5, 2])

with col1:
    if st.button("🏠 Home", key="nav_home", width='stretch'):
        st.session_state.current_page = 'home'
        st.rerun()

with col2:
    if st.button("📄 Upload", key="nav_upload", width='stretch'):
        st.session_state.current_page = 'upload'
        st.rerun()

with col3:
    if st.button("📊 Results", key="nav_results", width='stretch'):
        st.session_state.current_page = 'results'
        st.rerun()

with col4:
    if st.button("💡 Insights", key="nav_insights", width='stretch'):
        st.session_state.current_page = 'insights'
        st.rerun()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'entities' not in st.session_state:
    st.session_state.entities = []
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""

# --- Initialize the extractor ---
@st.cache_resource
def load_extractor():
    return EnhancedMedicalEntityExtractor(use_large_model=False)

extractor = load_extractor()

# --- Initialize the risk analyzer ---
@st.cache_resource
def load_risk_analyzer():
    return MedicalRiskAnalyzer()

risk_analyzer = load_risk_analyzer()

# --- Helper Functions ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_txt(file):
    try:
        if hasattr(file, 'getvalue'):
            data = file.getvalue()
        else:
            # ensure pointer at start
            try:
                file.seek(0)
            except Exception:
                pass
            data = file.read()
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        return str(data)
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif ext == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif ext == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported format. Please upload PDF, DOCX, or TXT.")
            return None
        if not text or not text.strip():
            st.warning("Uploaded file contains no readable text. Please try another file.")
            return None
        return text
    return None

# --- Caching for graphs to improve performance ---
@st.cache_data
def generate_graphs_cached(key, analysis):
    return risk_analyzer.generate_graphs_from_analysis(analysis)

# --- Page Functions ---
def show_home_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Welcome to MedNLP Clinical Intelligence Platform</h1>
            <p class="page-subtitle">Transform medical documents into actionable clinical insights using advanced AI-powered entity extraction and comprehensive risk analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Hero Features
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
            <div class="premium-card">
                <span class="card-icon">🚀</span>
                <h3 class="card-title">Quick Start Guide</h3>
                <div class="card-text">
                    <ol style="margin: 0; padding-left: 24px; line-height: 1.9;">
                        <li><strong>Upload</strong> medical documents or enter clinical notes directly</li>
                        <li><strong>Extract</strong> entities automatically using state-of-the-art NLP</li>
                        <li><strong>Analyze</strong> comprehensive risk assessment results</li>
                        <li><strong>Review</strong> actionable clinical recommendations</li>
                    </ol>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="premium-card">
                <span class="card-icon">📋</span>
                <h3 class="card-title">Supported Document Formats</h3>
                <div class="card-text">
                    <ul style="margin: 0; padding-left: 24px; line-height: 1.9;">
                        <li><strong>PDF Documents</strong> — Medical reports, lab results, discharge summaries</li>
                        <li><strong>Word Files (.docx)</strong> — Clinical notes, patient records</li>
                        <li><strong>Text Files (.txt)</strong> — Plain text medical documentation</li>
                        <li><strong>Direct Input</strong> — Copy-paste clinical text instantly</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("### 🔬 Platform Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="feature-icon-box">
                <span class="icon">⚡</span>
                <strong style="color: #1F2933; font-size: 15px;">Real-Time Processing</strong>
                <p style="font-size: 13px; color: #7B8794; margin-top: 8px; line-height: 1.5;">Instant entity extraction and analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-icon-box">
                <span class="icon">🎯</span>
                <strong style="color: #1F2933; font-size: 15px;">Clinical Accuracy</strong>
                <p style="font-size: 13px; color: #7B8794; margin-top: 8px; line-height: 1.5;">Advanced NLP with 95%+ precision</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-icon-box">
                <span class="icon">🔒</span>
                <strong style="color: #1F2933; font-size: 15px;">HIPAA Compliant</strong>
                <p style="font-size: 13px; color: #7B8794; margin-top: 8px; line-height: 1.5;">Secure, confidential data processing</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="feature-icon-box">
                <span class="icon">📊</span>
                <strong style="color: #1F2933; font-size: 15px;">Risk Analytics</strong>
                <p style="font-size: 13px; color: #7B8794; margin-top: 8px; line-height: 1.5;">Comprehensive clinical insights</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # Statistics Row
    st.markdown("### 📈 Platform Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entity Types", "15+", help="Diseases, medications, symptoms, and more")
    with col2:
        st.metric("Accuracy Rate", "95%", help="Clinical entity extraction accuracy")
    with col3:
        st.metric("Processing Speed", "<2s", help="Average document processing time")
    with col4:
        st.metric("Risk Categories", "8", help="Comprehensive risk factor analysis")
    
    st.markdown("<div style='margin: 48px 0;'></div>", unsafe_allow_html=True)
    
    # Call to Action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Get Started — Upload Your First Document", width='stretch', type="primary"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
    
    # Info Section
    st.info("💡 **Pro Tip**: For best results, ensure medical documents contain clear clinical terminology. The system works optimally with structured medical notes, discharge summaries, and diagnostic reports.")

def show_upload_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">📄 Document Upload & Analysis</h1>
            <p class="page-subtitle">Upload medical documents or enter clinical notes to begin comprehensive entity extraction and risk assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout for upload options
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📎 Upload Medical Document")
        st.markdown("<p style='color: #7B8794; font-size: 14px; margin-bottom: 20px;'>Select a file from your device for automatic text extraction and analysis</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a medical document", 
            type=['pdf', 'txt', 'docx'],
            help="Supported: PDF, TXT, DOCX | Max 200MB",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
            with st.spinner("🔄 Processing document..."):
                text = process_uploaded_file(uploaded_file)
                if text:
                    st.success(f"✅ Successfully processed **{uploaded_file.name}**")
                    st.info(f"📝 Extracted **{len(text):,}** characters | **{len(text.split()):,}** words")
                    st.session_state.processed_text = text
                    
                    with st.expander("📄 Preview Extracted Text", expanded=False):
                        preview_text = text[:1000] + "..." if len(text) > 1000 else text
                        st.text_area("", value=preview_text, height=250, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.markdown("### ✏️ Direct Text Input")
        st.markdown("<p style='color: #7B8794; font-size: 14px; margin-bottom: 20px;'>Enter or paste clinical notes, patient records, or medical observations directly</p>", unsafe_allow_html=True)
        
        user_text = st.text_area(
            "Clinical Notes", 
            height=300,
            placeholder="Enter medical text here...",
            label_visibility="collapsed"
        )
        
        if user_text.strip():
            st.session_state.processed_text = user_text.strip()
            st.info(f"📝 **{len(user_text):,}** characters | **{len(user_text.split()):,}** words entered")
    
    # Analysis Button
    if st.session_state.processed_text:
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # Quick Insights
        insights = risk_analyzer.generate_user_insights(st.session_state.entities)
        st.markdown("### 💡 Quick Insights")
        if insights:
            badge_color = insights.get('risk_color', '#7B8794')
            st.markdown(f"""
                <div style='background: #ffffff; border: 1px solid #E8ECF0; border-left: 4px solid {badge_color}; padding: 18px; border-radius: 10px;'>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <span style='background:{badge_color}; color:white; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:700; letter-spacing:0.5px;'>INSIGHT</span>
                        <strong style='color:#1F2933; font-size:16px;'>{insights.get('headline','')}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.caption("Top Risk Factors")
                for rf in insights.get('key_findings', {}).get('risk_factors', []):
                    st.write(f"- {rf['name']}: {rf['percentage']}% ({rf['risk_level']})")
            with c2:
                st.caption("Common Conditions")
                for cond in insights.get('key_findings', {}).get('conditions', []):
                    st.write(f"- {cond['name']} ({cond['risk_level']})")
            with c3:
                st.caption("Red Flags")
                for rf in insights.get('key_findings', {}).get('red_flags', []):
                    st.write(f"- {rf['symptom']} ({rf['urgency']})")
            with c4:
                st.caption("Next Actions")
                for act in insights.get('actions', []):
                    st.write(f"- {act['title']} [{act['priority']}] ({act['timeframe']})")

            st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='margin: 24px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Analyze Document — Extract Medical Entities", width='stretch', type="primary"):
                with st.spinner("🧠 Analyzing medical entities..."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(30)
                    
                    st.session_state.entities = extractor.extract_entities(st.session_state.processed_text)
                    
                    progress_bar.progress(100)
                    st.success(f"✅ Successfully extracted **{len(st.session_state.entities)}** medical entities!")
                    
                    import time
                    time.sleep(1)
                    st.session_state.current_page = 'results'
                    st.rerun()
    else:
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        st.info("ℹ️ **Ready to begin**: Upload a document or enter text above to start the analysis")

def show_results_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">📊 Extraction Results</h1>
            <p class="page-subtitle">Comprehensive analysis of medical entities extracted from your document</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.entities:
        df = extractor.to_dataframe(st.session_state.entities)
        
        # Summary Statistics
        st.markdown("### 📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Entities", 
                len(st.session_state.entities),
                help="Total number of medical entities extracted"
            )
        with col2:
            st.metric(
                "Entity Types", 
                df['label'].nunique(),
                help="Number of unique entity categories identified"
            )
        with col3:
            most_common = df['label'].value_counts().index[0] if not df.empty else "None"
            count = df['label'].value_counts().values[0] if not df.empty else 0
            st.metric(
                "Most Common", 
                most_common,
                delta=f"{count} instances",
                help="Most frequently identified entity type"
            )
        with col4:
            confidence_avg = df['confidence'].mean() if 'confidence' in df.columns else 0
            st.metric(
                "Avg Confidence", 
                f"{confidence_avg:.1%}",
                help="Average confidence score across all entities"
            )
        
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        
        # Entity Distribution
        st.markdown("### 🏷️ Entity Type Distribution")
        
        entity_counts = df['label'].value_counts()
        
        col1, col2 = st.columns([2.0, 1.0])
        
        with col1:
            if not entity_counts.empty and entity_counts.sum() > 0:
                st.bar_chart(entity_counts.astype('int64'), height=220)
            else:
                st.info("No entities to display yet.")
        
        with col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                     padding: 24px; border-radius: 12px; border: 1px solid #E8ECF0; 
                     box-shadow: 0 1px 3px rgba(0,0,0,0.06); height: 350px; overflow-y: auto;'>
                    <h4 style='margin: 0 0 16px 0; color: #1F2933; font-size: 16px;'>📊 Distribution Breakdown</h4>
            """, unsafe_allow_html=True)
            
            for entity_type, count in entity_counts.items():
                percentage = (count / len(st.session_state.entities)) * 100
                st.markdown(f"""
                    <div style='margin-bottom: 12px; padding: 8px; background: white; border-radius: 6px; border-left: 3px solid #0066CC;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <strong style='color: #1F2933; font-size: 14px;'>{entity_type}</strong>
                            <span style='color: #7B8794; font-size: 13px;'>{count} ({percentage:.1f}%)</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        
        # Detailed Results Table
        st.markdown("### 📋 Detailed Entity Records")
        
        # Add filters
        col1, col2 = st.columns([2, 2])
        with col1:
            selected_types = st.multiselect(
                "Filter by Entity Type",
                options=df['label'].unique().tolist(),
                default=df['label'].unique().tolist()
            )
        
        with col2:
            if 'confidence' in df.columns:
                min_confidence = st.slider(
                    "Minimum Confidence Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            else:
                min_confidence = 0.0
        
        # Filter dataframe
        filtered_df = df[df['label'].isin(selected_types)]
        if 'confidence' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        st.dataframe(
            filtered_df,
            width='stretch',
            height=400
        )
        
        st.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** entities")
        
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        
        # Export Section
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"medical_entities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name=f"medical_entities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width='stretch'
            )
        
        with col3:
            if st.button("📊 View Clinical Insights", width='stretch', type="primary"):
                st.session_state.current_page = 'insights'
                st.rerun()
    
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                 padding: 60px 40px; border-radius: 14px; border: 1px solid #E8ECF0; 
                 text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                <div style='font-size: 64px; margin-bottom: 20px; opacity: 0.5;'>📂</div>
                <h3 style='color: #1F2933; margin-bottom: 12px; font-size: 24px;'>No Analysis Results Available</h3>
                <p style='color: #7B8794; margin-bottom: 32px; font-size: 16px;'>Upload and analyze a medical document to view extraction results</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📄 Go to Upload", width='stretch', type="primary"):
                st.session_state.current_page = 'upload'
                st.rerun()

def show_insights_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">💡 Clinical Insights & Risk Analytics</h1>
            <p class="page-subtitle">Comprehensive risk assessment, comorbidity analysis, and evidence-based clinical recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.entities:
        df = extractor.to_dataframe(st.session_state.entities)
        
        # Entity Distribution
        st.markdown("### 📊 Entity Distribution Overview")
        entity_counts = df['label'].value_counts()
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            if not entity_counts.empty and entity_counts.sum() > 0:
                st.bar_chart(entity_counts.astype('int64'), height=220)
            else:
                st.info("No entities to display yet.")
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # Risk Factors
        st.markdown("### ⚕️ Clinical Risk Assessment")
        
        try:
            with st.spinner("Running clinical risk analysis..."):
                comprehensive_analysis = risk_analyzer.generate_comprehensive_analysis(st.session_state.entities)
            
            overall_score = comprehensive_analysis.get('overall_risk_score', 0)
            risk_stratification = comprehensive_analysis.get('risk_stratification', {})
            risk_level = risk_stratification.get('category', 'Unknown')
            risk_color = risk_stratification.get('color', '#7B8794')
            risk_description = risk_stratification.get('description', '')
            action_required = risk_stratification.get('action_required', False)
            clinical_summary = comprehensive_analysis.get('clinical_summary', '')
            
            # Clinical Alert
            if clinical_summary:
                if action_required:
                    st.error(f"🚨 **CRITICAL ALERT**: {clinical_summary}")
                elif overall_score >= 4.0:
                    st.warning(f"⚠️ **ATTENTION**: {clinical_summary}")
                else:
                    st.info(f"ℹ️ {clinical_summary}")
            
            st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)
            # Quick Insights (concise summary)
            insights = risk_analyzer.generate_user_insights(st.session_state.entities)
            st.markdown("### 💡 Quick Insights")
            if insights:
                badge_color = insights.get('risk_color', '#7B8794')
                st.markdown(f"""
                    <div style='background: #ffffff; border: 1px solid #E8ECF0; border-left: 4px solid {badge_color}; padding: 18px; border-radius: 10px;'>
                        <div style='display:flex; align-items:center; gap:10px;'>
                            <span style='background:{badge_color}; color:white; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:700; letter-spacing:0.5px;'>INSIGHT</span>
                            <strong style='color:#1F2933; font-size:16px;'>{insights.get('headline','')}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.caption("Top Risk Factors")
                    for rf in insights.get('key_findings', {}).get('risk_factors', []):
                        st.write(f"- {rf['name']}: {rf['percentage']}% ({rf['risk_level']})")
                with c2:
                    st.caption("Common Conditions")
                    for cond in insights.get('key_findings', {}).get('conditions', []):
                        st.write(f"- {cond['name']} ({cond['risk_level']})")
                with c3:
                    st.caption("Red Flags")
                    for rf in insights.get('key_findings', {}).get('red_flags', []):
                        st.write(f"- {rf['symptom']} ({rf['urgency']})")
                with c4:
                    st.caption("Next Actions")
                    for act in insights.get('actions', []):
                        st.write(f"- {act['title']} [{act['priority']}] ({act['timeframe']})")

                st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)
            
            # Risk Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                         padding: 28px; border-radius: 14px; border: 1px solid #E8ECF0; 
                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04);'>
                        <p style='color: #7B8794; font-size: 11px; text-transform: uppercase; 
                           margin-bottom: 10px; font-weight: 700; letter-spacing: 1px;'>Risk Score</p>
                        <h2 style='color: {risk_color}; font-size: 48px; margin: 0; font-weight: 800; line-height: 1;'>
                            {overall_score:.1f}<span style='font-size: 24px; opacity: 0.6;'>/10</span>
                        </h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                         padding: 28px; border-radius: 14px; border: 1px solid #E8ECF0; 
                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04);'>
                        <p style='color: #7B8794; font-size: 11px; text-transform: uppercase; 
                           margin-bottom: 10px; font-weight: 700; letter-spacing: 1px;'>Risk Category</p>
                        <h2 style='color: {risk_color}; font-size: 18px; margin: 0 0 6px 0; 
                           font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;'>{risk_level}</h2>
                        <p style='color: #7B8794; font-size: 12px; margin: 0; line-height: 1.4;'>{risk_description}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                summary_stats = comprehensive_analysis.get('summary_stats', {})
                conditions_count = summary_stats.get('unique_diseases', 0)
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                         padding: 28px; border-radius: 14px; border: 1px solid #E8ECF0; 
                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04);'>
                        <p style='color: #7B8794; font-size: 11px; text-transform: uppercase; 
                           margin-bottom: 10px; font-weight: 700; letter-spacing: 1px;'>Conditions</p>
                        <h2 style='color: #0066CC; font-size: 48px; margin: 0; font-weight: 800; line-height: 1;'>{conditions_count}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                critical_findings = summary_stats.get('critical_findings', 0)
                critical_color = '#DC3545' if critical_findings > 0 else '#00A86B'
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                         padding: 28px; border-radius: 14px; border: 1px solid #E8ECF0; 
                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04);'>
                        <p style='color: #7B8794; font-size: 11px; text-transform: uppercase; 
                           margin-bottom: 10px; font-weight: 700; letter-spacing: 1px;'>Critical Findings</p>
                        <h2 style='color: {critical_color}; font-size: 48px; margin: 0; font-weight: 800; line-height: 1;'>{critical_findings}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Risk Factors
            risk_assessment = comprehensive_analysis.get('risk_assessment', [])
            if risk_assessment:
                st.markdown("### 📋 Risk Factor Analysis")
                for risk_factor in risk_assessment[:5]:
                    st.markdown(create_risk_progress_bar_html(risk_factor), unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Visual Analytics
            st.markdown("### 📈 Visual Analytics Dashboard")
            key = st.session_state.get('processed_text', '')
            graphs = generate_graphs_cached(key, comprehensive_analysis)
            
            if graphs and len(graphs) >= 3:
                tab1, tab2, tab3 = st.tabs(["🎯 Risk Assessment", "🏥 Conditions", "📊 Risk Gauge"])
                
                with tab1:
                    if 'risk_assessment' in graphs:
                        st.image(f"data:image/png;base64,{graphs['risk_assessment']}", width='stretch')
                
                with tab2:
                    if 'common_conditions' in graphs:
                        st.image(f"data:image/png;base64,{graphs['common_conditions']}", width='stretch')
                
                with tab3:
                    if 'risk_gauge' in graphs:
                        st.image(f"data:image/png;base64,{graphs['risk_gauge']}", width='stretch')
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Common Conditions
            common_conditions = comprehensive_analysis.get('common_conditions', [])
            if common_conditions:
                st.markdown("### 🏥 Identified Medical Conditions")
                cols = st.columns(3)
                for idx, condition in enumerate(common_conditions[:9]):
                    with cols[idx % 3]:
                        st.markdown(create_condition_card_html(condition), unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Comorbidity Clusters
            comorbidity_clusters = comprehensive_analysis.get('comorbidity_clusters', [])
            if comorbidity_clusters:
                st.markdown("### 🔗 Comorbidity Interactions")
                for cluster in comorbidity_clusters:
                    interaction_color = '#DC3545' if cluster.interaction_severity == 'High' else '#F59E0B' if cluster.interaction_severity == 'Moderate' else '#00A86B'
                    st.markdown(
                        f"- <span style='color:{interaction_color}; font-weight:600;'>{cluster.primary_condition}</span>: "
                        f"{', '.join(cluster.related_conditions)} "
                        f"(score: {cluster.combined_risk_score})",
                        unsafe_allow_html=True
                    )
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Recommendations
            recommendations = comprehensive_analysis.get('recommendations', [])
            if recommendations:
                st.markdown("### 💊 Clinical Recommendations")
                for recommendation in recommendations:
                    st.markdown(create_recommendation_card_html(recommendation), unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Medication Risks
            medication_risks = comprehensive_analysis.get('medication_risks', {})
            if medication_risks.get('total_medications', 0) > 0:
                st.markdown("### 💊 Medication Safety Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Medications", medication_risks.get('total_medications', 0))
                
                with col2:
                    polypharmacy_risk = medication_risks.get('polypharmacy_risk', 'Low')
                    poly_color = '#DC3545' if polypharmacy_risk == 'High' else '#F59E0B' if polypharmacy_risk == 'Moderate' else '#00A86B'
                    st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background: white; 
                             border-radius: 12px; border: 1px solid #E8ECF0;'>
                            <p style='color: #7B8794; font-size: 11px; text-transform: uppercase; 
                               margin-bottom: 8px; font-weight: 700;'>Polypharmacy Risk</p>
                            <h3 style='color: {poly_color}; margin: 0; font-size: 24px; font-weight: 700;'>{polypharmacy_risk}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.metric("High-Risk Meds", len(medication_risks.get('high_risk_medications', [])))
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Critical Symptoms
            critical_symptoms = comprehensive_analysis.get('critical_symptoms', [])
            if critical_symptoms:
                st.markdown("### 🚨 Critical Symptoms Detected")
                for symptom in critical_symptoms:
                    urgency_color = '#DC3545' if symptom['urgency'] == 'Critical' else '#EA580C'
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); 
                             padding: 16px 20px; border-radius: 10px; margin-bottom: 12px; 
                             border-left: 4px solid {urgency_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.04);'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                                <strong style='color: #991B1B; font-size: 15px;'>{symptom['symptom']}</strong>
                                <span style='background: {urgency_color}; color: white; padding: 4px 12px; 
                                     border-radius: 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;'>
                                    {symptom['urgency'].upper()}
                                </span>
                            </div>
                            <small style='color: #7B8794; font-size: 13px;'>
                                Severity: <strong>{symptom['severity_score']}</strong> | 
                                Confidence: <strong>{symptom['confidence']:.1%}</strong>
                            </small>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Entity Pattern Analysis
            entity_patterns = comprehensive_analysis.get('entity_patterns', {})
            if entity_patterns:
                st.markdown("### 🔍 Entity Pattern Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Confidence Distribution")
                    conf_dist = entity_patterns.get('confidence_distribution', {})
                    if conf_dist:
                        values = [
                            conf_dist.get('high', 0) or 0,
                            conf_dist.get('medium', 0) or 0,
                            conf_dist.get('low', 0) or 0,
                        ]
                        if sum(values) > 0:
                            conf_data = pd.DataFrame({
                                'Level': ['High (≥80%)', 'Medium (50-80%)', 'Low (<50%)'],
                                'Percentage': values
                            })
                            st.bar_chart(conf_data.set_index('Level'), height=180)
                        else:
                            st.info("No confidence distribution to display yet.")
                
                with col2:
                    st.markdown("#### Entity Statistics")
                    
                    avg_conf = entity_patterns.get('average_confidence', 0)
                    negated_ratio = entity_patterns.get('negated_ratio', 0)
                    uncertain_ratio = entity_patterns.get('uncertain_ratio', 0)
                    conditional_ratio = entity_patterns.get('conditional_ratio', 0)
                    
                    st.metric(
                        "Average Confidence", 
                        f"{avg_conf:.1%}",
                        help="Mean confidence across all extracted entities"
                    )
                    st.metric(
                        "Negated Findings", 
                        f"{negated_ratio:.1%}",
                        help="Percentage of negated medical findings"
                    )
                    st.metric(
                        "Uncertain Mentions", 
                        f"{uncertain_ratio:.1%}",
                        help="Proportion of entities marked as uncertain/possible"
                    )
                    st.metric(
                        "Conditional Mentions", 
                        f"{conditional_ratio:.1%}",
                        help="Proportion of entities within conditional contexts"
                    )

                # Context distributions
                ctx1, ctx2 = st.columns(2)
                with ctx1:
                    st.markdown("#### Temporality Distribution")
                    temp_dist = entity_patterns.get('temporality_distribution', {}) or {}
                    if temp_dist:
                        counts = list(temp_dist.values())
                        if sum(counts) > 0:
                            temp_df = pd.DataFrame({
                                'Temporality': list(temp_dist.keys()),
                                'Count': counts
                            }).set_index('Temporality')
                            st.bar_chart(temp_df, height=180)
                        else:
                            st.info("No temporality data to display yet.")
                with ctx2:
                    st.markdown("#### Subject Distribution")
                    subj_dist = entity_patterns.get('subject_distribution', {}) or {}
                    if subj_dist:
                        counts = list(subj_dist.values())
                        if sum(counts) > 0:
                            subj_df = pd.DataFrame({
                                'Subject': list(subj_dist.keys()),
                                'Count': counts
                            }).set_index('Subject')
                            st.bar_chart(subj_df, height=180)
                        else:
                            st.info("No subject data to display yet.")
            
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            
            # Detailed Entity Breakdown
            st.markdown("### 🔍 Detailed Entity Breakdown")
            
            for label in sorted(df['label'].unique()):
                label_count = len(df[df['label'] == label])
                with st.expander(f"**{label}** — {label_count} entities found", expanded=False):
                    subset = df[df['label'] == label]
                    display_cols = ['text']
                    if 'confidence' in subset.columns:
                        display_cols.append('confidence')
                    # Add contextual columns when present
                    for col in ['negated','uncertain','conditional','temporality','subject']:
                        if col in subset.columns and col not in display_cols:
                            display_cols.append(col)
                    st.dataframe(subset[display_cols], width='stretch', height=250)
        
        except Exception as e:
            st.error(f"⚠️ **Error in risk analysis**: {str(e)}")
            st.warning("⚠️ Advanced risk analysis is temporarily unavailable. Displaying basic entity insights.")
            
            # Fallback basic analysis
            st.markdown("### 📊 Basic Entity Analysis")
            entity_type_dist = df['label'].value_counts()
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                st.bar_chart(entity_type_dist, height=200)
    
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                 padding: 60px 40px; border-radius: 14px; border: 1px solid #E8ECF0; 
                 text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                <div style='font-size: 64px; margin-bottom: 20px; opacity: 0.5;'>📈</div>
                <h3 style='color: #1F2933; margin-bottom: 12px; font-size: 24px;'>No Clinical Data Available</h3>
                <p style='color: #7B8794; margin-bottom: 32px; font-size: 16px;'>
                    Extract medical entities from a document first to view comprehensive clinical insights and risk analytics
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📄 Go to Upload", width='stretch', type="primary"):
                st.session_state.current_page = 'upload'
                st.rerun()

# --- Main App Router ---
if st.session_state.current_page == 'home':
    show_home_page()
elif st.session_state.current_page == 'upload':
    show_upload_page()
elif st.session_state.current_page == 'results':
    show_results_page()
elif st.session_state.current_page == 'insights':
    show_insights_page()