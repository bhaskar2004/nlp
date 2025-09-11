import streamlit as st
import pandas as pd
from medical_nlp import EnhancedMedicalEntityExtractor
from riskAnalytics import MedicalRiskAnalyzer
import io
import PyPDF2
import docx

# --- Page Config ---
st.set_page_config(
    page_title="MedNLP - Medical Entity Extractor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean & Simple CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* ===== CLEAN DESIGN SYSTEM ===== */
        :root {
            --primary: #2563eb;
            --primary-light: #93c5fd;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --white: #ffffff;
            --green: #10b981;
            --orange: #f59e0b;
            --red: #ef4444;
            --radius: 8px;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            --border: 1px solid var(--gray-200);
        }
        
        /* ===== GLOBAL RESET ===== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* ===== BASE STYLES ===== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: var(--gray-50) !important;
            color: var(--gray-900) !important;
        }
        
        .stApp {
            background: var(--gray-50) !important;
        }
        
        /* ===== CLEAN SIDEBAR ===== */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: var(--white) !important;
            border-right: var(--border) !important;
            padding: 0 !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding: 24px 16px !important;
        }
        
        /* Simple Logo */
        .sidebar-logo {
            padding: 16px;
            margin-bottom: 32px;
            text-align: center;
            border-bottom: var(--border);
        }
        
        .sidebar-logo h1 {
            font-size: 24px !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
            margin: 0 !important;
        }
        
        /* Clean Navigation */
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: var(--white) !important;
            color: var(--gray-700) !important;
            border: var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 12px 16px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            text-align: left !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--primary) !important;
            color: var(--white) !important;
            border-color: var(--primary) !important;
        }
        
        /* ===== MAIN CONTENT ===== */
        .main .block-container {
            padding: 32px 48px !important;
            max-width: 1200px !important;
        }
        
        /* Simple Header */
        .page-header {
            background: var(--white);
            border-radius: var(--radius);
            padding: 32px;
            margin-bottom: 32px;
            border: var(--border);
        }
        
        .page-title {
            font-size: 32px !important;
            font-weight: 700 !important;
            color: var(--gray-900) !important;
            margin: 0 0 8px 0 !important;
        }
        
        .page-subtitle {
            font-size: 16px !important;
            color: var(--gray-600) !important;
            margin: 0 !important;
        }
        
        /* ===== CLEAN COMPONENTS ===== */
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: var(--white);
            border: 2px dashed var(--gray-300);
            border-radius: var(--radius);
            padding: 32px;
            text-align: center;
            transition: border-color 0.2s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary);
        }
        
        [data-testid="stFileUploader"] label {
            font-weight: 500 !important;
            color: var(--gray-700) !important;
        }
        
        /* Text Area */
        .stTextArea textarea {
            background: var(--white) !important;
            border: var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 16px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            transition: border-color 0.2s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            outline: none !important;
        }
        
        .stTextArea label {
            font-weight: 500 !important;
            color: var(--gray-700) !important;
            margin-bottom: 8px !important;
        }
        
        /* Clean Buttons */
        .stButton > button {
            background: var(--primary) !important;
            color: var(--white) !important;
            border: none !important;
            border-radius: var(--radius) !important;
            padding: 12px 24px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background: #1d4ed8 !important;
            transform: translateY(-1px);
        }
        
        /* Simple Metrics */
        [data-testid="metric-container"] {
            background: var(--white);
            border-radius: var(--radius);
            padding: 24px;
            border: var(--border);
            text-align: center;
        }
        
        [data-testid="metric-container"] label {
            font-weight: 500 !important;
            color: var(--gray-600) !important;
            font-size: 12px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
        }
        
        /* Clean DataFrame */
        .stDataFrame {
            border-radius: var(--radius) !important;
            overflow: hidden !important;
            border: var(--border) !important;
        }
        
        [data-testid="stDataFrameResizable"] {
            background: var(--white) !important;
        }
        
        /* Simple Alerts */
        .stAlert {
            border-radius: var(--radius) !important;
            border: none !important;
            padding: 16px !important;
            margin: 16px 0 !important;
        }
        
        .stSuccess {
            background: #ecfdf5 !important;
            color: #065f46 !important;
            border-left: 4px solid var(--green) !important;
        }
        
        .stInfo {
            background: #eff6ff !important;
            color: #1e40af !important;
            border-left: 4px solid var(--primary) !important;
        }
        
        .stWarning {
            background: #fffbeb !important;
            color: #92400e !important;
            border-left: 4px solid var(--orange) !important;
        }
        
        .stError {
            background: #fef2f2 !important;
            color: #991b1b !important;
            border-left: 4px solid var(--red) !important;
        }
        
        /* Clean Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--white);
            border-radius: var(--radius);
            border: var(--border);
            padding: 4px;
            margin-bottom: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            color: var(--gray-600);
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--primary);
            color: var(--white);
        }
        
        /* Clean Spinner */
        .stSpinner > div {
            border-color: var(--primary) !important;
        }
        
        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 16px !important;
            }
            
            .page-header {
                padding: 24px 20px;
            }
            
            .page-title {
                font-size: 24px !important;
            }
            
            [data-testid="stFileUploader"] {
                padding: 24px 16px;
            }
        }
        
        /* ===== CHARTS ===== */
        .js-plotly-plot {
            border-radius: var(--radius) !important;
            border: var(--border) !important;
        }
        
        /* ===== FOCUS STATES ===== */
        button:focus,
        input:focus,
        textarea:focus {
            outline: 2px solid var(--primary-light) !important;
            outline-offset: 2px !important;
        }
        
        /* Remove unnecessary decorations */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--gray-900) !important;
        }
        
        /* Clean scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--gray-100);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--gray-400);
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'entities' not in st.session_state:
    st.session_state.entities = []
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""

# --- Clean Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <h1>🏥 MedNLP</h1>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 Home", key="nav_home"):
        st.session_state.current_page = 'home'
    if st.button("📄 Upload", key="nav_upload"):
        st.session_state.current_page = 'upload'
    if st.button("📊 Results", key="nav_results"):
        st.session_state.current_page = 'results'
    if st.button("💡 Insights", key="nav_insights"):
        st.session_state.current_page = 'insights'

# --- Initialize the extractor ---
@st.cache_resource
def load_extractor():
    return EnhancedMedicalEntityExtractor(use_large_model=False)

extractor = load_extractor()

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
        return str(file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf': return extract_text_from_pdf(uploaded_file)
        elif ext == 'docx': return extract_text_from_docx(uploaded_file)
        elif ext == 'txt': return extract_text_from_txt(uploaded_file)
        else: st.error("Unsupported format"); return None
    return None

# --- Clean Page Functions ---
def show_home_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Welcome to MedNLP</h1>
            <p class="page-subtitle">Extract medical entities from patient records with AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 **Quick Start**")
        st.write("1. Upload a medical document")
        st.write("2. Or paste patient notes directly")
        st.write("3. Get instant entity extraction")
        st.write("4. View detailed analytics")
    
    with col2:
        st.markdown("### 📋 **Supported Formats**")
        st.write("• PDF documents")
        st.write("• Word documents (.docx)")
        st.write("• Plain text files")
        st.write("• Direct text input")

def show_upload_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Upload Medical Record</h1>
            <p class="page-subtitle">Choose a file or enter text directly</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a medical document", 
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    st.markdown("### ✏️ Or Enter Text Directly")
    user_text = st.text_area(
        "Patient notes", 
        height=200,
        placeholder="Enter medical notes, symptoms, diagnoses, treatments..."
    )
    
    # Process uploaded file
    if uploaded_file:
        with st.spinner("Processing file..."):
            text = process_uploaded_file(uploaded_file)
            if text:
                st.success(f"✅ File '{uploaded_file.name}' processed successfully!")
                st.session_state.processed_text = text
                with st.expander("Preview extracted text"):
                    st.text(text[:500] + "..." if len(text) > 500 else text)
    
    # Process direct text input
    elif user_text.strip():
        st.session_state.processed_text = user_text.strip()
    
    # Extract entities button
    if st.session_state.processed_text:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Extract Entities", use_container_width=True):
                with st.spinner("Extracting medical entities..."):
                    st.session_state.entities = extractor.extract_entities(st.session_state.processed_text)
                    st.session_state.current_page = 'results'
                    st.rerun()

def show_results_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Analysis Results</h1>
            <p class="page-subtitle">Extracted medical entities from your document</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.entities:
        df = extractor.to_dataframe(st.session_state.entities)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(st.session_state.entities))
        with col2:
            st.metric("Unique Types", df['label'].nunique())
        with col3:
            most_common = df['label'].value_counts().index[0] if not df.empty else "None"
            st.metric("Most Common", most_common)
        with col4:
            confidence_avg = df['confidence'].mean() if 'confidence' in df.columns else 0
            st.metric("Avg Confidence", f"{confidence_avg:.1%}")
        
        st.markdown("### 📋 Detailed Results")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="medical_entities.csv",
            mime="text/csv"
        )
    else:
        st.info("🔍 No results available yet. Please upload a document first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = 'upload'
            st.rerun()

def show_insights_page():
    st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Insights & Analytics</h1>
            <p class="page-subtitle">Visual analysis of extracted medical data</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.entities:
        df = extractor.to_dataframe(st.session_state.entities)
        
        # Entity distribution chart
        st.markdown("### 📊 Entity Distribution")
        entity_counts = df['label'].value_counts()
        st.bar_chart(entity_counts)
        
        # Risk analysis section
        st.markdown("### ⚕️ Risk Assessment")
        try:
            risk_analyzer = MedicalRiskAnalyzer()
            comprehensive_analysis = risk_analyzer.generate_comprehensive_analysis(st.session_state.entities)
            overall_score = comprehensive_analysis.get('overall_risk_score', 0)
            risk_level = risk_analyzer._get_risk_level(overall_score)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{overall_score:.1f}/10")
            with col2:
                st.metric("Risk Level", risk_level.title())
            with col3:
                st.metric("Conditions Found", len([e for e in st.session_state.entities if e.get('label') == 'CONDITION']))
            
            # Generate and display graphs
            graphs = risk_analyzer.generate_graphs_from_analysis(comprehensive_analysis)
            if graphs:
                col1, col2 = st.columns(2)
                with col1:
                    if 'risk_assessment' in graphs:
                        st.image(f"data:image/png;base64,{graphs['risk_assessment']}", caption="Risk Assessment")
                with col2:
                    if 'common_conditions' in graphs:
                        st.image(f"data:image/png;base64,{graphs['common_conditions']}", caption="Common Conditions")
        
        except Exception as e:
            st.warning("Risk analysis unavailable. Showing basic insights only.")
        
        # Entity details
        if st.checkbox("Show detailed entity breakdown"):
            st.markdown("### 🔍 Entity Details")
            for label in df['label'].unique():
                with st.expander(f"{label} ({len(df[df['label'] == label])} found)"):
                    subset = df[df['label'] == label]
                    st.dataframe(subset[['text', 'confidence']] if 'confidence' in subset.columns else subset[['text']])
    else:
        st.info("📈 No data to analyze yet. Please extract entities first.")
        if st.button("← Go to Upload"):
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