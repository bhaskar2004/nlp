import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, Doc
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import combinations
import re
import io
import os
from pathlib import Path

# File processing imports
import PyPDF2
import docx
import fitz  # PyMuPDF - better PDF processing
from PIL import Image
import pytesseract  # OCR for images in PDFs

# Install required packages:
# pip install spacy pandas numpy scikit-learn networkx PyPDF2 python-docx PyMuPDF pillow pytesseract
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg  # For better word vectors

@dataclass
class MedicalEntity:
    """Enhanced medical entity with comprehensive metadata"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0
    context: str = ""
    negated: bool = False
    temporal_modifier: Optional[str] = None
    severity: Optional[str] = None
    certainty: Optional[str] = None
    subject: str = "patient"  # patient, family_member, etc.
    source_method: str = ""  # dictionary, pattern, contextual, etc.
    normalized_form: Optional[str] = None
    cui: Optional[str] = None  # Concept Unique Identifier (if available)
    semantic_type: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    page_number: Optional[int] = None  # For multi-page documents
    file_source: Optional[str] = None  # Source file name

@dataclass
class ProcessingResult:
    """Container for processing results"""
    entities: List[MedicalEntity]
    text: str
    metadata: Dict
    processing_time: float
    word_count: int
    confidence_stats: Dict

class FileProcessor:
    """Enhanced file processing with better error handling and OCR support"""
    
    def __init__(self):
        pass
    
    def process_file(self, file_path: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Process a file (PDF, DOCX, or image) and extract text.
        Returns the extracted text and metadata about the file.
        """
        text = ""
        metadata = {}
        
        # Handle PDF files
        if filename and filename.lower().endswith('.pdf'):
            text = self._process_pdf(file_path)
            metadata['file_type'] = 'pdf'
        # Handle DOCX files
        elif filename and filename.lower().endswith('.docx'):
            text = self._process_docx(file_path)
            metadata['file_type'] = 'docx'
        # Handle image files
        elif filename and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            text = self._process_image(file_path)
            metadata['file_type'] = 'image'
        # Handle text files or raw text
        else:
            text = self._process_text(file_path)
            metadata['file_type'] = 'text'
        
        # Common metadata
        metadata['filename'] = filename or "unknown"
        metadata['text_length'] = len(text)
        metadata['word_count'] = len(text.split())
        
        return text, metadata
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using PyMuPDF and OCR if needed"""
        text = ""
        try:
            # Use PyMuPDF for PDF processing
            with fitz.open(file_path) as pdf_document:
                for page in pdf_document:
                    # Extract text from each page
                    page_text = page.get_text()
                    if page_text:
                        text += page_text
                    else:
                        # If no text found, use OCR
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        text += ocr_text
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
        
        return text
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error processing DOCX file {file_path}: {e}")
        
        return text.strip()
    
    def _process_image(self, file_path: str) -> str:
        """Extract text from image file using OCR"""
        text = ""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error processing image file {file_path}: {e}")
        
        return text
    
    def _process_text(self, file_path: Union[str, bytes, io.BytesIO]) -> str:
        """Process raw text or text file"""
        text = ""
        try:
            if isinstance(file_path, (bytes, io.BytesIO)):
                # Bytes input (e.g., from an uploaded file)
                text = file_path.read().decode('utf-8', errors='ignore')
            else:
                # Regular file path
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
        except Exception as e:
            print(f"Error processing text input {file_path}: {e}")
        
        return text.strip()

class EnhancedMedicalEntityExtractor:
    """
    Advanced medical entity extraction system with file processing capabilities
    """
    
    def __init__(self, use_large_model: bool = False):
        # Load spaCy model (use large model for better accuracy)
        model_name = "en_core_web_lg" if use_large_model else "en_core_web_sm"
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom pipeline components
        self._add_custom_components()
        
        # Initialize comprehensive medical knowledge base
        self._initialize_enhanced_dictionaries()
        
        # Initialize pattern matchers
        self._initialize_pattern_matchers()
        
        # Initialize contextual analyzers
        self._initialize_contextual_analyzers()
        
        # Load medical abbreviations and normalize forms
        self._initialize_normalization_maps()
        
        # Initialize TF-IDF for semantic similarity
        self._initialize_semantic_analyzer()
        
        # Medical concept relationships
        self._initialize_concept_graph()
        
        # Initialize file processor
        self.file_processor = FileProcessor()
    
    def _add_custom_components(self):
        """Add custom pipeline components for medical text processing"""
        
        # Register custom extension attributes
        if not Span.has_extension("negated"):
            Span.set_extension("negated", default=False)
        if not Doc.has_extension("negations"):
            Doc.set_extension("negations", default=[])
        
        # Add negation detection component
        @self.nlp.component("negation_detector")
        def negation_component(doc):
            # Negation patterns
            negation_patterns = [
                r'\b(?:no|not|without|absence|absent|negative|deny|denies|ruled out)\b',
                r'\bnon-\w+',
                r'\bun\w+',
                r'\b(?:never|neither|nor)\b'
            ]
            
            # Combine patterns
            combined_pattern = '|'.join(negation_patterns)
            
            # Find negation triggers
            negations = []
            for match in re.finditer(combined_pattern, doc.text, re.IGNORECASE):
                # Mark entities within scope as negated
                start_pos = match.start()
                # Negation scope is typically 5-7 tokens after the trigger
                scope_end = min(len(doc.text), start_pos + 100)
                negations.append((start_pos, scope_end))
            
            doc._.negations = negations
            return doc
        
        # Add the component to the pipeline if not already present
        if "negation_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("negation_detector", last=True)
    
    def _initialize_enhanced_dictionaries(self):
        """Initialize comprehensive medical dictionaries with semantic relationships"""
        
        # Diseases with synonyms and related terms
        self.diseases = {
            'diabetes': ['diabetes mellitus', 'dm', 'diabetic', 'hyperglycemia', 'type 1 diabetes', 'type 2 diabetes', 't1dm', 't2dm'],
            'hypertension': ['high blood pressure', 'htn', 'hypertensive', 'elevated bp'],
            'myocardial infarction': ['heart attack', 'mi', 'stemi', 'nstemi', 'acute mi'],
            'pneumonia': ['pneumonitis', 'lung infection', 'pulmonary infection'],
            'asthma': ['bronchial asthma', 'asthmatic', 'reactive airway disease'],
            'copd': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
            'heart failure': ['congestive heart failure', 'chf', 'cardiac failure', 'hf'],
            'stroke': ['cerebrovascular accident', 'cva', 'brain attack', 'cerebral infarction'],
            'depression': ['major depressive disorder', 'mdd', 'depressive episode'],
            'anxiety': ['anxiety disorder', 'generalized anxiety', 'panic disorder'],
            'arthritis': ['osteoarthritis', 'rheumatoid arthritis', 'ra', 'oa', 'joint inflammation'],
            'cancer': ['malignancy', 'tumor', 'neoplasm', 'carcinoma', 'sarcoma', 'lymphoma'],
            'infection': ['sepsis', 'bacteremia', 'infectious disease'],
            'kidney disease': ['chronic kidney disease', 'ckd', 'renal failure', 'nephropathy'],
            'liver disease': ['hepatitis', 'cirrhosis', 'hepatic failure']
        }
        
        # Symptoms with variations
        self.symptoms = {
            'chest pain': ['chest discomfort', 'thoracic pain', 'angina', 'chest tightness'],
            'shortness of breath': ['dyspnea', 'breathlessness', 'sob', 'respiratory distress'],
            'abdominal pain': ['stomach pain', 'belly pain', 'abdominal discomfort'],
            'headache': ['cephalgia', 'head pain', 'migraine', 'tension headache'],
            'nausea': ['feeling sick', 'queasiness', 'stomach upset'],
            'vomiting': ['emesis', 'throwing up', 'retching'],
            'fever': ['pyrexia', 'elevated temperature', 'febrile'],
            'fatigue': ['tiredness', 'exhaustion', 'weakness', 'malaise'],
            'dizziness': ['vertigo', 'lightheadedness', 'giddiness'],
            'palpitations': ['heart racing', 'rapid heartbeat', 'tachycardia'],
            'swelling': ['edema', 'fluid retention', 'bloating'],
            'rash': ['skin eruption', 'dermatitis', 'skin irritation'],
            'joint pain': ['arthralgia', 'joint ache', 'joint stiffness'],
            'back pain': ['lumbar pain', 'spinal pain', 'backache'],
            'cough': ['productive cough', 'dry cough', 'persistent cough']
        }
        
        # Medications with brand names and generics
        self.medications = {
            'aspirin': ['acetylsalicylic acid', 'asa', 'bayer'],
            'metformin': ['glucophage', 'fortamet', 'glumetza'],
            'lisinopril': ['prinivil', 'zestril', 'ace inhibitor'],
            'atenolol': ['tenormin', 'beta blocker'],
            'simvastatin': ['zocor', 'statin'],
            'omeprazole': ['prilosec', 'proton pump inhibitor', 'ppi'],
            'insulin': ['insulin glargine', 'lantus', 'humalog', 'novolog'],
            'warfarin': ['coumadin', 'anticoagulant'],
            'prednisone': ['prednisolone', 'corticosteroid', 'steroid'],
            'morphine': ['opioid', 'narcotic', 'pain medication'],
            'furosemide': ['lasix', 'diuretic', 'water pill'],
            'albuterol': ['ventolin', 'proair', 'bronchodilator']
        }
        
        # Enhanced tests and procedures
        self.tests = {
            'electrocardiogram': ['ecg', 'ekg', '12-lead ecg'],
            'computed tomography': ['ct scan', 'cat scan', 'ct'],
            'magnetic resonance imaging': ['mri', 'mri scan'],
            'complete blood count': ['cbc', 'full blood count'],
            'blood urea nitrogen': ['bun'],
            'brain natriuretic peptide': ['bnp', 'pro-bnp'],
            'hemoglobin a1c': ['hba1c', 'glycated hemoglobin'],
            'thyroid stimulating hormone': ['tsh'],
            'prostate specific antigen': ['psa'],
            'echocardiogram': ['echo', 'cardiac ultrasound'],
            'stress test': ['exercise stress test', 'nuclear stress test'],
            'colonoscopy': ['lower endoscopy'],
            'endoscopy': ['upper endoscopy', 'egd']
        }
        
        # Body parts with anatomical variations
        self.body_parts = {
            'heart': ['cardiac', 'myocardium', 'coronary'],
            'lung': ['pulmonary', 'respiratory', 'bronchial'],
            'kidney': ['renal', 'nephro'],
            'liver': ['hepatic', 'hepato'],
            'brain': ['cerebral', 'neurological', 'cns'],
            'stomach': ['gastric', 'gastro'],
            'intestine': ['bowel', 'gut', 'gastrointestinal'],
            'blood vessel': ['vascular', 'arterial', 'venous'],
            'bone': ['skeletal', 'osseous'],
            'muscle': ['muscular', 'myopathy']
        }
        
        # Severity indicators
        self.severity_indicators = {
            'mild': ['slight', 'minor', 'minimal', 'low-grade'],
            'moderate': ['moderate', 'medium', 'intermediate'],
            'severe': ['severe', 'serious', 'marked', 'significant', 'profound'],
            'acute': ['sudden', 'rapid', 'immediate', 'emergent'],
            'chronic': ['long-term', 'persistent', 'ongoing', 'longstanding']
        }
        
        # Certainty indicators
        self.certainty_indicators = {
            'definite': ['confirmed', 'diagnosed', 'established', 'proven'],
            'probable': ['likely', 'probable', 'suspected', 'presumed'],
            'possible': ['possible', 'potential', 'may have', 'could be'],
            'rule_out': ['rule out', 'r/o', 'exclude', 'differential']
        }
    
    def _initialize_pattern_matchers(self):
        """Initialize advanced pattern matchers"""
        
        # Phrase matchers for multi-word entities
        self.phrase_matchers = {}
        
        # Create matchers for each category
        categories = {
            'DISEASE': self.diseases,
            'SYMPTOM': self.symptoms,
            'MEDICATION': self.medications,
            'TEST': self.tests,
            'BODY_PART': self.body_parts
        }
        
        for category, terms_dict in categories.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = []
            
            for main_term, synonyms in terms_dict.items():
                all_terms = [main_term] + synonyms
                for term in all_terms:
                    patterns.append(self.nlp.make_doc(term.lower()))
            
            if patterns:
                matcher.add(category, patterns)
            self.phrase_matchers[category] = matcher
        
        # Advanced rule-based matcher
        self.rule_matcher = Matcher(self.nlp.vocab)
        self._add_advanced_patterns()
    
    def _add_advanced_patterns(self):
        """Add sophisticated linguistic patterns"""
        
        # Dosage patterns
        dosage_patterns = [
            [{"TEXT": {"REGEX": r"\d+\.?\d*"}}, {"LOWER": {"IN": ["mg", "g", "ml", "cc", "units", "iu", "mcg", "µg"]}}],
            [{"LOWER": {"IN": ["once", "twice", "three", "four"]}}, {"LOWER": {"IN": ["daily", "times"]}}, {"LOWER": {"IN": ["daily", "day", "per"]}, "OP": "?"}],
            [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "times"}, {"LOWER": {"IN": ["daily", "day", "per"]}}]
        ]
        
        for i, pattern in enumerate(dosage_patterns):
            self.rule_matcher.add(f"DOSAGE_{i}", [pattern])
        
        # Temporal patterns
        temporal_patterns = [
            [{"TEXT": {"REGEX": r"\d{1,2}"}}, {"LOWER": {"IN": ["days", "weeks", "months", "years"]}}, {"LOWER": {"IN": ["ago", "prior", "before"]}}],
            [{"LOWER": "for"}, {"TEXT": {"REGEX": r"\d+"}}, {"LOWER": {"IN": ["days", "weeks", "months", "years"]}}],
            [{"LOWER": {"IN": ["since", "from"]}}, {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{2,4}"}}]
        ]
        
        for i, pattern in enumerate(temporal_patterns):
            self.rule_matcher.add(f"TIME_{i}", [pattern])
        
        # Medical procedure patterns
        procedure_patterns = [
            [{"LOWER": {"REGEX": r"\w+ectomy"}}],  # surgeries ending in -ectomy
            [{"LOWER": {"REGEX": r"\w+scopy"}}],  # procedures ending in -scopy
            [{"LOWER": {"REGEX": r"\w+plasty"}}],  # procedures ending in -plasty
        ]
        
        for i, pattern in enumerate(procedure_patterns):
            self.rule_matcher.add(f"PROCEDURE_{i}", [pattern])
    
    def _initialize_contextual_analyzers(self):
        """Initialize contextual analysis components"""
        
        # Negation triggers and their scope
        self.negation_triggers = [
            'no', 'not', 'without', 'absence', 'absent', 'negative', 'deny', 'denies',
            'ruled out', 'free of', 'clear of', 'unremarkable', 'within normal limits'
        ]
        
        # Uncertainty indicators
        self.uncertainty_indicators = [
            'possible', 'probable', 'likely', 'suspected', 'questionable', 'uncertain',
            'may', 'might', 'could', 'perhaps', 'appears', 'seems'
        ]
        
        # Subject indicators (who has the condition)
        self.subject_indicators = {
            'patient': ['patient', 'pt', 'he', 'she', 'they'],
            'family': ['family', 'mother', 'father', 'parent', 'sibling', 'relative'],
            'history': ['history', 'past', 'previous', 'former']
        }
    
    def _initialize_normalization_maps(self):
        """Initialize normalization and standardization maps"""
        
        # Enhanced abbreviations dictionary
        self.abbreviations = {
            # Diseases
            'mi': 'myocardial infarction',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'chf': 'congestive heart failure',
            'cad': 'coronary artery disease',
            'ckd': 'chronic kidney disease',
            'copd': 'chronic obstructive pulmonary disease',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'af': 'atrial fibrillation',
            'pvd': 'peripheral vascular disease',
            
            # Symptoms
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            
            # Tests
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'pt/ptt': 'prothrombin time/partial thromboplastin time',
            'inr': 'international normalized ratio',
            'bnp': 'brain natriuretic peptide',
            'tsh': 'thyroid stimulating hormone',
            'psa': 'prostate specific antigen',
            
            # Body systems
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'cv': 'cardiovascular',
            'resp': 'respiratory',
            'neuro': 'neurological',
            'psych': 'psychiatric',
            'derm': 'dermatological',
            'ent': 'ear nose throat',
            
            # Medications
            'asa': 'aspirin',
            'hctz': 'hydrochlorothiazide',
            'ace-i': 'ace inhibitor',
            'arb': 'angiotensin receptor blocker',
            'ppi': 'proton pump inhibitor',
            'nsaid': 'nonsteroidal anti-inflammatory drug'
        }
        
        # Create reverse mapping for normalization
        self.normalization_map = {}
        for category, terms_dict in [
            ('DISEASE', self.diseases),
            ('SYMPTOM', self.symptoms),
            ('MEDICATION', self.medications),
            ('TEST', self.tests),
            ('BODY_PART', self.body_parts)
        ]:
            for canonical, variants in terms_dict.items():
                self.normalization_map[canonical] = canonical
                for variant in variants:
                    self.normalization_map[variant] = canonical
    
    def _initialize_semantic_analyzer(self):
        """Initialize semantic similarity analyzer"""
        
        # Collect all medical terms for TF-IDF
        all_terms = []
        for terms_dict in [self.diseases, self.symptoms, self.medications, self.tests, self.body_parts]:
            for main_term, synonyms in terms_dict.items():
                all_terms.extend([main_term] + synonyms)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            max_features=5000
        )
        
        try:
            if all_terms:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_terms)
                self.term_index = {term: i for i, term in enumerate(all_terms)}
            else:
                self.tfidf_matrix = None
                self.term_index = {}
        except Exception as e:
            print(f"Warning: Could not initialize semantic analyzer: {e}")
            self.tfidf_matrix = None
            self.term_index = {}
    
    def _initialize_concept_graph(self):
        """Initialize medical concept relationship graph"""
        
        self.concept_graph = nx.Graph()
        
        # Add nodes and edges based on semantic relationships
        categories = {
            'DISEASE': self.diseases,
            'SYMPTOM': self.symptoms,
            'MEDICATION': self.medications,
            'TEST': self.tests,
            'BODY_PART': self.body_parts
        }
        
        for category, terms_dict in categories.items():
            for main_term, synonyms in terms_dict.items():
                # Add main term
                self.concept_graph.add_node(main_term, category=category)
                
                # Add synonyms and connect them
                for synonym in synonyms:
                    self.concept_graph.add_node(synonym, category=category)
                    self.concept_graph.add_edge(main_term, synonym, relation='synonym')
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common medical formatting
        text = re.sub(r'\b(\d+)\s*-\s*(\d+)\b', r'\1-\2', text)  # ranges
        text = re.sub(r'\b(\d+\.?\d*)\s*(mg|g|ml|cc)\b', r'\1\2', text)  # dosages
        
        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        # Standardize negation patterns
        text = re.sub(r'\bno\s+(?:evidence|signs?|symptoms?)\s+of\b', 'negative for', text, flags=re.IGNORECASE)
        text = re.sub(r'\bruled?\s+out\b', 'negative for', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_with_phrase_matcher(self, doc) -> List[MedicalEntity]:
        """Extract entities using phrase matchers"""
        entities = []
        
        for category, matcher in self.phrase_matchers.items():
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                
                # Get canonical form
                canonical = self.normalization_map.get(span.text.lower(), span.text.lower())
                
                entity = MedicalEntity(
                    text=span.text,
                    label=category,
                    start=span.start_char,
                    end=span.end_char,
                    confidence=0.95,
                    normalized_form=canonical,
                    source_method="phrase_matcher"
                )
                
                entities.append(entity)
        
        return entities
    
    def extract_with_patterns(self, doc) -> List[MedicalEntity]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        matches = self.rule_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id].split('_')[0]
            
            entity = MedicalEntity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=0.88,
                source_method="rule_pattern"
            )
            
            entities.append(entity)
        
        return entities
    
    def extract_contextual_entities(self, doc) -> List[MedicalEntity]:
        """Enhanced contextual entity extraction"""
        entities = []
        
        # Use spaCy's built-in NER
        for ent in doc.ents:
            label = self.map_spacy_label(ent.label_)
            if label:
                entity = MedicalEntity(
                    text=ent.text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.75,
                    source_method="spacy_ner"
                )
                entities.append(entity)
        
        # Enhanced dependency-based extraction
        for token in doc:
            if self.is_medical_candidate(token):
                label = self.classify_medical_term_enhanced(token, doc)
                if label:
                    entity = MedicalEntity(
                        text=token.text,
                        label=label,
                        start=token.idx,
                        end=token.idx + len(token.text),
                        confidence=0.70,
                        source_method="contextual"
                    )
                    entities.append(entity)
        
        return entities
    
    def is_medical_candidate(self, token) -> bool:
        """Enhanced medical term candidate detection"""
        
        # Basic filters
        if token.is_stop or token.is_punct or len(token.text) < 3:
            return False
        
        # Medical morphological patterns
        medical_suffixes = [
            'osis', 'itis', 'emia', 'uria', 'pathy', 'therapy', 'ectomy', 
            'otomy', 'scopy', 'graphy', 'plasty', 'ology', 'ologist'
        ]
        
        medical_prefixes = [
            'cardio', 'neuro', 'gastro', 'hepato', 'nephro', 'pulmo', 
            'osteo', 'arthro', 'dermato', 'hemato', 'pneumo', 'encephalo'
        ]
        
        text_lower = token.text.lower()
        
        # Check morphological patterns
        if any(text_lower.endswith(suffix) for suffix in medical_suffixes):
            return True
        
        if any(text_lower.startswith(prefix) for prefix in medical_prefixes):
            return True
        
        # Check if it's in our medical vocabularies
        for terms_dict in [self.diseases, self.symptoms, self.medications, self.tests, self.body_parts]:
            for main_term, synonyms in terms_dict.items():
                if text_lower in [main_term] + synonyms:
                    return True
        
        # Use word embeddings if available
        if hasattr(self.nlp.vocab, 'has_vector') and self.nlp.vocab.has_vector(token.text):
            # Check similarity with known medical terms
            similarities = []
            for category in ['disease', 'symptom', 'medication', 'test', 'anatomy']:
                if self.nlp.vocab.has_vector(category):
                    try:
                        sim = token.similarity(self.nlp(category)[0])
                        similarities.append(sim)
                    except:
                        similarities.append(0.0)
            
            if similarities and max(similarities) > 0.5:
                return True
        
        return False
    
    def classify_medical_term_enhanced(self, token, doc) -> Optional[str]:
        """Enhanced medical term classification"""
        
        text_lower = token.text.lower()
        
        # Rule-based classification with enhanced patterns
        disease_patterns = ['osis', 'itis', 'pathy', 'syndrome', 'disorder', 'disease']
        procedure_patterns = ['ectomy', 'otomy', 'scopy', 'plasty', 'surgery', 'operation']
        symptom_patterns = ['pain', 'ache', 'ness', 'difficulty']
        
        if any(pattern in text_lower for pattern in disease_patterns):
            return 'DISEASE'
        elif any(pattern in text_lower for pattern in procedure_patterns):
            return 'PROCEDURE'
        elif any(pattern in text_lower for pattern in symptom_patterns):
            return 'SYMPTOM'
        
        # Context-based classification
        context_window = 3
        surrounding_tokens = []
        
        start_idx = max(0, token.i - context_window)
        end_idx = min(len(doc), token.i + context_window + 1)
        
        for i in range(start_idx, end_idx):
            if i != token.i:
                surrounding_tokens.append(doc[i].text.lower())
        
        context_text = ' '.join(surrounding_tokens)
        
        # Classification based on context
        if any(word in context_text for word in ['diagnose', 'diagnosed', 'condition', 'disease']):
            return 'DISEASE'
        elif any(word in context_text for word in ['complains', 'reports', 'feels', 'experiencing']):
            return 'SYMPTOM'
        elif any(word in context_text for word in ['prescribed', 'taking', 'medication', 'drug']):
            return 'MEDICATION'
        elif any(word in context_text for word in ['test', 'scan', 'examination', 'lab']):
            return 'TEST'
        elif any(word in context_text for word in ['procedure', 'surgery', 'operation']):
            return 'PROCEDURE'
        
        # Use semantic similarity if available
        if self.tfidf_matrix is not None:
            similarity_scores = self.compute_semantic_similarity(token.text)
            if similarity_scores:
                best_match, best_score, category = max(similarity_scores, key=lambda x: x[1])
                if best_score > 0.3:
                    return category
        
        return None
    
    def compute_semantic_similarity(self, text: str) -> List[Tuple[str, float, str]]:
        """Compute semantic similarity using TF-IDF"""
        
        if self.tfidf_matrix is None:
            return []
        
        try:
            # Transform the input text
            text_vector = self.tfidf_vectorizer.transform([text])
            
            # Compute similarities
            similarities = cosine_similarity(text_vector, self.tfidf_matrix).flatten()
            
            # Get top matches
            top_indices = similarities.argsort()[-10:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold
                    # Find which term this corresponds to
                    term = list(self.term_index.keys())[idx]
                    category = self.get_term_category(term)
                    results.append((term, similarities[idx], category))
            
            return results
        except Exception as e:
            print(f"Warning: Semantic similarity computation failed: {e}")
            return []
    
    def get_term_category(self, term: str) -> str:
        """Get category for a given term"""
        
        term_lower = term.lower()
        
        for category, terms_dict in [
            ('DISEASE', self.diseases),
            ('SYMPTOM', self.symptoms),
            ('MEDICATION', self.medications),
            ('TEST', self.tests),
            ('BODY_PART', self.body_parts)
        ]:
            for main_term, synonyms in terms_dict.items():
                if term_lower in [main_term] + synonyms:
                    return category
        
        return 'UNKNOWN'
    
    def analyze_context(self, entity: MedicalEntity, doc, text: str) -> MedicalEntity:
        """Enhanced contextual analysis of entities"""
        
        # Get surrounding context
        context_window = 100
        start_context = max(0, entity.start - context_window)
        end_context = min(len(text), entity.end + context_window)
        context = text[start_context:end_context]
        entity.context = context

        # Negation detection
        entity.negated = False
        if hasattr(doc._, "negations"):
            for neg_start, neg_end in doc._.negations:
                if entity.start >= neg_start and entity.end <= neg_end:
                    entity.negated = True
                    break

        # Temporal modifier detection
        for indicator in self.severity_indicators.get('acute', []) + self.severity_indicators.get('chronic', []):
            if indicator in context.lower():
                entity.temporal_modifier = indicator
                break

        # Severity detection
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if indicator in context.lower():
                    entity.severity = severity
                    break

        # Certainty detection
        for certainty, indicators in self.certainty_indicators.items():
            for indicator in indicators:
                if indicator in context.lower():
                    entity.certainty = certainty
                    break

        # Subject detection
        for subject, indicators in self.subject_indicators.items():
            for indicator in indicators:
                if indicator in context.lower():
                    entity.subject = subject
                    break

        return entity

    def map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy NER labels to medical categories"""
        mapping = {
            "DISEASE": ["DISEASE", "MEDICAL_CONDITION", "ILLNESS"],
            "SYMPTOM": ["SYMPTOM"],
            "MEDICATION": ["DRUG", "MEDICATION"],
            "TEST": ["TEST", "LAB_TEST", "PROCEDURE"],
            "BODY_PART": ["ANATOMY", "BODY_PART", "ORGAN"]
        }
        
        for cat, labels in mapping.items():
            if label.upper() in labels:
                return cat
        
        # Map common spaCy labels to medical categories
        spacy_mapping = {
            "PERSON": None,  # Skip person names
            "ORG": None,     # Skip organizations
            "GPE": None,     # Skip locations
            "DATE": "TIME",
            "TIME": "TIME",
            "MONEY": None,
            "PERCENT": None,
            "QUANTITY": "DOSAGE"
        }
        
        return spacy_mapping.get(label, None)
    
    def remove_overlapping_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping the highest confidence ones"""
        
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: (x.start, -x.confidence))
        
        filtered_entities = []
        for entity in entities:
            # Check for overlap with already selected entities
            overlaps = False
            for selected in filtered_entities:
                if (entity.start < selected.end and entity.end > selected.start):
                    overlaps = True
                    # If current entity has higher confidence, replace the selected one
                    if entity.confidence > selected.confidence:
                        filtered_entities.remove(selected)
                        overlaps = False
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def post_process_entities(self, entities: List[MedicalEntity], doc, text: str) -> List[MedicalEntity]:
        """Post-process entities with enhanced contextual analysis"""
        
        processed_entities = []
        
        for entity in entities:
            # Analyze context
            entity = self.analyze_context(entity, doc, text)
            
            # Skip entities that are too generic or common words
            if self.is_too_generic(entity.text):
                continue
            
            # Enhance entity with additional metadata
            entity = self.enhance_entity_metadata(entity, doc)
            
            processed_entities.append(entity)
        
        return processed_entities
    
    def is_too_generic(self, text: str) -> bool:
        """Check if text is too generic to be a meaningful medical entity"""
        
        generic_terms = {
            'patient', 'person', 'people', 'man', 'woman', 'male', 'female',
            'day', 'week', 'month', 'year', 'time', 'today', 'yesterday',
            'hospital', 'clinic', 'doctor', 'nurse', 'physician',
            'good', 'bad', 'better', 'worse', 'normal', 'abnormal',
            'some', 'many', 'few', 'several', 'other', 'another'
        }
        
        return text.lower() in generic_terms or len(text) < 2
    
    def enhance_entity_metadata(self, entity: MedicalEntity, doc) -> MedicalEntity:
        """Enhance entity with additional metadata"""
        
        # Find related entities within the same sentence
        sentence_entities = []
        for sent in doc.sents:
            if entity.start >= sent.start_char and entity.end <= sent.end_char:
                # This entity is in this sentence
                break
        
        # Set semantic type based on label
        semantic_types = {
            'DISEASE': 'disorder',
            'SYMPTOM': 'sign_symptom',
            'MEDICATION': 'pharmacologic_substance',
            'TEST': 'laboratory_procedure',
            'BODY_PART': 'body_part_organ_component',
            'PROCEDURE': 'therapeutic_procedure'
        }
        
        entity.semantic_type = semantic_types.get(entity.label, 'unknown')
        
        return entity
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Main method to extract medical entities from text"""
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Process with spaCy
        doc = self.nlp(processed_text)
        
        # Extract entities using different methods
        all_entities = []
        
        # 1. Phrase matcher extraction
        all_entities.extend(self.extract_with_phrase_matcher(doc))
        
        # 2. Pattern-based extraction
        all_entities.extend(self.extract_with_patterns(doc))
        
        # 3. Contextual extraction
        all_entities.extend(self.extract_contextual_entities(doc))
        
        # Remove overlapping entities
        filtered_entities = self.remove_overlapping_entities(all_entities)
        
        # Post-process entities
        final_entities = self.post_process_entities(filtered_entities, doc, processed_text)
        
        return final_entities
    
    def extract_relationships(self, entities: List[MedicalEntity], doc) -> List[Dict]:
        """Extract relationships between medical entities"""
        
        relationships = []
        
        # Simple dependency-based relationships
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                head = token.head
                
                # Find entities that correspond to these tokens
                subject_entity = None
                object_entity = None
                
                for entity in entities:
                    if entity.start <= token.idx < entity.end:
                        subject_entity = entity
                    if entity.start <= head.idx < entity.end:
                        object_entity = entity
                
                if subject_entity and object_entity and subject_entity != object_entity:
                    relationship = {
                        'subject': subject_entity.text,
                        'predicate': head.lemma_,
                        'object': object_entity.text,
                        'confidence': 0.7
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def generate_summary(self, entities: List[MedicalEntity]) -> Dict:
        """Generate a summary of extracted entities"""
        
        summary = {
            'total_entities': len(entities),
            'by_category': defaultdict(int),
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'negated_entities': 0,
            'unique_entities': set()
        }
        
        for entity in entities:
            summary['by_category'][entity.label] += 1
            summary['unique_entities'].add(entity.normalized_form or entity.text.lower())
            
            if entity.negated:
                summary['negated_entities'] += 1
            
            if entity.confidence >= 0.9:
                summary['by_confidence']['high'] += 1
            elif entity.confidence >= 0.7:
                summary['by_confidence']['medium'] += 1
            else:
                summary['by_confidence']['low'] += 1
        
        summary['unique_entities'] = len(summary['unique_entities'])
        summary['by_category'] = dict(summary['by_category'])
        
        return summary
    
    def to_dataframe(self, entities: List[MedicalEntity]) -> pd.DataFrame:
        """Convert entities to pandas DataFrame"""
        
        data = []
        for entity in entities:
            data.append({
                'text': entity.text,
                'label': entity.label,
                'start': entity.start,
                'end': entity.end,
                'confidence': entity.confidence,
                'negated': entity.negated,
                'severity': entity.severity,
                'certainty': entity.certainty,
                'subject': entity.subject,
                'normalized_form': entity.normalized_form,
                'source_method': entity.source_method,
                'semantic_type': entity.semantic_type
            })
        
        return pd.DataFrame(data)

    # Add a method to process either text or file
    def process(self, input_data: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> ProcessingResult:
        """
        Main entry point: accepts either raw text or a file, returns ProcessingResult.
        """
        start_time = datetime.now()
        metadata = {}
        text = ""
        if filename or (isinstance(input_data, (bytes, io.BytesIO))):
            # Assume file input
            text, metadata = self.file_processor.process_file(input_data, filename=filename)
        else:
            # Assume direct text input
            text = str(input_data)
            metadata = {
                'file_type': 'text',
                'filename': None,
                'processing_time': 0,
                'text_length': len(text),
                'word_count': len(text.split())
            }
        # Entity extraction
        entities = self.extract_entities(text)
        processing_time = (datetime.now() - start_time).total_seconds()
        metadata['processing_time'] = processing_time
        word_count = len(text.split())
        confidence_stats = {
            'high': sum(1 for e in entities if e.confidence >= 0.8),
            'medium': sum(1 for e in entities if 0.5 <= e.confidence < 0.8),
            'low': sum(1 for e in entities if e.confidence < 0.5)
        }
        return ProcessingResult(
            entities=entities,
            text=text,
            metadata=metadata,
            processing_time=processing_time,
            word_count=word_count,
            confidence_stats=confidence_stats
        )

# Example usage and testing
def main():
    """Example usage of the Enhanced Medical Entity Extractor"""
    
    # Initialize the extractor
    extractor = EnhancedMedicalEntityExtractor(use_large_model=False)
    
    # Sample medical text
    sample_text = """
    Patient presents with chest pain and shortness of breath. 
    History of diabetes mellitus and hypertension. 
    Current medications include metformin 500mg twice daily and lisinopril 10mg daily.
    Physical exam reveals no acute distress. 
    ECG shows normal sinus rhythm. 
    CBC and BMP ordered. 
    No evidence of myocardial infarction.
    Patient denies nausea or vomiting.
    """
    
    # Extract entities
    print("Extracting medical entities...")
    entities = extractor.extract_entities(sample_text)
    
    # Display results
    print(f"\nFound {len(entities)} medical entities:")
    print("-" * 80)
    
    for entity in entities:
        print(f"Text: {entity.text}")
        print(f"Label: {entity.label}")
        print(f"Confidence: {entity.confidence:.2f}")
        print(f"Negated: {entity.negated}")
        print(f"Normalized: {entity.normalized_form}")
        print(f"Source: {entity.source_method}")
        print("-" * 40)
    
    # Generate summary
    summary = extractor.generate_summary(entities)
    print("\nSummary:")
    print(f"Total entities: {summary['total_entities']}")
    print(f"By category: {summary['by_category']}")
    print(f"Negated entities: {summary['negated_entities']}")
    print(f"Unique entities: {summary['unique_entities']}")
    
    # Convert to DataFrame
    df = extractor.to_dataframe(entities)
    print("\nDataFrame shape:", df.shape)
    print(df.head())

    # For file (example: PDF)
    # result_file = extractor.process("path/to/file.pdf", filename="file.pdf")
    # print("Entities from file:", result_file.entities)


if __name__ == "__main__":
    main()