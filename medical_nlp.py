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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            logger.error(f"Error processing PDF file {file_path}: {e}")
        
        return text
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {e}")
        
        return text.strip()
    
    def _process_image(self, file_path: str) -> str:
        """Extract text from image file using OCR"""
        text = ""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
        
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
            logger.error(f"Error processing text input {file_path}: {e}")
        
        return text.strip()

class EnhancedMedicalEntityExtractor:
    """
    Advanced medical entity extraction system with file processing capabilities
    """
    
    def __init__(self, use_large_model: bool = False):
        # Load spaCy model (use large model for better accuracy)
        model_name = "en_core_web_lg" if use_large_model else "en_core_web_sm"
        self.nlp = None
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e1:
            # Try import the model package and load directly
            try:
                import importlib
                pkg = importlib.import_module(model_name.replace('-', '_'))
                self.nlp = pkg.load()
            except Exception as e1b:
                try:
                    import spacy.cli
                    spacy.cli.download(model_name)
                    self.nlp = spacy.load(model_name)
                except Exception as e2:
                    # Final fallback to small model
                    try:
                        fallback = "en_core_web_sm"
                        if fallback != model_name:
                            import spacy.cli
                            spacy.cli.download(fallback)
                        # Try package import first, then spacy.load
                        try:
                            import en_core_web_sm as en_sm
                            self.nlp = en_sm.load()
                        except Exception:
                            self.nlp = spacy.load(fallback)
                    except Exception as e3:
                        # Last-resort fallback to a blank English pipeline
                        logger.warning(f"Failed to load spaCy models ({e1}/{e1b}); download attempts failed ({e2}/{e3}). Using blank 'en' pipeline.")
                        self.nlp = spacy.blank("en")
        
        # Load medical data from JSON
        self.load_medical_data()

        # Add custom pipeline components
        self._add_custom_components()
        
        # Initialize comprehensive medical knowledge base
        # self._initialize_enhanced_dictionaries() # Removed, data loaded from JSON
        
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
    
    def load_medical_data(self):
        """Load medical dictionaries from JSON file"""
        try:
            # Assuming medical_data.json is in the same directory as this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'medical_data.json')
            
            with open(json_path, 'r') as f:
                self.medical_data = json.load(f)
                
            # Map JSON keys to class attributes for backward compatibility if needed
            # or just use self.medical_data directly        # Flatten dictionaries for easier matching
            self.disease_terms = {term.lower() for terms in self.medical_data.get('diseases', {}).values() for term in terms}
            self.symptom_terms = {term.lower() for terms in self.medical_data.get('symptoms', {}).values() for term in terms}
            self.medication_terms = {term.lower() for terms in self.medical_data.get('medications', {}).values() for term in terms}
            self.body_part_terms = {term.lower() for terms in self.medical_data.get('body_parts', {}).values() for term in terms}
            self.procedure_terms = {term.lower() for terms in self.medical_data.get('procedures', {}).values() for term in terms}
            
            # Load Lab Tests
            self.lab_test_terms = {term.lower() for terms in self.medical_data.get('lab_tests', {}).values() for term in terms}
            # Add common standalone lab terms
            self.lab_test_terms.update([
                "wbc", "rbc", "hgb", "hct", "plt", "mcv", "mch", "mchc", "rdw", "mpv",
                "glucose", "bun", "creatinine", "sodium", "potassium", "chloride", "calcium",
                "albumin", "protein", "bilirubin", "alp", "alt", "ast", "troponin", "ck-mb",
                "bnp", "cholesterol", "triglycerides", "hdl", "ldl", "tsh", "t3", "t4",
                "psa", "hba1c", "a1c", "inr", "pt", "ptt", "bp", "hr", "rr", "temp", "spo2",
                "bmi", "weight", "height", "pulse", "respirations", "temperature", "saturation"
            ])
            self.diseases = self.medical_data.get('diseases', {})
            self.symptoms = self.medical_data.get('symptoms', {})
            self.medications = self.medical_data.get('medications', {})
            self.tests = self.medical_data.get('tests', {})
            self.body_parts = self.medical_data.get('body_parts', {})
            self.procedures = self.medical_data.get('procedures', {})
            self.clinical_findings = self.medical_data.get('clinical_findings', {})
            self.microorganisms = self.medical_data.get('microorganisms', {})
            self.severity_indicators = self.medical_data.get('severity_indicators', {})
            self.certainty_indicators = self.medical_data.get('certainty_indicators', {})
            self.temporal_indicators = self.medical_data.get('temporal_indicators', {})
            self.laterality = self.medical_data.get('laterality', {})
            self.anatomical_locations = self.medical_data.get('anatomical_locations', {})
            self.treatments = self.medical_data.get('treatments', {})
            self.vital_signs = self.medical_data.get('vital_signs', {})
            self.allergy_terms = self.medical_data.get('allergy_terms', {})
            self.social_history = self.medical_data.get('social_history', {})
            self.family_history = self.medical_data.get('family_history', {})
            self.lab_values = self.medical_data.get('lab_values', {})
            self.imaging_findings = self.medical_data.get('imaging_findings', {})
            self.specialties = self.medical_data.get('specialties', {})
            self.units = self.medical_data.get('units', {})
            self.risk_factors = self.medical_data.get('risk_factors', {})
            self.status_descriptors = self.medical_data.get('status_descriptors', {})
            self.negations = self.medical_data.get('negations', {})
            self.abbreviations = self.medical_data.get('abbreviations', {}) # Ensure abbreviations are loaded
            
            # New categories
            self.vaccines = self.medical_data.get('vaccines', {})
            self.medical_devices = self.medical_data.get('medical_devices', {})
            self.lifestyle_factors = self.medical_data.get('lifestyle_factors', {})
            self.dietary_supplements = self.medical_data.get('dietary_supplements', {})
            self.genetic_info = self.medical_data.get('genetic_info', {})
            self.social_determinants = self.medical_data.get('social_determinants', {})
            self.substances = self.medical_data.get('substances', {})
            
        except FileNotFoundError:
            logger.warning(f"medical_data.json not found at {json_path}. Initializing with empty medical data.")
            self.medical_data = defaultdict(dict)
            # Initialize all attributes to empty dicts to prevent AttributeError
            self.diseases = {}
            self.symptoms = {}
            self.medications = {}
            self.tests = {}
            self.body_parts = {}
            self.procedures = {}
            self.clinical_findings = {}
            self.microorganisms = {}
            self.severity_indicators = {}
            self.certainty_indicators = {}
            self.temporal_indicators = {}
            self.laterality = {}
            self.anatomical_locations = {}
            self.treatments = {}
            self.vital_signs = {}
            self.allergy_terms = {}
            self.social_history = {}
            self.family_history = {}
            self.lab_values = {}
            self.imaging_findings = {}
            self.specialties = {}
            self.units = {}
            self.risk_factors = {}
            self.status_descriptors = {}
            self.negations = {}
            self.abbreviations = {}
            self.vaccines = {}
            self.medical_devices = {}
            self.lifestyle_factors = {}
            self.dietary_supplements = {}
            self.dietary_supplements = {}
            self.genetic_info = {}
            self.social_determinants = {}
            self.substances = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding medical_data.json: {e}. Initializing with empty medical data.")
            self.medical_data = defaultdict(dict)
            self.diseases = {} # etc.
        except Exception as e:
            logger.error(f"Error loading medical data: {e}. Initializing with empty medical data.")
            self.medical_data = defaultdict(dict)
            self.diseases = {} # etc.

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
        """
        This method is now deprecated as data is loaded from medical_data.json.
        It's kept as a placeholder or can be removed if not needed for other purposes.
        """
        pass # Data is loaded via load_medical_data()
        
    def _initialize_pattern_matchers(self):
        """Initialize advanced pattern matchers with comprehensive medical patterns"""
        
        # Phrase matchers for multi-word entities
        self.phrase_matchers = {}
        
        # Create matchers for each category including new ones
        categories = {
            'DISEASE': self.diseases,
            'SYMPTOM': self.symptoms,
            'MEDICATION': self.medications,
            'TEST': self.tests,
            'TEST': self.tests,
            'BODY_PART': {**self.body_parts, **self.anatomical_locations},
            'PROCEDURE': self.procedures,
            'CLINICAL_FINDING': self.clinical_findings,
            'MICROORGANISM': self.microorganisms,
            'VACCINE': self.vaccines,
            'MEDICAL_DEVICE': self.medical_devices,
            'LIFESTYLE': self.lifestyle_factors,
            'SUPPLEMENT': self.dietary_supplements,
            'GENETIC': self.genetic_info,
            'SDOH': self.social_determinants,
            'SUBSTANCE': self.substances,
            'ALLERGY': self.allergy_terms
        }
        
        for category, terms_dict in categories.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = []
            
            for main_term, synonyms in terms_dict.items():
                all_terms = [main_term] + synonyms
                for term in all_terms:
                    # Handle multi-word terms properly
                    patterns.append(self.nlp.make_doc(term.lower()))
            
            if patterns:
                matcher.add(category, patterns)
            self.phrase_matchers[category] = matcher
        
        # Advanced rule-based matcher
        self.rule_matcher = Matcher(self.nlp.vocab)
        self._add_advanced_patterns()
    
    def _add_advanced_patterns(self):
        """Add sophisticated linguistic patterns for medical entities"""
        
        # Dosage patterns - comprehensive medication dosing
        dosage_patterns = [
            # Standard dosage: "50 mg", "2.5 g", "10 units"
            [{"TEXT": {"REGEX": r"\d+\.?\d*"}}, 
             {"LOWER": {"IN": ["mg", "g", "ml", "cc", "units", "iu", "mcg", "µg", "grams", "milligrams", "micrograms"]}}],
            
            # Frequency patterns: "once daily", "twice a day"
            [{"LOWER": {"IN": ["once", "twice"]}}, 
             {"LOWER": {"IN": ["daily", "times", "time"]}, "OP": "?"}, 
             {"LOWER": {"IN": ["daily", "day", "per", "a"]}, "OP": "?"}],

            # Frequency with numbers: "3 times daily", "three times a day" - REQUIRED 'times' or similar
            [{"LOWER": {"IN": ["three", "four", "five", "six", "seven", "eight", "nine", "ten", "1", "2", "3", "4", "5", "6"]}}, 
             {"LOWER": {"IN": ["times", "time", "x", "daily"]}}, 
             {"LOWER": {"IN": ["daily", "day", "per", "a"]}, "OP": "?"}],
            
            # PRN patterns: "as needed", "prn"
            [{"LOWER": "prn"}],
            [{"LOWER": "as"}, {"LOWER": {"IN": ["needed", "required", "necessary"]}}],
            
            # Route patterns: "orally", "by mouth", "IV", "subcutaneous"
            [{"LOWER": {"IN": ["orally", "oral", "po", "iv", "im", "sq", "subq", "subcutaneous", 
                               "intramuscular", "intravenous", "topical", "sublingual"]}}],
            
            # Complex dosing: "1-2 tablets"
            [{"TEXT": {"REGEX": r"\d+-\d+"}}, 
             {"LOWER": {"IN": ["tablet", "tablets", "capsule", "capsules", "pill", "pills"]}}],
            
            # Taper patterns: "taper dose", "gradually decrease"
            [{"LOWER": {"IN": ["taper", "gradually"]}}, 
             {"LOWER": {"IN": ["dose", "decrease", "reduce", "increase"]}, "OP": "?"}]
        ]
        
        for i, pattern in enumerate(dosage_patterns):
            self.rule_matcher.add(f"DOSAGE_{i}", [pattern])
        
        # Temporal patterns - when conditions occurred
        temporal_patterns = [
            # Duration (Digits): "for 3 days"
            [{"LOWER": {"IN": ["for", "x"]}}, 
             {"TEXT": {"REGEX": r"\d+"}}, 
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years", "hr", "hrs", "hours"]}}],

            # Duration (Words): "for three days"
            [{"LOWER": {"IN": ["for", "x"]}}, 
             {"LOWER": {"IN": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]}}, 
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years", "hr", "hrs", "hours"]}}],
            
            # Ago (Digits): "3 days ago"
            [{"TEXT": {"REGEX": r"\d+"}}, 
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years"]}}, 
             {"LOWER": {"IN": ["ago", "prior", "before", "earlier"]}}],

            # Ago (Words): "three days ago"
            [{"LOWER": {"IN": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]}}, 
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years"]}}, 
             {"LOWER": {"IN": ["ago", "prior", "before", "earlier"]}}],
            
            # Since patterns: "since 2020", "since January"
            [{"LOWER": "since"}, 
             {"TEXT": {"REGEX": r"\d{4}|\w+"}}],
            
            # Date patterns: "on 01/15/2024", "in March 2024"
            [{"LOWER": {"IN": ["on", "in"]}, "OP": "?"}, 
             {"TEXT": {"REGEX": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"}}],
            
            # Onset patterns: "sudden onset", "gradual onset"
            [{"LOWER": {"IN": ["sudden", "acute", "gradual", "slow", "rapid", "abrupt"]}}, 
             {"LOWER": "onset"}],
            
            # Chronic/acute temporal: "chronic for 5 years"
            [{"LOWER": {"IN": ["chronic", "persistent", "recurrent", "ongoing"]}}, 
             {"LOWER": "for", "OP": "?"}, 
             {"TEXT": {"REGEX": r"\d+"}, "OP": "?"}, 
             {"LOWER": {"IN": ["years", "months", "weeks"]}, "OP": "?"}]
        ]
        
        for i, pattern in enumerate(temporal_patterns):
            self.rule_matcher.add(f"TIME_{i}", [pattern])
        
        # Medical procedure patterns
        procedure_patterns = [
            # Surgery suffixes
            [{"LOWER": {"REGEX": r"\w+ectomy"}}],  # appendectomy, mastectomy
            [{"LOWER": {"REGEX": r"\w+ostomy"}}],  # colostomy, tracheostomy
            [{"LOWER": {"REGEX": r"\w+otomy"}}],   # laparotomy, thoracotomy
            [{"LOWER": {"REGEX": r"\w+scopy"}}],   # colonoscopy, endoscopy
            [{"LOWER": {"REGEX": r"\w+plasty"}}],  # angioplasty, arthroplasty
            [{"LOWER": {"REGEX": r"\w+rrhaphy"}}], # herniorrhaphy
            [{"LOWER": {"REGEX": r"\w+pexy"}}],    # gastropexy
            [{"LOWER": {"REGEX": r"\w+lysis"}}],   # dialysis, hemolysis
            
            # Status post patterns: "s/p appendectomy", "status post CABG"
            [{"LOWER": {"IN": ["s/p", "sp", "status"]}}, 
             {"LOWER": "post", "OP": "?"}, 
             {"IS_ALPHA": True}],
            
            # Procedure with timing: "underwent surgery yesterday"
            [{"LOWER": {"IN": ["underwent", "received", "had"]}}, 
             {"IS_ALPHA": True}, 
             {"LOWER": {"IN": ["yesterday", "today", "recently", "last"]}, "OP": "?"}]
        ]
        
        for i, pattern in enumerate(procedure_patterns):
            self.rule_matcher.add(f"PROCEDURE_{i}", [pattern])
        
        # Severity and qualifier patterns
        severity_patterns = [
            # Severity descriptors: "severe pain", "mild discomfort"
            [{"LOWER": {"IN": ["mild", "moderate", "severe", "acute", "chronic", "critical", "significant", "marked"]}}, 
             {"IS_ALPHA": True}],
            
            # Grade patterns: "grade 3", "stage IV"
            [{"LOWER": {"IN": ["grade", "stage", "class", "level"]}}, 
             {"TEXT": {"REGEX": r"\d+|[IVX]+"}}],
            
            # Progression: "worsening", "improving", "stable"
            [{"LOWER": {"IN": ["worsening", "improving", "progressive", "stable", "deteriorating", "resolving"]}}]
        ]
        
        for i, pattern in enumerate(severity_patterns):
            self.rule_matcher.add(f"SEVERITY_{i}", [pattern])
        
        # Lab value patterns - WHITELIST BASED
        # Convert set to list for spaCy pattern
        lab_term_list = list(self.lab_test_terms)
        
        lab_patterns = [
            # 1. Name + Number (e.g. "WBC 5.5")
            [{"LOWER": {"IN": lab_term_list}}, 
             {"LIKE_NUM": True}],
             
            # 2. Name + Number + Unit (e.g. "Glucose 100 mg/dL")
            [{"LOWER": {"IN": lab_term_list}}, 
             {"LIKE_NUM": True}, 
             {"LOWER": {"REGEX": r"^[a-zA-Z0-9/%]+$"}}],
             
            # 3. Name + Number + Split Unit (e.g. "Glucose 100 mg / dL")
            [{"LOWER": {"IN": lab_term_list}}, 
             {"LIKE_NUM": True}, 
             {"IS_ALPHA": True},
             {"ORTH": "/"},
             {"IS_ALPHA": True}],
            
            # Range patterns: "between 5-10", "within normal limits"
            [{"LOWER": "between"}, 
             {"LIKE_NUM": True}, 
             {"ORTH": "-"}, 
             {"LIKE_NUM": True}],
            
            # Normal/abnormal: "elevated glucose", "decreased sodium"
            [{"LOWER": {"IN": ["elevated", "increased", "high", "decreased", "low", "reduced", "normal"]}}, 
             {"LOWER": {"IN": lab_term_list}}]
        ]
        
        for i, pattern in enumerate(lab_patterns):
            self.rule_matcher.add(f"LAB_VALUE_{i}", [pattern])
        
        # Anatomical location patterns
        location_patterns = [
            # Laterality with specific body parts: "left arm", "right leg"
            # We restrict the second token to be a likely body part to avoid over-matching
            [{"LOWER": {"IN": ["left", "right", "bilateral"]}}, 
             {"LOWER": {"IN": ["upper", "lower", "mid"]}, "OP": "?"}, 
             {"LOWER": {"IN": ["arm", "leg", "hand", "foot", "shoulder", "knee", "hip", "elbow", "wrist", "ankle", 
                               "eye", "ear", "lung", "kidney", "breast", "lobe", "ventricle", "atrium", "side", "flank",
                               "quadrant", "extremity", "lobe"]}}],
            
            # Specific locations: "lower back", "upper abdomen"
            [{"LOWER": {"IN": ["upper", "lower", "mid", "central", "distal", "proximal"]}}, 
             {"LOWER": {"IN": ["back", "abdomen", "chest", "spine", "neck", "thoracic", "lumbar", "sacral"]}}],
            
            # Quadrants: "RUQ", "left lower quadrant"
            [{"TEXT": {"REGEX": r"[RL][UL]Q"}}, {"LOWER": "quadrant", "OP": "?"}]
        ]
        
        for i, pattern in enumerate(location_patterns):
            self.rule_matcher.add(f"BODY_PART_{i}", [pattern]) # Changed label from LOCATION to BODY_PART
        
        # Vital sign patterns
        vital_patterns = [
            # Blood pressure: "BP 120/80", "blood pressure 140/90"
            [{"LOWER": {"IN": ["bp", "blood"]}}, 
             {"LOWER": "pressure", "OP": "?"}, 
             {"TEXT": {"REGEX": r"\d{2,3}/\d{2,3}"}}],
            
            # Heart rate: "HR 72", "pulse 80 bpm"
            [{"LOWER": {"IN": ["hr", "heart", "pulse"]}}, 
             {"LOWER": {"IN": ["rate", ""]}, "OP": "?"}, 
             {"TEXT": {"REGEX": r"\d{2,3}"}}, 
             {"LOWER": "bpm", "OP": "?"}],
            
            # Temperature: "temp 98.6", "temperature 37.5 C"
            [{"LOWER": {"IN": ["temp", "temperature"]}}, 
             {"TEXT": {"REGEX": r"\d{2,3}\.?\d*"}}, 
             {"LOWER": {"IN": ["f", "c", "fahrenheit", "celsius"]}, "OP": "?"}],
            
            # Oxygen saturation: "O2 sat 95%", "SpO2 98%"
            [{"TEXT": {"REGEX": r"o2|spo2"}}, 
             {"LOWER": {"IN": ["sat", "saturation"]}, "OP": "?"}, 
             {"TEXT": {"REGEX": r"\d{2,3}"}}, 
             {"ORTH": "%", "OP": "?"}]
        ]
        
        for i, pattern in enumerate(vital_patterns):
            self.rule_matcher.add(f"VITAL_SIGN_{i}", [pattern])
        
        # Allergy patterns
        allergy_patterns = [
            # "allergic to penicillin", "allergy to shellfish"
            [{"LOWER": {"IN": ["allergic", "allergy", "allergies"]}}, 
             {"LOWER": "to"}, 
             {"IS_ALPHA": True}],
            
            # "NKDA", "no known drug allergies"
            [{"LOWER": {"IN": ["nkda", "nka", "no"]}}, 
             {"LOWER": "known", "OP": "?"}, 
             {"LOWER": {"IN": ["drug", "allergies", "allergy"]}, "OP": "?"}]
        ]
        
        for i, pattern in enumerate(allergy_patterns):
            self.rule_matcher.add(f"ALLERGY_{i}", [pattern])
    
    def _initialize_contextual_analyzers(self):
        """Initialize comprehensive contextual analysis components"""
        
        # Negation triggers with scope windows
        self.negation_triggers = {
            'explicit_negation': [
                'no', 'not', 'without', 'absence', 'absent', 'negative', 
                'deny', 'denies', 'denied', 'never', 'neither', 'nor',
                'cannot', 'can\'t', 'won\'t', 'didn\'t', 'doesn\'t', 'don\'t'
            ],
            'ruled_out': [
                'ruled out', 'rule out', 'r/o', 'exclude', 'excluded',
                'unlikely', 'not consistent with'
            ],
            'normal_findings': [
                'unremarkable', 'within normal limits', 'wnl', 'normal',
                'clear', 'free of', 'no evidence of', 'no sign of',
                'negative for', 'clean', 'benign'
            ]
        }
        
        # Negation scope (how many tokens after negation trigger to check)
        self.negation_scope = 6
        
        # Pseudo-negations (words that look like negations but aren't)
        self.pseudo_negations = [
            'no increase', 'no decrease', 'no change', 'no longer',
            'not only', 'no significant', 'not significant',
            'no further', 'no new'
        ]
        
        # Uncertainty indicators with confidence levels
        self.uncertainty_indicators = {
            'high_uncertainty': [
                'possible', 'possibly', 'potential', 'potentially',
                'may', 'might', 'could', 'perhaps', 'maybe'
            ],
            'moderate_uncertainty': [
                'probable', 'probably', 'likely', 'suspected',
                'suspect', 'questionable', 'uncertain',
                'appears', 'seems', 'suggests', 'suggestive'
            ],
            'low_uncertainty': [
                'presumed', 'presumptive', 'impression', 'consistent with',
                'compatible with', 'favor', 'favors'
            ]
        }
        
        # Assertion indicators (definite presence)
        self.assertion_indicators = [
            'confirmed', 'diagnosed', 'documented', 'established',
            'proven', 'positive for', 'present', 'found',
            'identified', 'shows', 'demonstrates', 'reveals'
        ]
        
        # Subject indicators (experiencer of condition)
        self.subject_indicators = {
            'patient': [
                'patient', 'pt', 'he', 'she', 'they', 'him', 'her',
                'his', 'hers', 'their', 'the patient', 'this patient'
            ],
            'family': [
                'family', 'mother', 'father', 'mom', 'dad', 'parent',
                'sibling', 'brother', 'sister', 'relative', 'grandmother',
                'grandfather', 'aunt', 'uncle', 'cousin', 'son', 'daughter',
                'maternal', 'paternal', 'family history'
            ],
            'other': [
                'doctor', 'physician', 'nurse', 'provider', 'staff'
            ]
        }
        
        # Historical indicators (past vs current)
        self.historical_indicators = {
            'past': [
                'history of', 'h/o', 'past', 'previous', 'prior',
                'former', 'old', 'previous episode', 'in the past',
                'previously', 'historically'
            ],
            'current': [
                'current', 'currently', 'present', 'now', 'today',
                'active', 'ongoing', 'this admission', 'new onset',
                'recent', 'recently'
            ]
        }
        
        # Conditional indicators (hypothetical)
        self.conditional_indicators = [
            'if', 'should', 'would', 'in case of', 'consider',
            'to rule out', 'differential', 'versus', 'vs'
        ]
        
        # Continuation indicators (condition persists)
        self.continuation_indicators = [
            'continue', 'continues', 'continued', 'ongoing',
            'persistent', 'persists', 'still', 'remains',
            'chronic', 'longstanding'
        ]
    
    def _initialize_normalization_maps(self):
        """Initialize comprehensive normalization and standardization maps"""
        
        # Abbreviations are now loaded from medical_data.json
        # self.abbreviations = { ... } # Removed hardcoded dict
        
        # Create reverse mapping for normalization
        self.normalization_map = {}
        categories_to_normalize = [
            ('DISEASE', self.diseases),
            ('SYMPTOM', self.symptoms),
            ('MEDICATION', self.medications),
            ('TEST', self.tests),
            ('BODY_PART', {**self.body_parts, **self.anatomical_locations}),
            ('PROCEDURE', self.procedures),
            ('CLINICAL_FINDING', self.clinical_findings),
            ('MICROORGANISM', self.microorganisms),
            ('VACCINE', self.vaccines),
            ('MEDICAL_DEVICE', self.medical_devices),
            ('LIFESTYLE', self.lifestyle_factors),
            ('SUPPLEMENT', self.dietary_supplements),
            ('GENETIC', self.genetic_info),
            ('SDOH', self.social_determinants),
            ('SUBSTANCE', self.substances),
            ('ALLERGY', self.allergy_terms)
        ]
        
        for category, terms_dict in categories_to_normalize:
            for canonical, variants in terms_dict.items():
                # Map canonical term to itself
                self.normalization_map[canonical.lower()] = {
                    'canonical': canonical,
                    'category': category
                }
                # Map all variants to canonical
                for variant in variants:
                    self.normalization_map[variant.lower()] = {
                        'canonical': canonical,
                        'category': category
                    }
        
        # Add abbreviations to normalization map
        for abbrev, full_form in self.abbreviations.items():
            if abbrev.lower() not in self.normalization_map:
                self.normalization_map[abbrev.lower()] = {
                    'canonical': full_form,
                    'category': 'ABBREVIATION'
                }
    
    def _initialize_semantic_analyzer(self):
        """Initialize TF-IDF vectorizer for semantic similarity"""
        # Combine all medical terms for TF-IDF
        all_medical_terms = []
        for category_dict in [self.diseases, self.symptoms, self.medications, self.tests, self.body_parts, self.procedures, self.clinical_findings, self.microorganisms, self.vaccines, self.medical_devices, self.lifestyle_factors, self.dietary_supplements, self.genetic_info, self.social_determinants, self.substances, self.allergy_terms]:
            for main_term, synonyms in category_dict.items():
                all_medical_terms.append(main_term)
                all_medical_terms.extend(synonyms)
        
        # Add abbreviations
        all_medical_terms.extend(list(self.abbreviations.keys()))
        all_medical_terms.extend(list(self.abbreviations.values()))

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        if all_medical_terms:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_medical_terms)
            self.tfidf_terms = all_medical_terms
        else:
            self.tfidf_matrix = None
            self.tfidf_terms = []

    def _initialize_concept_graph(self):
        """Initialize a graph for medical concept relationships"""
        self.concept_graph = nx.Graph()
        
        # Add nodes for all canonical terms
        for category_dict in [self.diseases, self.symptoms, self.medications, self.tests, self.body_parts, self.procedures, self.clinical_findings, self.microorganisms]:
            for main_term in category_dict.keys():
                self.concept_graph.add_node(main_term.lower(), type=self._identify_entity_type(main_term))
        
        # Add edges for known relationships (e.g., disease-symptom, disease-medication)
        # This part would typically be populated from a more structured knowledge base
        # For now, we can add some basic inferred relationships
        
        # Example: Disease-Symptom relationships
        for disease, synonyms in self.diseases.items():
            for symptom, sym_synonyms in self.symptoms.items():
                # Simple heuristic: if symptom is often mentioned with disease
                if symptom in ' '.join(synonyms): # Very basic, needs improvement
                    self.concept_graph.add_edge(disease.lower(), symptom.lower(), relation='has_symptom')
        
        # Example: Disease-Medication relationships
        for disease, dis_synonyms in self.diseases.items():
            for medication, med_synonyms in self.medications.items():
                # Simple heuristic: if medication is often used for disease
                if disease in ' '.join(med_synonyms): # Very basic, needs improvement
                    self.concept_graph.add_edge(disease.lower(), medication.lower(), relation='treated_by')



    def _identify_entity_type(self, text):
        """Identify the type of medical entity"""
        text_lower = text.lower()
        
        # Check against loaded data
        if hasattr(self, 'medical_data'):
            # Check diseases
            for disease, variations in self.medical_data.get('diseases', {}).items():
                if text_lower == disease.lower() or text_lower in [v.lower() for v in variations]:
                    return "DISEASE"
            
            # Check symptoms
            for symptom, variations in self.medical_data.get('symptoms', {}).items():
                if text_lower == symptom.lower() or text_lower in [v.lower() for v in variations]:
                    return "SYMPTOM"
            
            # Check medications
            for med, variations in self.medical_data.get('medications', {}).items():
                if text_lower == med.lower() or text_lower in [v.lower() for v in variations]:
                    return "MEDICATION"
            
            # Check tests
            for test, variations in self.medical_data.get('tests', {}).items():
                if text_lower == test.lower() or text_lower in [v.lower() for v in variations]:
                    return "TEST"
            
            # Check body parts
            for part, variations in self.medical_data.get('body_parts', {}).items():
                if text_lower == part.lower() or text_lower in [v.lower() for v in variations]:
                    return "BODY_PART"
            
            # Check procedures
            for proc, variations in self.medical_data.get('procedures', {}).items():
                if text_lower == proc.lower() or text_lower in [v.lower() for v in variations]:
                    return "PROCEDURE"
        
        return "UNKNOWN"
       
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common medical formatting
        text = re.sub(r'\b(\d+)\s*-\s*(\d+)\b', r'\1-\2', text)  # ranges
        text = re.sub(r'\b(\d+\.?\d*)\s*(mg|g|ml|cc|dL|L|kg|lbs)\b', r'\1 \2', text, flags=re.IGNORECASE)  # ensure space for dosages
        
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
                
                # Get canonical form (ensure it's a plain string)
                nm_val = self.normalization_map.get(span.text.lower())
                if isinstance(nm_val, dict):
                    canonical = nm_val.get('canonical', span.text.lower())
                elif isinstance(nm_val, str):
                    canonical = nm_val
                else:
                    canonical = span.text.lower()
                
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
        
        # Apply all rule patterns
        matches = self.rule_matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            match_id_string = self.nlp.vocab.strings[match_id]
            
            # Determine label and confidence
            label = "UNKNOWN"
            confidence = 0.85
            
            if "DOSAGE" in match_id_string:
                label = "DOSAGE"
                confidence = 0.88
            elif "TIME" in match_id_string:
                label = "TIME"
                confidence = 0.75
            elif "LAB_VALUE" in match_id_string:
                label = "LAB_VALUE"
                confidence = 0.96  # Higher confidence
            elif "LAB" in match_id_string: # Fallback for generic LAB
                label = "LAB_VALUE"
                confidence = 0.90
            elif "SEVERITY" in match_id_string:
                label = "SEVERITY"
                confidence = 0.88
            elif "BODY_PART" in match_id_string:
                label = "BODY_PART"
                confidence = 0.90
            elif "VITAL_SIGN" in match_id_string:
                label = "VITAL_SIGN"
                confidence = 0.96
            elif "PROCEDURE" in match_id_string:
                label = "PROCEDURE"
                confidence = 0.90
            
            # Create MedicalEntity object
            entity = MedicalEntity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=confidence,
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
        # Classification based on context
        if any(word in context_text for word in ['diagnose', 'diagnosed', 'condition', 'disease', 'disorder', 'syndrome']):
            return 'DISEASE'
        elif any(word in context_text for word in ['complains', 'reports', 'feels', 'experiencing', 'suffering', 'pain']):
            return 'SYMPTOM'
        elif any(word in context_text for word in ['prescribed', 'taking', 'medication', 'drug', 'dose', 'mg', 'tablet']):
            return 'MEDICATION'
        elif any(word in context_text for word in ['test', 'scan', 'examination', 'lab', 'level', 'result']):
            return 'TEST'
        elif any(word in context_text for word in ['procedure', 'surgery', 'operation', 'resection', 'repair']):
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
            logger.warning(f"Semantic similarity computation failed: {e}")
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
    
    def _initialize_semantic_analyzer(self) -> None:
        """Initialize TF-IDF vectorizer and corpus for semantic similarity."""
        try:
            corpus_terms = []
            # Collect terms from available dictionaries
            dict_attrs = [
                'diseases', 'symptoms', 'medications', 'tests', 'body_parts',
                'procedures', 'clinical_findings', 'microorganisms'
            ]
            for attr in dict_attrs:
                if hasattr(self, attr):
                    terms_dict = getattr(self, attr) or {}
                    for main_term, synonyms in terms_dict.items():
                        corpus_terms.append(main_term)
                        corpus_terms.extend(synonyms)
            # Deduplicate while preserving order
            seen = set()
            deduped_terms = []
            for t in corpus_terms:
                tl = t.strip().lower()
                if tl and tl not in seen:
                    seen.add(tl)
                    deduped_terms.append(tl)

            if not deduped_terms:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                self.term_index = {}
                return

            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english', ngram_range=(1, 2), min_df=1
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(deduped_terms)
            # Keep index aligned with matrix row order
            self.term_index = {term: i for i, term in enumerate(deduped_terms)}
        except Exception as e:
            logger.warning(f"Failed to initialize semantic analyzer: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.term_index = {}

    def _initialize_concept_graph(self) -> None:
        """Initialize a lightweight medical concept graph from dictionaries."""
        try:
            self.concept_graph = nx.Graph()
            dict_attrs = [
                ('DISEASE', 'diseases'), ('SYMPTOM', 'symptoms'), ('MEDICATION', 'medications'),
                ('TEST', 'tests'), ('BODY_PART', 'body_parts'), ('PROCEDURE', 'procedures'),
                ('CLINICAL_FINDING', 'clinical_findings'), ('MICROORGANISM', 'microorganisms')
            ]
            for label, attr in dict_attrs:
                if hasattr(self, attr):
                    terms_dict = getattr(self, attr) or {}
                    for main_term, synonyms in terms_dict.items():
                        main = main_term.lower()
                        self.concept_graph.add_node(main, label=label)
                        for syn in synonyms:
                            s = syn.lower()
                            self.concept_graph.add_node(s, label=label)
                            # Connect synonym to main term
                            self.concept_graph.add_edge(main, s, relation='synonym')
        except Exception as e:
            logger.warning(f"Failed to initialize concept graph: {e}")
            self.concept_graph = nx.Graph()
    
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

        # Check for negation (following)
        if not entity.negated:
            # Look at text immediately following the entity
            post_entity_start = entity.end
            post_entity_end = min(len(text), entity.end + 50)
            post_text = text[post_entity_start:post_entity_end].lower()
            
            for neg_term in ['ruled out', 'negative', 'absent', 'unlikely']:
                if neg_term in post_text:
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
            "QUANTITY": "DOSAGE",
            "CARDINAL": None # Skip cardinal numbers unless caught by other patterns
        }
        
        return spacy_mapping.get(label, None)
    
    def remove_overlapping_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping the highest confidence ones"""
        
        if not entities:
            return entities
        
        # Sort by start position, then confidence (desc), then length (desc)
        entities.sort(key=lambda x: (x.start, -x.confidence, -(x.end - x.start)))
        
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
            'some', 'many', 'few', 'several', 'other', 'another',
            'condition', 'problem', 'result', 'results', 'level', 'levels',
            'care', 'health', 'history', 'complaint', 'complaints',
            'cardiology', 'oncology', 'neurology', 'dermatology', 'radiology', 'pathology', 'surgery',
            'pediatrics', 'psychiatry', 'urology', 'nephrology', 'gastroenterology', 'hematology',
            'endocrinology', 'rheumatology', 'pulmonology', 'immunology', 'anesthesiology'
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
        
        # Standardize labels
        if entity.label == 'LAB':
            entity.label = 'LAB_VALUE'
        elif entity.label == 'BODY':
            entity.label = 'BODY_PART'
        elif entity.label == 'VITAL':
            entity.label = 'VITAL_SIGN'
            
        # If a TEST has a value (number) in it, upgrade it to LAB_VALUE
        if entity.label == 'TEST' and any(char.isdigit() for char in entity.text):
            entity.label = 'LAB_VALUE'
            
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

