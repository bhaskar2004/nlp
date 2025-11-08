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
            'diabetes': ['diabetes mellitus', 'dm', 'diabetic', 'hyperglycemia', 'type 1 diabetes', 'type 2 diabetes', 't1dm', 't2dm', 'insulin dependent', 'non-insulin dependent', 'gestational diabetes'],
            'hypertension': ['high blood pressure', 'htn', 'hypertensive', 'elevated bp', 'essential hypertension', 'secondary hypertension', 'malignant hypertension'],
            'myocardial infarction': ['heart attack', 'mi', 'stemi', 'nstemi', 'acute mi', 'cardiac infarction', 'coronary occlusion', 'acute coronary syndrome', 'acs'],
            'pneumonia': ['pneumonitis', 'lung infection', 'pulmonary infection', 'lobar pneumonia', 'bronchopneumonia', 'aspiration pneumonia', 'cap', 'hap'],
            'asthma': ['bronchial asthma', 'asthmatic', 'reactive airway disease', 'exercise-induced asthma', 'allergic asthma'],
            'copd': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis', 'obstructive lung disease'],
            'heart failure': ['congestive heart failure', 'chf', 'cardiac failure', 'hf', 'left ventricular failure', 'right heart failure', 'systolic dysfunction', 'diastolic dysfunction'],
            'stroke': ['cerebrovascular accident', 'cva', 'brain attack', 'cerebral infarction', 'ischemic stroke', 'hemorrhagic stroke', 'tia', 'transient ischemic attack'],
            'depression': ['major depressive disorder', 'mdd', 'depressive episode', 'clinical depression', 'unipolar depression'],
            'anxiety': ['anxiety disorder', 'generalized anxiety', 'panic disorder', 'gad', 'social anxiety', 'phobia'],
            'arthritis': ['osteoarthritis', 'rheumatoid arthritis', 'ra', 'oa', 'joint inflammation', 'degenerative joint disease', 'djd'],
            'cancer': ['malignancy', 'tumor', 'neoplasm', 'carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'metastasis', 'adenocarcinoma'],
            'infection': ['sepsis', 'bacteremia', 'infectious disease', 'septicemia', 'systemic infection', 'localized infection'],
            'kidney disease': ['chronic kidney disease', 'ckd', 'renal failure', 'nephropathy', 'esrd', 'end stage renal disease', 'acute kidney injury', 'aki'],
            'liver disease': ['hepatitis', 'cirrhosis', 'hepatic failure', 'liver cirrhosis', 'fatty liver', 'nafld', 'alcoholic liver disease'],
            'atrial fibrillation': ['afib', 'af', 'atrial flutter', 'arrhythmia', 'irregular heartbeat'],
            'pulmonary embolism': ['pe', 'lung embolism', 'pulmonary thromboembolism'],
            'deep vein thrombosis': ['dvt', 'venous thrombosis', 'leg clot'],
            'hypothyroidism': ['underactive thyroid', 'low thyroid', 'myxedema'],
            'hyperthyroidism': ['overactive thyroid', 'thyrotoxicosis', 'graves disease'],
            'anemia': ['low hemoglobin', 'iron deficiency anemia', 'pernicious anemia', 'aplastic anemia'],
            'osteoporosis': ['bone loss', 'low bone density', 'osteopenia'],
            'peptic ulcer': ['gastric ulcer', 'duodenal ulcer', 'stomach ulcer', 'pud'],
            'gerd': ['gastroesophageal reflux disease', 'acid reflux', 'heartburn', 'reflux'],
            'pancreatitis': ['pancreatic inflammation', 'acute pancreatitis', 'chronic pancreatitis'],
            'cholecystitis': ['gallbladder inflammation', 'acute cholecystitis'],
            'appendicitis': ['acute appendicitis', 'inflamed appendix'],
            'diverticulitis': ['diverticular disease', 'colonic diverticulitis'],
            'celiac disease': ['gluten sensitivity', 'celiac sprue', 'gluten enteropathy'],
            'crohn disease': ['crohns', 'inflammatory bowel disease', 'ibd', 'regional enteritis'],
            'ulcerative colitis': ['uc', 'colitis', 'inflammatory bowel disease'],
            'multiple sclerosis': ['ms', 'demyelinating disease'],
            'parkinson disease': ['parkinsons', 'pd', 'parkinsonism'],
            'alzheimer disease': ['alzheimers', 'dementia', 'senile dementia'],
            'epilepsy': ['seizure disorder', 'convulsions', 'epileptic'],
            'migraine': ['migraine headache', 'vascular headache'],
            'glaucoma': ['increased intraocular pressure', 'iop', 'optic neuropathy'],
            'cataracts': ['lens opacity', 'cloudy lens'],
            'macular degeneration': ['amd', 'age-related macular degeneration'],
            'psoriasis': ['plaque psoriasis', 'psoriatic'],
            'eczema': ['atopic dermatitis', 'dermatitis'],
            'lupus': ['sle', 'systemic lupus erythematosus', 'lupus erythematosus'],
            'scleroderma': ['systemic sclerosis', 'hardening of skin'],
            'gout': ['gouty arthritis', 'hyperuricemia', 'uric acid arthritis'],
            'benign prostatic hyperplasia': ['bph', 'enlarged prostate', 'prostatic hypertrophy'],
            'urinary tract infection': ['uti', 'bladder infection', 'cystitis', 'pyelonephritis'],
            'endometriosis': ['endometrial implants'],
            'polycystic ovary syndrome': ['pcos', 'polycystic ovarian syndrome'],
            'hyperlipidemia': ['high cholesterol', 'dyslipidemia', 'hypercholesterolemia'],
            'metabolic syndrome': ['syndrome x', 'insulin resistance syndrome'],
            'obesity': ['overweight', 'morbid obesity', 'bmi over 30'],
            'sleep apnea': ['obstructive sleep apnea', 'osa', 'sleep disordered breathing'],
            'schizophrenia': ['psychotic disorder', 'psychosis'],
            'bipolar disorder': ['manic depression', 'bipolar affective disorder'],
            'adhd': ['attention deficit hyperactivity disorder', 'add', 'attention deficit disorder'],
            'autism': ['autism spectrum disorder', 'asd', 'autistic'],
            'tuberculosis': ['tb', 'pulmonary tuberculosis', 'mycobacterial infection'],
            'hiv': ['human immunodeficiency virus', 'aids', 'acquired immunodeficiency syndrome'],
            'hepatitis c': ['hcv', 'hep c'],
            'hepatitis b': ['hbv', 'hep b'],
            'lyme disease': ['borrelia', 'tick-borne illness'],
            'meningitis': ['meningeal inflammation', 'bacterial meningitis', 'viral meningitis'],
            'encephalitis': ['brain inflammation', 'viral encephalitis']
        }
        
        # Symptoms with variations
        self.symptoms = {
            'chest pain': ['chest discomfort', 'thoracic pain', 'angina', 'chest tightness', 'substernal pain', 'precordial pain', 'retrosternal pain'],
            'shortness of breath': ['dyspnea', 'breathlessness', 'sob', 'respiratory distress', 'air hunger', 'difficulty breathing'],
            'abdominal pain': ['stomach pain', 'belly pain', 'abdominal discomfort', 'epigastric pain', 'lower abdominal pain', 'upper abdominal pain'],
            'headache': ['cephalgia', 'head pain', 'migraine', 'tension headache', 'cluster headache'],
            'nausea': ['feeling sick', 'queasiness', 'stomach upset', 'nauseous'],
            'vomiting': ['emesis', 'throwing up', 'retching', 'vomited'],
            'fever': ['pyrexia', 'elevated temperature', 'febrile', 'hyperthermia', 'high temperature'],
            'fatigue': ['tiredness', 'exhaustion', 'weakness', 'malaise', 'lethargy', 'asthenia'],
            'dizziness': ['vertigo', 'lightheadedness', 'giddiness', 'spinning sensation', 'unsteadiness'],
            'palpitations': ['heart racing', 'rapid heartbeat', 'tachycardia', 'irregular heartbeat', 'heart fluttering'],
            'swelling': ['edema', 'fluid retention', 'bloating', 'peripheral edema', 'ankle swelling'],
            'rash': ['skin eruption', 'dermatitis', 'skin irritation', 'erythema', 'hives', 'urticaria'],
            'joint pain': ['arthralgia', 'joint ache', 'joint stiffness', 'joint swelling'],
            'back pain': ['lumbar pain', 'spinal pain', 'backache', 'lower back pain', 'upper back pain'],
            'cough': ['productive cough', 'dry cough', 'persistent cough', 'chronic cough', 'hemoptysis'],
            'wheezing': ['bronchospasm', 'whistling breath', 'stridor'],
            'syncope': ['fainting', 'loss of consciousness', 'passing out', 'blackout'],
            'confusion': ['altered mental status', 'disorientation', 'delirium', 'mental confusion'],
            'seizure': ['convulsion', 'fit', 'epileptic episode', 'tonic-clonic seizure'],
            'bleeding': ['hemorrhage', 'blood loss', 'hematoma', 'bruising', 'ecchymosis'],
            'numbness': ['paresthesia', 'tingling', 'loss of sensation', 'pins and needles'],
            'vision changes': ['blurred vision', 'diplopia', 'double vision', 'visual disturbance', 'loss of vision'],
            'hearing loss': ['deafness', 'auditory impairment', 'hearing impairment'],
            'tinnitus': ['ringing in ears', 'ear ringing', 'buzzing in ears'],
            'difficulty swallowing': ['dysphagia', 'swallowing difficulty', 'odynophagia'],
            'hoarseness': ['voice changes', 'dysphonia', 'raspy voice'],
            'sore throat': ['pharyngitis', 'throat pain', 'odynophagia'],
            'difficulty urinating': ['dysuria', 'urinary retention', 'hesitancy', 'painful urination'],
            'frequent urination': ['polyuria', 'urinary frequency', 'nocturia'],
            'incontinence': ['urinary incontinence', 'loss of bladder control', 'enuresis'],
            'diarrhea': ['loose stools', 'frequent bowel movements', 'watery stools'],
            'constipation': ['hard stools', 'infrequent bowel movements', 'difficulty passing stool'],
            'blood in stool': ['melena', 'hematochezia', 'rectal bleeding', 'black tarry stools'],
            'jaundice': ['icterus', 'yellow skin', 'yellowing', 'hyperbilirubinemia'],
            'weight loss': ['unintentional weight loss', 'cachexia', 'wasting'],
            'weight gain': ['weight increase', 'obesity'],
            'night sweats': ['nocturnal sweats', 'diaphoresis', 'excessive sweating'],
            'chills': ['rigors', 'shaking', 'shivering'],
            'tremor': ['shaking', 'trembling', 'involuntary movement'],
            'muscle weakness': ['myasthenia', 'weakness', 'muscle fatigue'],
            'muscle pain': ['myalgia', 'muscle ache', 'muscle soreness'],
            'stiffness': ['rigidity', 'joint stiffness', 'morning stiffness'],
            'anxiety symptoms': ['nervousness', 'panic', 'worry', 'restlessness'],
            'depression symptoms': ['sadness', 'low mood', 'anhedonia', 'hopelessness'],
            'insomnia': ['sleep disturbance', 'difficulty sleeping', 'sleeplessness'],
            'excessive thirst': ['polydipsia', 'increased thirst'],
            'excessive hunger': ['polyphagia', 'increased appetite']
        }
        
        # Medications with brand names and generics
        self.medications = {
            'aspirin': ['acetylsalicylic acid', 'asa', 'bayer', 'ecotrin'],
            'metformin': ['glucophage', 'fortamet', 'glumetza', 'riomet'],
            'lisinopril': ['prinivil', 'zestril', 'ace inhibitor'],
            'atenolol': ['tenormin', 'beta blocker'],
            'simvastatin': ['zocor', 'statin'],
            'atorvastatin': ['lipitor', 'statin'],
            'omeprazole': ['prilosec', 'proton pump inhibitor', 'ppi'],
            'pantoprazole': ['protonix', 'ppi'],
            'insulin': ['insulin glargine', 'lantus', 'humalog', 'novolog', 'insulin aspart', 'insulin lispro'],
            'warfarin': ['coumadin', 'anticoagulant', 'blood thinner'],
            'apixaban': ['eliquis', 'anticoagulant', 'doac', 'novel anticoagulant'],
            'rivaroxaban': ['xarelto', 'anticoagulant'],
            'prednisone': ['prednisolone', 'corticosteroid', 'steroid'],
            'dexamethasone': ['decadron', 'steroid'],
            'morphine': ['opioid', 'narcotic', 'pain medication'],
            'oxycodone': ['oxycontin', 'percocet', 'opioid'],
            'hydrocodone': ['vicodin', 'norco', 'opioid'],
            'furosemide': ['lasix', 'diuretic', 'water pill', 'loop diuretic'],
            'hydrochlorothiazide': ['hctz', 'microzide', 'diuretic', 'thiazide'],
            'albuterol': ['ventolin', 'proair', 'bronchodilator', 'beta agonist'],
            'levothyroxine': ['synthroid', 'levoxyl', 'thyroid hormone'],
            'amlodipine': ['norvasc', 'calcium channel blocker', 'ccb'],
            'metoprolol': ['lopressor', 'toprol', 'beta blocker'],
            'carvedilol': ['coreg', 'beta blocker'],
            'losartan': ['cozaar', 'arb', 'angiotensin receptor blocker'],
            'valsartan': ['diovan', 'arb'],
            'clopidogrel': ['plavix', 'antiplatelet', 'blood thinner'],
            'amoxicillin': ['amoxil', 'antibiotic', 'penicillin'],
            'azithromycin': ['zithromax', 'z-pak', 'antibiotic', 'macrolide'],
            'ciprofloxacin': ['cipro', 'antibiotic', 'fluoroquinolone'],
            'doxycycline': ['vibramycin', 'antibiotic'],
            'cephalexin': ['keflex', 'antibiotic', 'cephalosporin'],
            'fluoxetine': ['prozac', 'ssri', 'antidepressant'],
            'sertraline': ['zoloft', 'ssri', 'antidepressant'],
            'escitalopram': ['lexapro', 'ssri'],
            'duloxetine': ['cymbalta', 'snri'],
            'gabapentin': ['neurontin', 'anticonvulsant', 'neuropathic pain medication'],
            'pregabalin': ['lyrica', 'anticonvulsant'],
            'lorazepam': ['ativan', 'benzodiazepine', 'benzo'],
            'alprazolam': ['xanax', 'benzodiazepine'],
            'diazepam': ['valium', 'benzodiazepine'],
            'zolpidem': ['ambien', 'sleep medication', 'hypnotic'],
            'tramadol': ['ultram', 'pain medication', 'analgesic'],
            'ibuprofen': ['advil', 'motrin', 'nsaid', 'anti-inflammatory'],
            'naproxen': ['aleve', 'naprosyn', 'nsaid'],
            'acetaminophen': ['tylenol', 'paracetamol', 'analgesic'],
            'montelukast': ['singulair', 'leukotriene inhibitor'],
            'allopurinol': ['zyloprim', 'gout medication'],
            'tamsulosin': ['flomax', 'alpha blocker'],
            'finasteride': ['proscar', 'propecia', '5-alpha reductase inhibitor'],
            'digoxin': ['lanoxin', 'cardiac glycoside'],
            'spironolactone': ['aldactone', 'potassium sparing diuretic'],
            'nitroglycerin': ['nitro', 'nitrate', 'angina medication'],
            'heparin': ['anticoagulant', 'blood thinner'],
            'enoxaparin': ['lovenox', 'low molecular weight heparin', 'lmwh'],
            'vancomycin': ['vancocin', 'antibiotic'],
            'methylprednisolone': ['medrol', 'solu-medrol', 'steroid'],
            'ranitidine': ['zantac', 'h2 blocker'],
            'famotidine': ['pepcid', 'h2 blocker'],
            'diphenhydramine': ['benadryl', 'antihistamine'],
            'cetirizine': ['zyrtec', 'antihistamine'],
            'fexofenadine': ['allegra', 'antihistamine'],
            'ondansetron': ['zofran', 'antiemetic'],
            'metoclopramide': ['reglan', 'antiemetic', 'prokinetic']
        }
        
        # Enhanced tests and procedures
        self.tests = {
            'electrocardiogram': ['ecg', 'ekg', '12-lead ecg', 'cardiac monitoring'],
            'computed tomography': ['ct scan', 'cat scan', 'ct', 'ct imaging'],
            'magnetic resonance imaging': ['mri', 'mri scan', 'magnetic resonance'],
            'complete blood count': ['cbc', 'full blood count', 'hemogram'],
            'comprehensive metabolic panel': ['cmp', 'metabolic panel', 'chemistry panel'],
            'basic metabolic panel': ['bmp', 'chem 7', 'electrolytes'],
            'lipid panel': ['cholesterol panel', 'lipid profile', 'fasting lipids'],
            'liver function tests': ['lfts', 'hepatic panel', 'liver panel'],
            'renal function tests': ['kidney function', 'creatinine', 'bun'],
            'blood urea nitrogen': ['bun'],
            'creatinine': ['serum creatinine', 'scr'],
            'glomerular filtration rate': ['gfr', 'egfr', 'estimated gfr'],
            'brain natriuretic peptide': ['bnp', 'pro-bnp', 'nt-probnp'],
            'troponin': ['cardiac troponin', 'troponin i', 'troponin t'],
            'hemoglobin a1c': ['hba1c', 'glycated hemoglobin', 'glycohemoglobin', 'a1c'],
            'thyroid stimulating hormone': ['tsh', 'thyrotropin'],
            'free t4': ['thyroxine', 'free thyroxine'],
            'free t3': ['triiodothyronine'],
            'prostate specific antigen': ['psa'],
            'urinalysis': ['ua', 'urine test', 'urine dipstick'],
            'urine culture': ['urine cx'],
            'blood culture': ['blood cx'],
            'sputum culture': ['sputum cx'],
            'echocardiogram': ['echo', 'cardiac ultrasound', 'transthoracic echo', 'tte'],
            'stress test': ['exercise stress test', 'nuclear stress test', 'treadmill test', 'cardiac stress test'],
            'cardiac catheterization': ['cardiac cath', 'angiogram', 'coronary angiography'],
            'colonoscopy': ['lower endoscopy', 'colon screening'],
            'endoscopy': ['upper endoscopy', 'egd', 'esophagogastroduodenoscopy'],
            'ultrasound': ['sonogram', 'us', 'ultrasonography'],
            'x-ray': ['radiograph', 'plain film', 'chest x-ray', 'cxr'],
            'mammogram': ['breast imaging', 'screening mammography'],
            'bone density scan': ['dexa scan', 'dxa', 'bone densitometry'],
            'pulmonary function test': ['pft', 'spirometry', 'lung function test'],
            'arterial blood gas': ['abg', 'blood gas'],
            'coagulation studies': ['pt', 'ptt', 'inr', 'prothrombin time'],
            'lumbar puncture': ['spinal tap', 'lp', 'csf analysis'],
            'biopsy': ['tissue biopsy', 'pathology'],
            'pap smear': ['cervical cytology', 'pap test'],
            'eeg': ['electroencephalogram', 'brain wave test'],
            'emg': ['electromyography', 'nerve conduction study'],
            'holter monitor': ['ambulatory ecg', '24-hour monitor', 'cardiac monitor'],
            'sleep study': ['polysomnography', 'psg', 'sleep test'],
            'd-dimer': ['fibrin degradation product'],
            'sed rate': ['esr', 'erythrocyte sedimentation rate'],
            'c-reactive protein': ['crp', 'inflammatory marker'],
            'rheumatoid factor': ['rf'],
            'ana': ['antinuclear antibody'],
            'cea': ['carcinoembryonic antigen', 'tumor marker'],
            'ca 19-9': ['cancer antigen', 'tumor marker'],
            'ca 125': ['cancer antigen', 'ovarian tumor marker'],
            'vitamin d': ['25-hydroxyvitamin d', 'vitamin d level'],
            'vitamin b12': ['cobalamin', 'b12 level'],
            'folate': ['folic acid', 'folate level'],
            'iron studies': ['serum iron', 'ferritin', 'tibc', 'transferrin saturation']
        }
        
        # Body parts with anatomical variations
        self.body_parts = {
            'heart': ['cardiac', 'myocardium', 'coronary', 'atrium', 'ventricle', 'valve'],
            'lung': ['pulmonary', 'respiratory', 'bronchial', 'alveolar', 'pleural'],
            'kidney': ['renal', 'nephro', 'urinary'],
            'liver': ['hepatic', 'hepato', 'biliary'],
            'brain': ['cerebral', 'neurological', 'cns', 'cerebrum', 'cerebellum'],
            'stomach': ['gastric', 'gastro'],
            'intestine': ['bowel', 'gut', 'gastrointestinal', 'enteric', 'colon', 'small bowel'],
            'blood vessel': ['vascular', 'arterial', 'venous', 'artery', 'vein', 'capillary'],
            'bone': ['skeletal', 'osseous', 'vertebra', 'spine'],
            'muscle': ['muscular', 'myopathy', 'musculoskeletal'],
            'pancreas': ['pancreatic'],
            'spleen': ['splenic'],
            'thyroid': ['thyroid gland', 'thyroidal'],
            'adrenal': ['adrenal gland', 'suprarenal'],
            'pituitary': ['pituitary gland', 'hypophysis'],
            'bladder': ['urinary bladder', 'vesical'],
            'urethra': ['urethral'],
            'prostate': ['prostatic'],
            'uterus': ['uterine', 'endometrial'],
            'ovary': ['ovarian'],
            'breast': ['mammary'],
            'skin': ['dermal', 'cutaneous', 'integumentary'],
            'eye': ['ocular', 'ophthalmic', 'retina', 'cornea'],
            'ear': ['auditory', 'otic', 'tympanic'],
            'nose': ['nasal', 'sinus'],
            'throat': ['pharyngeal', 'laryngeal'],
            'esophagus': ['esophageal'],
            'gallbladder': ['cholecystic', 'biliary'],
            'appendix': ['appendiceal'],
            'rectum': ['rectal', 'anorectal'],
            'anus': ['anal'],
            'joint': ['articular', 'synovial'],
            'cartilage': ['chondral'],
            'tendon': ['tendinous'],
            'ligament': ['ligamentous'],
            'nerve': ['neural', 'neurological', 'peripheral nerve'],
            'spinal cord': ['myelopathy', 'spinal'],
            'lymph node': ['lymphatic', 'lymphoid'],
            'tonsil': ['tonsillar'],
            'adenoid': ['adenoidal']
        }
        
        # Severity indicators
        self.severity_indicators = {
            'mild': ['slight', 'minor', 'minimal', 'low-grade', 'trivial', 'negligible'],
            'moderate': ['moderate', 'medium', 'intermediate', 'moderately severe'],
            'severe': ['severe', 'serious', 'marked', 'significant', 'profound', 'critical', 'grave'],
            'acute': ['sudden', 'rapid', 'immediate', 'emergent', 'urgent', 'abrupt'],
            'chronic': ['long-term', 'persistent', 'ongoing', 'longstanding', 'recurrent'],
            'progressive': ['worsening', 'advancing', 'deteriorating', 'declining'],
            'stable': ['unchanged', 'steady', 'controlled', 'maintained'],
            'resolving': ['improving', 'recovering', 'healing', 'subsiding']
        }
        
        # Certainty indicators
        self.certainty_indicators = {
            'definite': ['confirmed', 'diagnosed', 'established', 'proven', 'documented'],
            'probable': ['likely', 'probable', 'suspected', 'presumed', 'suggestive'],
            'possible': ['possible', 'potential', 'may have', 'could be', 'questionable'],
            'rule_out': ['rule out', 'r/o', 'exclude', 'differential', 'consider']
        }
        
        # Temporal indicators
        self.temporal_indicators = {
            'current': ['present', 'active', 'ongoing', 'current'],
            'past': ['history of', 'previous', 'prior', 'former', 'old'],
            'recent': ['recent', 'new onset', 'newly diagnosed'],
            'childhood': ['since childhood', 'lifelong', 'congenital'],
            'recurrent': ['recurrent', 'recurring', 'repeated', 'episodic']
        }
        
        # Laterality indicators
        self.laterality = {
            'left': ['left', 'left-sided', 'sinister'],
            'right': ['right', 'right-sided', 'dexter'],
            'bilateral': ['bilateral', 'both sides', 'bilaterally'],
            'unilateral': ['unilateral', 'one-sided']
        }
        
        # Anatomical locations
        self.anatomical_locations = {
            'upper': ['superior', 'proximal', 'upper', 'cranial'],
            'lower': ['inferior', 'distal', 'lower', 'caudal'],
            'anterior': ['front', 'ventral', 'anterior'],
            'posterior': ['back', 'dorsal', 'posterior'],
            'medial': ['inner', 'medial', 'middle'],
            'lateral': ['outer', 'lateral', 'side'],
            'central': ['central', 'midline', 'middle']
        }
        
        # Treatment modalities
        self.treatments = {
            'surgery': ['surgical intervention', 'operation', 'procedure', 'operative', 'resection', 'excision'],
            'radiation': ['radiotherapy', 'radiation therapy', 'xrt', 'irradiation'],
            'chemotherapy': ['chemo', 'cytotoxic therapy', 'antineoplastic therapy'],
            'physical therapy': ['pt', 'physiotherapy', 'rehabilitation'],
            'occupational therapy': ['ot'],
            'dialysis': ['hemodialysis', 'peritoneal dialysis', 'renal replacement therapy'],
            'oxygen therapy': ['supplemental oxygen', 'o2 therapy'],
            'ventilation': ['mechanical ventilation', 'intubation', 'respiratory support'],
            'transfusion': ['blood transfusion', 'packed red blood cells', 'prbc']
        }
        
        # Vital signs
        self.vital_signs = {
            'blood pressure': ['bp', 'systolic', 'diastolic', 'hypertensive', 'hypotensive'],
            'heart rate': ['pulse', 'hr', 'bpm', 'beats per minute'],
            'respiratory rate': ['rr', 'breathing rate', 'respirations'],
            'temperature': ['temp', 'fever', 'afebrile', 'febrile'],
            'oxygen saturation': ['spo2', 'o2 sat', 'pulse ox', 'saturation']
        }
        
        # Allergies and adverse reactions
        self.allergy_terms = {
            'allergy': ['allergic', 'hypersensitivity', 'allergic reaction'],
            'anaphylaxis': ['anaphylactic', 'severe allergic reaction'],
            'adverse reaction': ['adverse effect', 'side effect', 'drug reaction', 'intolerance'],
            'rash': ['urticaria', 'hives', 'skin reaction', 'erythema'],
            'nausea from medication': ['medication-induced nausea', 'drug-induced nausea']
        }
        
        # Social history terms
        self.social_history = {
            'smoking': ['tobacco use', 'cigarette', 'smoker', 'pack years', 'nicotine'],
            'alcohol': ['ethanol', 'drinking', 'alcohol use', 'alcoholic', 'etoh'],
            'drug use': ['substance abuse', 'illicit drugs', 'recreational drugs', 'narcotics'],
            'exercise': ['physical activity', 'sedentary', 'active lifestyle'],
            'occupation': ['work', 'employment', 'occupational exposure']
        }
        
        # Family history terms
        self.family_history = {
            'family history': ['fh', 'familial', 'hereditary', 'genetic predisposition'],
            'maternal': ['mother', "mother's side", 'maternal lineage'],
            'paternal': ['father', "father's side", 'paternal lineage'],
            'sibling': ['brother', 'sister', 'siblings'],
            'grandparent': ['grandmother', 'grandfather', 'grandparents']
        }
        
        # Surgical procedures
        self.procedures = {
            'appendectomy': ['appendix removal', 'removal of appendix'],
            'cholecystectomy': ['gallbladder removal', 'gb removal'],
            'hysterectomy': ['uterus removal', 'removal of uterus'],
            'mastectomy': ['breast removal'],
            'prostatectomy': ['prostate removal'],
            'colectomy': ['colon resection', 'bowel resection'],
            'coronary artery bypass': ['cabg', 'bypass surgery', 'heart bypass'],
            'angioplasty': ['pci', 'percutaneous coronary intervention', 'stent placement'],
            'hip replacement': ['total hip arthroplasty', 'tha', 'hip arthroplasty'],
            'knee replacement': ['total knee arthroplasty', 'tka', 'knee arthroplasty'],
            'cataract surgery': ['cataract extraction', 'lens replacement'],
            'tonsillectomy': ['tonsil removal'],
            'cesarean section': ['c-section', 'cesarean delivery', 'cs'],
            'laparoscopy': ['laparoscopic surgery', 'minimally invasive surgery'],
            'arthroscopy': ['arthroscopic surgery', 'joint scope'],
            'biopsy': ['tissue sampling', 'needle biopsy', 'excisional biopsy'],
            'lumpectomy': ['breast-conserving surgery', 'partial mastectomy'],
            'spinal fusion': ['spondylodesis', 'vertebral fusion'],
            'hernia repair': ['herniorrhaphy', 'hernioplasty'],
            'pacemaker insertion': ['pacemaker placement', 'cardiac pacemaker'],
            'icd placement': ['defibrillator implantation', 'implantable cardioverter defibrillator'],
            'tracheostomy': ['trach', 'tracheotomy'],
            'thoracotomy': ['chest surgery', 'open chest procedure'],
            'craniotomy': ['skull surgery', 'brain surgery access']
        }
        
        # Clinical findings
        self.clinical_findings = {
            'murmur': ['heart murmur', 'cardiac murmur', 'systolic murmur', 'diastolic murmur'],
            'rales': ['crackles', 'pulmonary rales', 'lung crackles'],
            'wheezes': ['wheezing', 'bronchospasm', 'expiratory wheeze'],
            'hepatomegaly': ['enlarged liver', 'liver enlargement'],
            'splenomegaly': ['enlarged spleen', 'spleen enlargement'],
            'lymphadenopathy': ['swollen lymph nodes', 'enlarged lymph nodes'],
            'ascites': ['abdominal fluid', 'peritoneal fluid'],
            'pleural effusion': ['fluid in lungs', 'pleural fluid'],
            'clubbing': ['digital clubbing', 'finger clubbing'],
            'cyanosis': ['blue discoloration', 'bluish skin'],
            'pallor': ['pale', 'paleness', 'pale skin'],
            'jaundice': ['icterus', 'yellowing', 'yellow discoloration'],
            'petechiae': ['pinpoint bleeding', 'small hemorrhages'],
            'ecchymosis': ['bruising', 'bruise', 'contusion'],
            'organomegaly': ['organ enlargement', 'enlarged organ']
        }
        
        # Lab values and ranges
        self.lab_values = {
            'elevated': ['high', 'increased', 'raised', 'above normal'],
            'decreased': ['low', 'reduced', 'below normal', 'depressed'],
            'normal': ['within normal limits', 'wnl', 'unremarkable', 'normal range'],
            'critical': ['critically high', 'critically low', 'panic value'],
            'negative': ['negative', 'not detected', 'absent'],
            'positive': ['positive', 'detected', 'present', 'reactive']
        }
        
        # Imaging findings
        self.imaging_findings = {
            'mass': ['lesion', 'nodule', 'tumor', 'growth', 'space-occupying lesion'],
            'consolidation': ['infiltrate', 'opacity', 'pulmonary consolidation'],
            'atelectasis': ['lung collapse', 'collapsed lung'],
            'pneumothorax': ['collapsed lung', 'air in pleural space'],
            'fracture': ['break', 'broken bone', 'bone fracture'],
            'dislocation': ['joint dislocation', 'displaced joint'],
            'stenosis': ['narrowing', 'stricture', 'constriction'],
            'occlusion': ['blockage', 'obstruction', 'complete blockage'],
            'aneurysm': ['dilation', 'bulge', 'vascular aneurysm'],
            'hemorrhage': ['bleeding', 'blood', 'hematoma'],
            'infarction': ['dead tissue', 'ischemic area', 'tissue death'],
            'ischemia': ['decreased blood flow', 'poor perfusion'],
            'cardiomegaly': ['enlarged heart', 'heart enlargement'],
            'calcification': ['calcium deposits', 'calcified'],
            'edema': ['swelling', 'fluid accumulation']
        }
        
        # Microorganisms
        self.microorganisms = {
            'bacteria': ['bacterial', 'bacterium', 'gram positive', 'gram negative'],
            'virus': ['viral', 'viruses'],
            'fungus': ['fungal', 'yeast', 'mold'],
            'parasite': ['parasitic', 'parasites'],
            'staphylococcus': ['staph', 's aureus', 'mrsa', 'mssa'],
            'streptococcus': ['strep', 's pneumoniae', 's pyogenes'],
            'escherichia coli': ['e coli', 'ecoli'],
            'pseudomonas': ['p aeruginosa'],
            'clostridium': ['c diff', 'c difficile', 'clostridium difficile'],
            'mycobacterium': ['tb', 'tuberculosis', 'm tuberculosis'],
            'candida': ['yeast infection', 'candidiasis'],
            'influenza': ['flu', 'influenza virus'],
            'covid': ['covid-19', 'coronavirus', 'sars-cov-2'],
            'herpes': ['hsv', 'herpes simplex', 'herpes virus'],
            'hepatitis virus': ['hav', 'hbv', 'hcv', 'hepatitis a', 'hepatitis b', 'hepatitis c']
        }
        
        # Medical specialties
        self.specialties = {
            'cardiology': ['cardiac', 'heart specialist', 'cardiologist'],
            'pulmonology': ['pulmonary', 'lung specialist', 'pulmonologist'],
            'gastroenterology': ['gi', 'gastroenterologist', 'gi specialist'],
            'neurology': ['neurologist', 'neurological', 'neuro'],
            'nephrology': ['nephrologist', 'kidney specialist'],
            'endocrinology': ['endocrinologist', 'hormone specialist'],
            'hematology': ['hematologist', 'blood specialist'],
            'oncology': ['oncologist', 'cancer specialist'],
            'rheumatology': ['rheumatologist', 'arthritis specialist'],
            'dermatology': ['dermatologist', 'skin specialist'],
            'psychiatry': ['psychiatrist', 'mental health'],
            'orthopedics': ['orthopedic', 'bone specialist', 'orthopedist'],
            'urology': ['urologist', 'urinary specialist'],
            'gynecology': ['gynecologist', 'women\'s health'],
            'obstetrics': ['obstetrician', 'ob', 'pregnancy specialist'],
            'ophthalmology': ['ophthalmologist', 'eye specialist'],
            'otolaryngology': ['ent', 'ear nose throat'],
            'surgery': ['surgeon', 'surgical', 'operative'],
            'emergency medicine': ['er', 'emergency', 'emergency physician'],
            'internal medicine': ['internist', 'general medicine'],
            'family medicine': ['family practice', 'primary care'],
            'pediatrics': ['pediatrician', 'children\'s doctor'],
            'geriatrics': ['geriatrician', 'elderly care'],
            'anesthesiology': ['anesthesiologist', 'anesthesia'],
            'radiology': ['radiologist', 'imaging specialist'],
            'pathology': ['pathologist', 'lab medicine']
        }
        
        # Units of measurement
        self.units = {
            'blood pressure': ['mmhg', 'mm hg', 'millimeters of mercury'],
            'weight': ['kg', 'kilogram', 'lb', 'pound', 'lbs'],
            'height': ['cm', 'centimeter', 'inch', 'inches', 'feet', 'ft'],
            'temperature': ['celsius', 'fahrenheit', 'degrees', 'c', 'f'],
            'laboratory': ['mg/dl', 'mmol/l', 'u/l', 'iu/l', 'pg/ml', 'ng/ml', 'mcg/ml'],
            'volume': ['ml', 'milliliter', 'liter', 'l', 'cc'],
            'dosage': ['mg', 'milligram', 'mcg', 'microgram', 'gram', 'g', 'units']
        }
        
        # Risk factors
        self.risk_factors = {
            'modifiable': ['smoking', 'obesity', 'sedentary lifestyle', 'poor diet', 'alcohol abuse'],
            'non_modifiable': ['age', 'gender', 'family history', 'genetics', 'race', 'ethnicity'],
            'cardiovascular': ['hypertension', 'high cholesterol', 'diabetes', 'smoking'],
            'cancer': ['smoking', 'family history', 'radiation exposure', 'chemical exposure']
        }
        
        # Patient status descriptors
        self.status_descriptors = {
            'stable': ['stable condition', 'clinically stable', 'hemodynamically stable'],
            'unstable': ['unstable', 'critical', 'deteriorating', 'decompensated'],
            'improved': ['improving', 'better', 'resolved', 'recovery'],
            'worsened': ['worse', 'worsening', 'progressive', 'declining'],
            'unchanged': ['no change', 'static', 'status quo', 'same']
        }
        
        # Negation terms (important for NLP)
        self.negations = {
            'no': ['no', 'not', 'without', 'absent', 'denies', 'negative for'],
            'never': ['never', 'never had'],
            'ruled_out': ['ruled out', 'ro', 'excluded', 'unlikely'],
            'free_of': ['free of', 'clear of', 'no evidence of']
        }
    
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
            'BODY_PART': self.body_parts,
            'PROCEDURE': self.procedures,
            'CLINICAL_FINDING': self.clinical_findings,
            'MICROORGANISM': self.microorganisms
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
            
            # Frequency patterns: "once daily", "twice a day", "three times daily"
            [{"LOWER": {"IN": ["once", "twice", "three", "four", "1", "2", "3", "4"]}}, 
             {"LOWER": {"IN": ["daily", "times", "time"]}, "OP": "?"}, 
             {"LOWER": {"IN": ["daily", "day", "per", "a"]}, "OP": "?"}],
            
            # PRN patterns: "as needed", "prn"
            [{"LOWER": {"IN": ["as", "prn"]}}, 
             {"LOWER": {"IN": ["needed", "required", "necessary"]}, "OP": "?"}],
            
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
            # Duration: "for 3 days", "x 2 weeks"
            [{"LOWER": {"IN": ["for", "x"]}}, 
             {"TEXT": {"REGEX": r"\d+"}}, 
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years", "hr", "hrs", "hours"]}}],
            
            # Ago patterns: "3 days ago", "2 weeks prior"
            [{"TEXT": {"REGEX": r"\d+"}}, 
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
        
        # Lab value patterns
        lab_patterns = [
            # Value with unit: "WBC 12.5", "glucose 180 mg/dl"
            [{"IS_ALPHA": True}, 
             {"TEXT": {"REGEX": r"\d+\.?\d*"}}, 
             {"LOWER": {"REGEX": r"\w+/\w+"}, "OP": "?"}],
            
            # Range patterns: "between 5-10", "within normal limits"
            [{"LOWER": "between"}, 
             {"TEXT": {"REGEX": r"\d+\.?\d*"}}, 
             {"ORTH": "-"}, 
             {"TEXT": {"REGEX": r"\d+\.?\d*"}}],
            
            # Normal/abnormal: "elevated glucose", "decreased sodium"
            [{"LOWER": {"IN": ["elevated", "increased", "high", "decreased", "low", "reduced", "normal"]}}, 
             {"IS_ALPHA": True}]
        ]
        
        for i, pattern in enumerate(lab_patterns):
            self.rule_matcher.add(f"LAB_VALUE_{i}", [pattern])
        
        # Anatomical location patterns
        location_patterns = [
            # Laterality: "left lower extremity", "right upper quadrant"
            [{"LOWER": {"IN": ["left", "right", "bilateral"]}}, 
             {"LOWER": {"IN": ["upper", "lower", "mid"]}, "OP": "?"}, 
             {"IS_ALPHA": True}],
            
            # Specific locations: "lower back", "upper abdomen"
            [{"LOWER": {"IN": ["upper", "lower", "mid", "central", "distal", "proximal"]}}, 
             {"IS_ALPHA": True}],
            
            # Quadrants: "RUQ", "left lower quadrant"
            [{"TEXT": {"REGEX": r"[RL][UL]Q"}}, {"LOWER": "quadrant", "OP": "?"}]
        ]
        
        for i, pattern in enumerate(location_patterns):
            self.rule_matcher.add(f"LOCATION_{i}", [pattern])
        
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
        
        # Enhanced abbreviations dictionary
        self.abbreviations = {
            # Diseases
            'mi': 'myocardial infarction',
            'dm': 'diabetes mellitus',
            't1dm': 'type 1 diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            'htn': 'hypertension',
            'chf': 'congestive heart failure',
            'cad': 'coronary artery disease',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'aki': 'acute kidney injury',
            'copd': 'chronic obstructive pulmonary disease',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'af': 'atrial fibrillation',
            'afib': 'atrial fibrillation',
            'pvd': 'peripheral vascular disease',
            'pad': 'peripheral arterial disease',
            'cva': 'cerebrovascular accident',
            'tia': 'transient ischemic attack',
            'cabg': 'coronary artery bypass graft',
            'pci': 'percutaneous coronary intervention',
            'stemi': 'st-elevation myocardial infarction',
            'nstemi': 'non-st-elevation myocardial infarction',
            'acs': 'acute coronary syndrome',
            'gerd': 'gastroesophageal reflux disease',
            'ibd': 'inflammatory bowel disease',
            'ra': 'rheumatoid arthritis',
            'oa': 'osteoarthritis',
            'ms': 'multiple sclerosis',
            'als': 'amyotrophic lateral sclerosis',
            'bph': 'benign prostatic hyperplasia',
            'pcos': 'polycystic ovary syndrome',
            'osa': 'obstructive sleep apnea',
            'adhd': 'attention deficit hyperactivity disorder',
            'ptsd': 'post-traumatic stress disorder',
            'ocd': 'obsessive compulsive disorder',
            'mdd': 'major depressive disorder',
            'gad': 'generalized anxiety disorder',
            'hiv': 'human immunodeficiency virus',
            'aids': 'acquired immunodeficiency syndrome',
            'hcv': 'hepatitis c virus',
            'hbv': 'hepatitis b virus',
            'tb': 'tuberculosis',
            'mrsa': 'methicillin-resistant staphylococcus aureus',
            'c diff': 'clostridium difficile',
            
            # Symptoms
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            'loc': 'loss of consciousness',
            'lbp': 'lower back pain',
            'rlq': 'right lower quadrant',
            'ruq': 'right upper quadrant',
            'llq': 'left lower quadrant',
            'luq': 'left upper quadrant',
            
            # Tests
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'lfts': 'liver function tests',
            'pt': 'prothrombin time',
            'ptt': 'partial thromboplastin time',
            'inr': 'international normalized ratio',
            'bnp': 'brain natriuretic peptide',
            'hba1c': 'hemoglobin a1c',
            'tsh': 'thyroid stimulating hormone',
            'psa': 'prostate specific antigen',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'echo': 'echocardiogram',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'cxr': 'chest x-ray',
            'kub': 'kidney ureter bladder',
            'ua': 'urinalysis',
            'abg': 'arterial blood gas',
            'pft': 'pulmonary function test',
            'eeg': 'electroencephalogram',
            'emg': 'electromyography',
            'lp': 'lumbar puncture',
            'egd': 'esophagogastroduodenoscopy',
            'ercp': 'endoscopic retrograde cholangiopancreatography',
            
            # Body systems
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'cv': 'cardiovascular',
            'resp': 'respiratory',
            'neuro': 'neurological',
            'psych': 'psychiatric',
            'derm': 'dermatological',
            'ent': 'ear nose throat',
            'msk': 'musculoskeletal',
            'heent': 'head eyes ears nose throat',
            'cns': 'central nervous system',
            'pns': 'peripheral nervous system',
            
            # Medications
            'asa': 'aspirin',
            'hctz': 'hydrochlorothiazide',
            'ace-i': 'ace inhibitor',
            'arb': 'angiotensin receptor blocker',
            'bb': 'beta blocker',
            'ccb': 'calcium channel blocker',
            'ppi': 'proton pump inhibitor',
            'h2ra': 'h2 receptor antagonist',
            'nsaid': 'nonsteroidal anti-inflammatory drug',
            'ssri': 'selective serotonin reuptake inhibitor',
            'snri': 'serotonin norepinephrine reuptake inhibitor',
            'tca': 'tricyclic antidepressant',
            'doac': 'direct oral anticoagulant',
            'lmwh': 'low molecular weight heparin',
            
            # General medical terms
            'h/o': 'history of',
            's/p': 'status post',
            'r/o': 'rule out',
            'w/': 'with',
            'w/o': 'without',
            'c/o': 'complains of',
            'prn': 'as needed',
            'qd': 'once daily',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sq': 'subcutaneous',
            'npo': 'nothing by mouth',
            'dnr': 'do not resuscitate',
            'dnd': 'do not disturb',
            'nkda': 'no known drug allergies',
            'nka': 'no known allergies',
            'wnl': 'within normal limits',
            'nad': 'no acute distress'
        }
        
        # Create reverse mapping for normalization
        self.normalization_map = {}
        categories_to_normalize = [
            ('DISEASE', self.diseases),
            ('SYMPTOM', self.symptoms),
            ('MEDICATION', self.medications),
            ('TEST', self.tests),
            ('BODY_PART', self.body_parts),
            ('PROCEDURE', self.procedures),
            ('CLINICAL_FINDING', self.clinical_findings),
            ('MICROORGANISM', self.microorganisms)
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
    
    def get_entity_context(self, entity_span, doc):
        """
        Extract contextual information about an entity
        
        Args:
            entity_span: The entity span from doc
            doc: The spaCy Doc object
            
        Returns:
            dict: Context information including negation, certainty, subject, temporality
        """
        context = {
            'negated': False,
            'uncertain': False,
            'certainty_level': 'definite',
            'subject': 'patient',
            'temporality': 'current',
            'conditional': False
        }
        
        # Check tokens before the entity (window of 6 tokens)
        start_idx = max(0, entity_span.start - self.negation_scope)
        preceding_tokens = doc[start_idx:entity_span.start]
        preceding_text = ' '.join([token.text.lower() for token in preceding_tokens])
        
        # Check for negation
        for neg_type, neg_terms in self.negation_triggers.items():
            for neg_term in neg_terms:
                if neg_term in preceding_text:
                    # Check if it's a pseudo-negation
                    is_pseudo = any(pseudo in preceding_text for pseudo in self.pseudo_negations)
                    if not is_pseudo:
                        context['negated'] = True
                        break
        
        # Check for uncertainty
        for certainty_level, terms in self.uncertainty_indicators.items():
            for term in terms:
                if term in preceding_text:
                    context['uncertain'] = True
                    context['certainty_level'] = certainty_level.replace('_uncertainty', '')
                    break
        
        # Check for assertion (overrides uncertainty)
        for assertion_term in self.assertion_indicators:
            if assertion_term in preceding_text:
                context['uncertain'] = False
                context['certainty_level'] = 'definite'
                break
        
        # Check subject
        for subject_type, terms in self.subject_indicators.items():
            for term in terms:
                if term in preceding_text:
                    context['subject'] = subject_type
                    break
        
        # Check temporality
        for temp_type, terms in self.historical_indicators.items():
            for term in terms:
                if term in preceding_text:
                    context['temporality'] = temp_type
                    break
        
        # Check if conditional
        for cond_term in self.conditional_indicators:
            if cond_term in preceding_text:
                context['conditional'] = True
                break
        
        return context
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
            print(f"Warning: Failed to initialize semantic analyzer: {e}")
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
            print(f"Warning: Failed to initialize concept graph: {e}")
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

