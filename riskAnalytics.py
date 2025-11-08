"""
Enhanced Medical Risk Analytics Module
Provides comprehensive, clinically-relevant risk assessment and analysis for medical entities
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from medical_nlp import MedicalEntity

# Add matplotlib imports for graph generation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import numpy as np

@dataclass
class RiskFactor:
    """Represents a risk factor with its assessment"""
    name: str
    percentage: float
    risk_level: str  # 'low', 'moderate', 'high', 'critical'
    trend: str  # 'increasing', 'stable', 'decreasing'
    description: str
    contributing_entities: int
    severity_score: float

@dataclass
class CommonCondition:
    """Represents a commonly identified condition"""
    name: str
    cases: int
    risk_level: str
    severity: str
    clinical_significance: str

@dataclass
class Recommendation:
    """Represents a clinical recommendation"""
    title: str
    description: str
    category: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    rationale: str
    timeframe: str

@dataclass
class ComorbidityCluster:
    """Represents a cluster of related conditions (comorbidities)"""
    primary_condition: str
    related_conditions: List[str]
    combined_risk_score: float
    interaction_severity: str

class MedicalRiskAnalyzer:
    """
    Advanced medical risk analyzer with enhanced clinical intelligence
    """

    def __init__(self):
        # Enhanced risk scoring weights with clinical relevance
        self.risk_weights = {
            'DISEASE': 1.0,
            'CONDITION': 0.9,
            'SYMPTOM': 0.6,
            'MEDICATION': 0.5,
            'TEST': 0.3,
            'PROCEDURE': 0.4,
            'BODY_PART': 0.2,
            'TREATMENT': 0.5,
            'DIAGNOSIS': 1.0,
            # New enriched labels (tolerant defaults)
            'CLINICAL_FINDING': 0.5,
            'IMAGING': 0.3,
            'IMAGING_FINDING': 0.4,
            'LAB': 0.3,
            'LAB_TEST': 0.3,
            'MICROORGANISM': 0.4,
            'ALLERGY': 0.2,
            'FAMILY_HISTORY': 0.2,
            'SOCIAL_HISTORY': 0.2,
            'VITAL': 0.3,
        }

        # Comprehensive high-risk conditions with clinical severity scores
        self.high_risk_conditions = {
            # Critical Cardiac Conditions
            'myocardial infarction': {'score': 95, 'category': 'cardiac', 'urgency': 'critical'},
            'heart failure': {'score': 90, 'category': 'cardiac', 'urgency': 'high'},
            'cardiac arrest': {'score': 98, 'category': 'cardiac', 'urgency': 'critical'},
            'ventricular fibrillation': {'score': 96, 'category': 'cardiac', 'urgency': 'critical'},
            'unstable angina': {'score': 88, 'category': 'cardiac', 'urgency': 'high'},
            'cardiogenic shock': {'score': 94, 'category': 'cardiac', 'urgency': 'critical'},
            'atrial fibrillation': {'score': 75, 'category': 'cardiac', 'urgency': 'moderate'},
            
            # Cerebrovascular
            'stroke': {'score': 93, 'category': 'cerebrovascular', 'urgency': 'critical'},
            'hemorrhagic stroke': {'score': 95, 'category': 'cerebrovascular', 'urgency': 'critical'},
            'ischemic stroke': {'score': 92, 'category': 'cerebrovascular', 'urgency': 'critical'},
            'transient ischemic attack': {'score': 82, 'category': 'cerebrovascular', 'urgency': 'high'},
            'subarachnoid hemorrhage': {'score': 96, 'category': 'cerebrovascular', 'urgency': 'critical'},
            
            # Respiratory
            'pneumonia': {'score': 78, 'category': 'respiratory', 'urgency': 'high'},
            'acute respiratory distress': {'score': 91, 'category': 'respiratory', 'urgency': 'critical'},
            'pulmonary embolism': {'score': 90, 'category': 'respiratory', 'urgency': 'critical'},
            'copd': {'score': 76, 'category': 'respiratory', 'urgency': 'moderate'},
            'asthma': {'score': 65, 'category': 'respiratory', 'urgency': 'moderate'},
            'respiratory failure': {'score': 93, 'category': 'respiratory', 'urgency': 'critical'},
            
            # Infections/Sepsis
            'sepsis': {'score': 94, 'category': 'infectious', 'urgency': 'critical'},
            'septic shock': {'score': 97, 'category': 'infectious', 'urgency': 'critical'},
            'meningitis': {'score': 89, 'category': 'infectious', 'urgency': 'critical'},
            'encephalitis': {'score': 88, 'category': 'infectious', 'urgency': 'critical'},
            
            # Oncological
            'cancer': {'score': 85, 'category': 'oncological', 'urgency': 'high'},
            'metastatic cancer': {'score': 95, 'category': 'oncological', 'urgency': 'critical'},
            'malignant tumor': {'score': 88, 'category': 'oncological', 'urgency': 'high'},
            'carcinoma': {'score': 86, 'category': 'oncological', 'urgency': 'high'},
            'lymphoma': {'score': 84, 'category': 'oncological', 'urgency': 'high'},
            'leukemia': {'score': 87, 'category': 'oncological', 'urgency': 'high'},
            
            # Metabolic/Endocrine
            'diabetes': {'score': 72, 'category': 'metabolic', 'urgency': 'moderate'},
            'diabetic ketoacidosis': {'score': 90, 'category': 'metabolic', 'urgency': 'critical'},
            'hypoglycemia': {'score': 75, 'category': 'metabolic', 'urgency': 'high'},
            'thyroid storm': {'score': 92, 'category': 'endocrine', 'urgency': 'critical'},
            'adrenal crisis': {'score': 93, 'category': 'endocrine', 'urgency': 'critical'},
            
            # Renal
            'kidney failure': {'score': 85, 'category': 'renal', 'urgency': 'high'},
            'acute kidney injury': {'score': 88, 'category': 'renal', 'urgency': 'high'},
            'chronic kidney disease': {'score': 78, 'category': 'renal', 'urgency': 'moderate'},
            'renal failure': {'score': 86, 'category': 'renal', 'urgency': 'high'},
            
            # Hepatic
            'liver failure': {'score': 89, 'category': 'hepatic', 'urgency': 'critical'},
            'cirrhosis': {'score': 80, 'category': 'hepatic', 'urgency': 'high'},
            'hepatic encephalopathy': {'score': 87, 'category': 'hepatic', 'urgency': 'high'},
            
            # Hematological
            'hemorrhage': {'score': 86, 'category': 'hematological', 'urgency': 'critical'},
            'anemia': {'score': 60, 'category': 'hematological', 'urgency': 'moderate'},
            'thrombocytopenia': {'score': 74, 'category': 'hematological', 'urgency': 'moderate'},
            'coagulopathy': {'score': 82, 'category': 'hematological', 'urgency': 'high'},
            
            # Chronic Conditions
            'hypertension': {'score': 68, 'category': 'cardiovascular', 'urgency': 'moderate'},
            'hyperlipidemia': {'score': 62, 'category': 'metabolic', 'urgency': 'moderate'},
            'obesity': {'score': 58, 'category': 'metabolic', 'urgency': 'low'},
        }

        # Enhanced risk factor definitions with clinical categories
        self.risk_factor_definitions = {
            'Cardiovascular Risk': {
                'keywords': ['myocardial infarction', 'heart failure', 'hypertension', 'cholesterol', 
                            'arrhythmia', 'angina', 'cardiac', 'coronary', 'valve disease', 'cardiomyopathy'],
                'weight': 1.2
            },
            'Respiratory Risk': {
                'keywords': ['pneumonia', 'copd', 'asthma', 'respiratory failure', 'pulmonary embolism',
                            'dyspnea', 'hypoxia', 'lung disease', 'bronchitis'],
                'weight': 1.1
            },
            'Metabolic Risk': {
                'keywords': ['diabetes', 'hyperglycemia', 'insulin resistance', 'obesity', 'metabolic syndrome',
                            'hyperlipidemia', 'thyroid', 'ketoacidosis'],
                'weight': 0.9
            },
            'Neurological Risk': {
                'keywords': ['stroke', 'seizure', 'neurological deficit', 'headache', 'migraine',
                            'encephalopathy', 'neuropathy', 'dementia', 'parkinson'],
                'weight': 1.1
            },
            'Oncological Risk': {
                'keywords': ['cancer', 'tumor', 'malignancy', 'carcinoma', 'metastasis', 'neoplasm',
                            'lymphoma', 'leukemia', 'sarcoma'],
                'weight': 1.3
            },
            'Infectious Risk': {
                'keywords': ['sepsis', 'infection', 'pneumonia', 'meningitis', 'bacteremia',
                            'viral', 'bacterial', 'fungal', 'abscess'],
                'weight': 1.2
            },
            'Renal Risk': {
                'keywords': ['kidney failure', 'renal', 'dialysis', 'acute kidney injury',
                            'chronic kidney disease', 'nephropathy', 'uremia'],
                'weight': 1.0
            },
            'Hepatic Risk': {
                'keywords': ['liver failure', 'cirrhosis', 'hepatitis', 'hepatic',
                            'jaundice', 'ascites', 'encephalopathy'],
                'weight': 1.0
            }
        }

        # Critical symptom patterns that elevate risk
        self.critical_symptoms = {
            'chest pain': 85,
            'shortness of breath': 80,
            'altered mental status': 88,
            'severe pain': 75,
            'loss of consciousness': 92,
            'difficulty breathing': 85,
            'confusion': 78,
            'severe headache': 80,
            'hemoptysis': 86,
            'syncope': 82
        }

        # Medication risk indicators
        self.high_risk_medications = {
            'chemotherapy': 0.8,
            'immunosuppressant': 0.7,
            'anticoagulant': 0.6,
            'insulin': 0.5,
            'opioid': 0.5,
            'corticosteroid': 0.4
        }

        # Comorbidity interaction patterns
        self.comorbidity_multipliers = {
            ('diabetes', 'hypertension'): 1.4,
            ('heart failure', 'kidney disease'): 1.5,
            ('copd', 'heart failure'): 1.4,
            ('diabetes', 'kidney disease'): 1.5,
            ('cancer', 'diabetes'): 1.3,
            ('stroke', 'atrial fibrillation'): 1.4,
        }

        # Normal/healthy indicators (more comprehensive)
        self.normal_indicators = {
            'normal': 0.4,
            'stable': 0.5,
            'unremarkable': 0.5,
            'within normal limits': 0.7,
            'within normal range': 0.7,
            'no acute distress': 0.6,
            'clear': 0.5,
            'regular': 0.4,
            'healthy': 0.8,
            'good general health': 0.9,
            'well-controlled': 0.6,
            'asymptomatic': 0.7,
            'resolved': 0.8,
            'improved': 0.6,
            'optimal': 0.7,
            'negative': 0.6,
            'absent': 0.6,
            'no evidence of': 0.6,
            'ruled out': 0.7
        }

    def generate_comprehensive_analysis(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """
        Generate enhanced comprehensive risk analysis from medical entities

        Args:
            entities: List of extracted medical entities

        Returns:
            Dictionary containing comprehensive risk analysis with clinical insights
        """
        if not entities:
            return self._empty_analysis()

        # Calculate overall risk score with enhanced algorithms
        overall_risk_score = self._calculate_overall_risk_score(entities)

        # Generate detailed risk assessment factors
        risk_assessment = self._generate_risk_assessment(entities)

        # Identify and analyze common conditions
        common_conditions = self._identify_common_conditions(entities)

        # Detect comorbidity clusters
        comorbidity_clusters = self._detect_comorbidity_clusters(entities)

        # Generate evidence-based clinical recommendations
        recommendations = self._generate_recommendations(entities, overall_risk_score, comorbidity_clusters)

        # Calculate comprehensive summary statistics
        summary_stats = self._calculate_summary_stats(entities)

        # Analyze entity patterns and relationships
        entity_patterns = self._analyze_entity_patterns(entities)

        # Assess medication-related risks
        medication_risks = self._assess_medication_risks(entities)

        # Identify critical symptoms
        critical_symptoms = self._identify_critical_symptoms(entities)

        # Extract additional clinical contexts from enriched dictionaries
        vitals = self._extract_vital_signs(entities)
        labs = self._extract_lab_tests(entities)
        allergies = self._extract_allergies(entities)
        social_history = self._extract_social_history(entities)
        family_history = self._extract_family_history(entities)
        imaging_findings = self._extract_imaging_findings(entities)
        microorganisms = self._extract_microorganisms(entities)

        # Calculate risk stratification
        risk_stratification = self._calculate_risk_stratification(overall_risk_score)

        return {
            'overall_risk_score': overall_risk_score,
            'risk_stratification': risk_stratification,
            'risk_assessment': risk_assessment,
            'common_conditions': common_conditions,
            'comorbidity_clusters': comorbidity_clusters,
            'recommendations': recommendations,
            'summary_stats': summary_stats,
            'entity_patterns': entity_patterns,
            'medication_risks': medication_risks,
            'critical_symptoms': critical_symptoms,
            'clinical_summary': self._generate_clinical_summary(entities, overall_risk_score),
            'vitals': vitals,
            'labs': labs,
            'allergies': allergies,
            'social_history': social_history,
            'family_history': family_history,
            'imaging_findings': imaging_findings,
            'microorganisms': microorganisms,
        }

    def generate_user_insights(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """
        Generate a concise insight summary based on extracted entities.
        Returns a lightweight dict optimized for UI display.
        """
        analysis = self.generate_comprehensive_analysis(entities)
        risk = analysis.get('risk_stratification', {})
        overall = analysis.get('overall_risk_score', 0)

        # Top risk factors (names and percentages)
        top_risk_factors = []
        for rf in (analysis.get('risk_assessment') or [])[:3]:
            top_risk_factors.append({
                'name': rf.name,
                'percentage': rf.percentage,
                'risk_level': rf.risk_level,
                'desc': rf.description
            })

        # Highlighted conditions
        top_conditions = []
        for c in (analysis.get('common_conditions') or [])[:5]:
            top_conditions.append({
                'name': c.name,
                'cases': c.cases,
                'risk_level': c.risk_level,
                'severity': c.severity
            })

        # Red flags (critical symptoms)
        red_flags = []
        for s in (analysis.get('critical_symptoms') or [])[:5]:
            red_flags.append({
                'symptom': s['symptom'],
                'urgency': s['urgency'],
                'severity_score': s['severity_score']
            })

        # Actionable next steps
        actions = []
        for rec in (analysis.get('recommendations') or [])[:3]:
            actions.append({
                'title': rec.title,
                'priority': rec.priority,
                'timeframe': rec.timeframe
            })

        headline = analysis.get('clinical_summary') or f"Overall risk: {overall}/10 ({risk.get('category', 'Unknown')})"

        # History and context snippets
        history = {
            'allergies': [a['term'] for a in (analysis.get('allergies') or [])][:3],
            'social': [s['term'] for s in (analysis.get('social_history') or [])][:3],
            'family': [f['term'] for f in (analysis.get('family_history') or [])][:3],
        }

        return {
            'headline': headline,
            'overall_risk': overall,
            'risk_category': risk.get('category'),
            'risk_color': risk.get('color'),
            'key_findings': {
                'risk_factors': top_risk_factors,
                'conditions': top_conditions,
                'red_flags': red_flags,
                'vitals': (analysis.get('vitals') or [])[:5],
                'labs': (analysis.get('labs') or [])[:5],
            },
            'summary_stats': analysis.get('summary_stats', {}),
            'actions': actions,
            'history': history,
        }

    # ===== Enriched extraction helpers (keyword-based, label-tolerant) =====
    def _extract_vital_signs(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = [
            'blood pressure', 'bp', 'systolic', 'diastolic',
            'heart rate', 'hr', 'pulse', 'bpm',
            'respiratory rate', 'rr',
            'temperature', 'temp', 'fever', 'afebrile',
            'oxygen saturation', 'spo2', 'o2 sat', 'pulse ox', 'saturation'
        ]
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords):
                out.append({'term': e.text, 'label': e.label, 'confidence': e.confidence})
        return out

    def _extract_lab_tests(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = [
            'cbc', 'complete blood count', 'cmp', 'metabolic panel', 'bmp', 'lipid panel',
            'lfts', 'hepatic panel', 'renal function', 'bun', 'creatinine', 'gfr', 'egfr',
            'bnp', 'troponin', 'hba1c', 'a1c', 'tsh', 'free t4', 'free t3', 'psa', 'urinalysis', 'ua',
            'urine culture', 'blood culture', 'sputum culture', 'd-dimer', 'esr', 'crp', 'rf', 'ana',
            'vitamin d', 'vitamin b12', 'folate', 'iron', 'ferritin', 'tibc'
        ]
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('TEST', 'LAB', 'LAB_TEST'):
                out.append({'term': e.text, 'label': e.label, 'confidence': e.confidence})
        return out

    def _extract_allergies(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = ['allergy', 'allergic', 'hypersensitivity', 'anaphylaxis', 'adverse reaction', 'side effect']
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('ALLERGY', 'ADVERSE_REACTION'):
                out.append({'term': e.text, 'label': e.label})
        return out

    def _extract_social_history(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = ['smoking', 'tobacco', 'cigarette', 'pack years', 'alcohol', 'etoh', 'drug', 'exercise', 'occupation']
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('SOCIAL_HISTORY',):
                out.append({'term': e.text, 'label': e.label})
        return out

    def _extract_family_history(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = ['family history', 'maternal', 'paternal', 'sibling', 'grandparent', 'hereditary', 'genetic']
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('FAMILY_HISTORY',):
                out.append({'term': e.text, 'label': e.label})
        return out

    def _extract_imaging_findings(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = ['x-ray', 'cxr', 'ct', 'mri', 'ultrasound', 'echocardiogram', 'echo', 'angiography', 'mass', 'lesion', 'nodule', 'consolidation', 'atelectasis', 'pneumothorax']
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('IMAGING', 'IMAGING_FINDING'):
                out.append({'term': e.text, 'label': e.label})
        return out

    def _extract_microorganisms(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        keywords = ['bacteria', 'viral', 'fungal', 'parasite', 'staph', 'mrsa', 'strep', 'pseudomonas', 'clostridium', 'c diff', 'tuberculosis', 'candida', 'influenza', 'covid', 'herpes', 'hepatitis']
        out = []
        for e in entities:
            t = e.text.lower()
            if any(k in t for k in keywords) or e.label in ('MICROORGANISM', 'PATHOGEN'):
                out.append({'term': e.text, 'label': e.label})
        return out

    def _calculate_overall_risk_score(self, entities: List[MedicalEntity]) -> float:
        """Enhanced risk score calculation with clinical intelligence"""
        if not entities:
            return 0.0

        base_score = 0.0
        condition_scores = []
        symptom_severity_bonus = 0.0
        medication_risk_bonus = 0.0
        normal_reduction = 0.0
        
        # Extract all conditions for comorbidity analysis
        conditions = []
        
        for entity in entities:
            # Base entity score
            entity_base = self.risk_weights.get(entity.label, 0.5)
            confidence_mult = entity.confidence
            negation_mult = 0.2 if entity.negated else 1.0
            # Contextual multipliers (if provided by extractor)
            uncertain_mult = 0.7 if getattr(entity, 'uncertain', False) else 1.0
            temporality = getattr(entity, 'temporality', 'current') or 'current'
            temporality_mult = 0.7 if temporality == 'past' else 1.0
            subject = getattr(entity, 'subject', 'patient') or 'patient'
            subject_mult = 1.0 if subject == 'patient' else 0.6
            conditional_mult = 0.7 if getattr(entity, 'conditional', False) else 1.0
            
            # Severity multiplier
            severity_mult = self._get_severity_multiplier(entity)
            
            # Check for high-risk conditions
            condition_risk = self._get_condition_risk_score(entity)
            if condition_risk > 0:
                condition_scores.append(condition_risk)
                conditions.append(entity.text.lower())
            
            # Critical symptom bonus
            symptom_risk = self._get_symptom_risk_score(entity)
            symptom_severity_bonus += symptom_risk
            
            # Medication risk assessment
            med_risk = self._get_medication_risk_score(entity)
            medication_risk_bonus += med_risk
            
            # Normal indicator reduction
            normal_reduction += self._get_normal_indicator_reduction(entity)
            
            # Calculate entity contribution
            entity_score = (
                entity_base
                * confidence_mult
                * negation_mult
                * severity_mult
                * uncertain_mult
                * temporality_mult
                * subject_mult
                * conditional_mult
            )
            base_score += entity_score
        
        # Apply comorbidity multipliers
        comorbidity_multiplier = self._calculate_comorbidity_multiplier(conditions)
        
        # Combine all components
        total_score = (
            base_score * 10 +  # Base entity score
            sum(condition_scores) * 0.5 +  # High-risk condition scores
            symptom_severity_bonus * 2 +  # Critical symptom bonus
            medication_risk_bonus * 5  # Medication risk
        )
        
        # Apply comorbidity effects
        total_score *= comorbidity_multiplier
        
        # Apply normal indicator reductions
        total_score = max(0, total_score - (normal_reduction * 10))
        
        # Normalize to 0-10 scale with clinical calibration
        normalized_score = min(10.0, max(0.0, total_score / len(entities) if entities else 0))
        
        return round(normalized_score, 2)

    def _generate_risk_assessment(self, entities: List[MedicalEntity]) -> List[RiskFactor]:
        """Generate detailed risk factor assessment with clinical context"""
        risk_factors = []

        for factor_name, factor_config in self.risk_factor_definitions.items():
            keywords = factor_config['keywords']
            weight = factor_config['weight']
            
            # Find relevant entities
            relevant_entities = []
            for entity in entities:
                entity_text = entity.text.lower()
                if any(keyword in entity_text for keyword in keywords):
                    if not entity.negated:  # Exclude negated entities
                        relevant_entities.append(entity)

            if relevant_entities:
                # Calculate severity-adjusted risk percentage
                severity_scores = [self._get_entity_severity_score(e) for e in relevant_entities]
                avg_severity = sum(severity_scores) / len(severity_scores)
                confidence_avg = sum(e.confidence for e in relevant_entities) / len(relevant_entities)
                
                # Enhanced calculation
                base_score = len(relevant_entities) * 12
                weighted_score = base_score * weight * confidence_avg * avg_severity
                percentage = min(100, weighted_score)

                risk_level = self._get_risk_level_detailed(percentage)
                trend = self._determine_trend(relevant_entities)

                # Generate clinical description
                high_risk_entities = [e for e in relevant_entities if self._is_high_risk_entity(e)]
                description = f"Identified {len(relevant_entities)} relevant clinical findings"
                if high_risk_entities:
                    description += f" including {len(high_risk_entities)} high-severity conditions"

                risk_factors.append(RiskFactor(
                    name=factor_name,
                    percentage=round(percentage, 1),
                    risk_level=risk_level,
                    trend=trend,
                    description=description,
                    contributing_entities=len(relevant_entities),
                    severity_score=round(avg_severity, 2)
                ))

        # Sort by percentage descending
        risk_factors.sort(key=lambda x: x.percentage, reverse=True)
        return risk_factors

    def _identify_common_conditions(self, entities: List[MedicalEntity]) -> List[CommonCondition]:
        """Identify conditions with enhanced clinical categorization"""
        condition_data = defaultdict(lambda: {'count': 0, 'severity_scores': [], 'entities': []})

        for entity in entities:
            if entity.label in ['DISEASE', 'CONDITION', 'DIAGNOSIS'] and not entity.negated:
                condition_name = self._norm_text_lower(entity)
                condition_data[condition_name]['count'] += 1
                condition_data[condition_name]['severity_scores'].append(
                    self._get_entity_severity_score(entity)
                )
                condition_data[condition_name]['entities'].append(entity)

        common_conditions = []
        for condition, data in condition_data.items():
            if data['count'] >= 1:
                # Get condition info
                condition_info = self.high_risk_conditions.get(condition, {
                    'score': 50, 'category': 'general', 'urgency': 'moderate'
                })
                
                risk_score = condition_info['score']
                avg_severity = sum(data['severity_scores']) / len(data['severity_scores'])
                adjusted_score = risk_score * avg_severity
                
                risk_level = self._get_risk_level_detailed(adjusted_score)
                severity = self._determine_condition_severity(data['entities'])
                clinical_sig = self._assess_clinical_significance(condition_info, data['count'])

                common_conditions.append(CommonCondition(
                    name=condition.title(),
                    cases=data['count'],
                    risk_level=risk_level,
                    severity=severity,
                    clinical_significance=clinical_sig
                ))

        # Sort by clinical importance (risk score * frequency)
        common_conditions.sort(
            key=lambda x: self.high_risk_conditions.get(x.name.lower(), {}).get('score', 50) * x.cases,
            reverse=True
        )
        
        return common_conditions[:15]

    def _detect_comorbidity_clusters(self, entities: List[MedicalEntity]) -> List[ComorbidityCluster]:
        """Detect clinically significant comorbidity patterns"""
        conditions = [
            self._norm_text_lower(e)
            for e in entities
            if e.label in ['DISEASE', 'CONDITION'] and not e.negated
        ]
        
        clusters = []
        processed = set()
        
        for primary in conditions:
            if primary in processed:
                continue
                
            related = []
            combined_score = self.high_risk_conditions.get(primary, {}).get('score', 50)
            
            for secondary in conditions:
                if secondary != primary and secondary not in processed:
                    # Check for known comorbidity interactions
                    pair = tuple(sorted([primary, secondary]))
                    if pair in self.comorbidity_multipliers:
                        related.append(secondary)
                        secondary_score = self.high_risk_conditions.get(secondary, {}).get('score', 50)
                        combined_score += secondary_score * self.comorbidity_multipliers[pair]
            
            if related:
                interaction_severity = 'High' if combined_score > 150 else 'Moderate' if combined_score > 100 else 'Low'
                clusters.append(ComorbidityCluster(
                    primary_condition=primary.title(),
                    related_conditions=[r.title() for r in related],
                    combined_risk_score=round(combined_score, 1),
                    interaction_severity=interaction_severity
                ))
                processed.add(primary)
                processed.update(related)
        
        return clusters

    def _generate_recommendations(self, entities: List[MedicalEntity], 
                                 overall_risk: float,
                                 comorbidity_clusters: List[ComorbidityCluster]) -> List[Recommendation]:
        """Generate evidence-based clinical recommendations"""
        recommendations = []

        # Critical risk recommendations
        if overall_risk >= 7.0:
            recommendations.append(Recommendation(
                title="Immediate Medical Evaluation Required",
                description="High-risk indicators detected requiring urgent clinical assessment and possible hospital admission",
                category="urgent_care",
                priority="critical",
                rationale=f"Overall risk score of {overall_risk}/10 indicates significant health concerns",
                timeframe="Immediate (within 1 hour)"
            ))
            
            recommendations.append(Recommendation(
                title="Continuous Monitoring",
                description="Implement continuous vital signs monitoring and frequent clinical reassessment",
                category="monitoring",
                priority="critical",
                rationale="Multiple high-risk factors require close observation",
                timeframe="Continuous"
            ))

        # High risk recommendations
        elif overall_risk >= 5.0:
            recommendations.append(Recommendation(
                title="Urgent Clinical Review",
                description="Schedule urgent evaluation by healthcare provider within 24 hours",
                category="assessment",
                priority="high",
                rationale=f"Risk score of {overall_risk}/10 indicates concerning clinical findings",
                timeframe="Within 24 hours"
            ))

        # Comorbidity-specific recommendations
        if comorbidity_clusters:
            for cluster in comorbidity_clusters[:2]:  # Top 2 clusters
                recommendations.append(Recommendation(
                    title=f"Integrated Management: {cluster.primary_condition}",
                    description=f"Coordinate care for {cluster.primary_condition} and related conditions: {', '.join(cluster.related_conditions[:3])}",
                    category="care_coordination",
                    priority="high" if cluster.interaction_severity == "High" else "medium",
                    rationale=f"Comorbidity interaction requires integrated approach ({cluster.interaction_severity} interaction severity)",
                    timeframe="Within 1 week"
                ))

        # System-specific recommendations
        cardio_entities = [e for e in entities if any(k in e.text.lower() for k in ['heart', 'cardiac', 'coronary', 'myocardial'])]
        if cardio_entities and not any(e.negated for e in cardio_entities):
            recommendations.append(Recommendation(
                title="Cardiovascular Workup",
                description="Comprehensive cardiovascular assessment including ECG, cardiac biomarkers, and echocardiogram as clinically indicated",
                category="diagnostic",
                priority="high" if len(cardio_entities) > 2 else "medium",
                rationale=f"Detected {len(cardio_entities)} cardiovascular-related findings",
                timeframe="Within 48-72 hours"
            ))

        # Medication recommendations
        med_entities = [e for e in entities if e.label == 'MEDICATION']
        if len(med_entities) >= 5:
            recommendations.append(Recommendation(
                title="Comprehensive Medication Reconciliation",
                description="Review all medications for potential interactions, duplications, and appropriateness",
                category="medication_safety",
                priority="medium",
                rationale=f"Patient on {len(med_entities)} medications - polypharmacy assessment needed",
                timeframe="Within 1 week"
            ))

        # Preventive recommendations
        if overall_risk >= 3.0:
            recommendations.append(Recommendation(
                title="Lifestyle Modification Counseling",
                description="Provide comprehensive education on diet, exercise, smoking cessation, and stress management",
                category="preventive",
                priority="medium",
                rationale="Risk factor modification can significantly improve health outcomes",
                timeframe="Ongoing"
            ))

        # Follow-up recommendations
        if overall_risk >= 4.0:
            timeframe = "Weekly" if overall_risk >= 7.0 else "Every 2-4 weeks"
            recommendations.append(Recommendation(
                title="Regular Follow-up Monitoring",
                description=f"Schedule {timeframe.lower()} follow-up appointments to monitor condition and treatment response",
                category="followup",
                priority="high" if overall_risk >= 7.0 else "medium",
                rationale="Regular monitoring essential for managing identified risk factors",
                timeframe=timeframe
            ))

        # Specialist referral recommendations
        oncology_entities = [e for e in entities if any(k in e.text.lower() for k in ['cancer', 'tumor', 'malignancy', 'carcinoma'])]
        if oncology_entities and not any(e.negated for e in oncology_entities):
            recommendations.append(Recommendation(
                title="Oncology Consultation",
                description="Refer to oncology for evaluation and management of identified malignancy concerns",
                category="specialist_referral",
                priority="critical",
                rationale="Oncological findings require specialized assessment",
                timeframe="Within 1-2 weeks"
            ))

        return recommendations

    def _assess_medication_risks(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Assess medication-related risks"""
        med_entities = [e for e in entities if e.label == 'MEDICATION']
        
        if not med_entities:
            return {'total_medications': 0, 'high_risk_medications': [], 'polypharmacy_risk': 'Low'}
        
        high_risk_meds = []
        for med in med_entities:
            med_text = med.text.lower()
            for risk_med, risk_score in self.high_risk_medications.items():
                if risk_med in med_text:
                    high_risk_meds.append({
                        'medication': med.text,
                        'risk_category': risk_med,
                        'risk_score': risk_score
                    })
        
        polypharmacy_risk = 'Low'
        if len(med_entities) >= 10:
            polypharmacy_risk = 'High'
        elif len(med_entities) >= 5:
            polypharmacy_risk = 'Moderate'
        
        return {
            'total_medications': len(med_entities),
            'high_risk_medications': high_risk_meds,
            'polypharmacy_risk': polypharmacy_risk,
            'interaction_warning': len(high_risk_meds) > 0
        }

    def _identify_critical_symptoms(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        """Identify critical symptoms requiring immediate attention"""
        critical_findings = []
        
        for entity in entities:
            if entity.label == 'SYMPTOM' and not entity.negated:
                entity_text = entity.text.lower()
                for symptom, severity in self.critical_symptoms.items():
                    if symptom in entity_text:
                        critical_findings.append({
                            'symptom': entity.text,
                            'severity_score': severity,
                            'urgency': 'Critical' if severity > 85 else 'High',
                            'confidence': entity.confidence
                        })
        
        return sorted(critical_findings, key=lambda x: x['severity_score'], reverse=True)

    def _calculate_risk_stratification(self, risk_score: float) -> Dict[str, Any]:
        """Calculate risk stratification category"""
        if risk_score >= 8.0:
            category = 'Critical Risk'
            description = 'Immediate intervention required'
            color = '#DC2626'
        elif risk_score >= 6.0:
            category = 'High Risk'
            description = 'Urgent medical attention needed'
            color = '#EA580C'
        elif risk_score >= 4.0:
            category = 'Moderate Risk'
            description = 'Close monitoring recommended'
            color = '#F59E0B'
        elif risk_score >= 2.0:
            category = 'Low-Moderate Risk'
            description = 'Regular follow-up advised'
            color = '#10B981'
        else:
            category = 'Low Risk'
            description = 'Routine care appropriate'
            color = '#059669'
        
        return {
            'category': category,
            'description': description,
            'color': color,
            'action_required': risk_score >= 6.0
        }

    def _generate_clinical_summary(self, entities: List[MedicalEntity], risk_score: float) -> str:
        """Generate a clinical summary narrative"""
        conditions = [e for e in entities if e.label in ['DISEASE', 'CONDITION'] and not e.negated]
        symptoms = [e for e in entities if e.label == 'SYMPTOM' and not e.negated]
        medications = [e for e in entities if e.label == 'MEDICATION']
        
        summary_parts = []
        
        if risk_score >= 7.0:
            summary_parts.append(f"HIGH RISK patient with overall risk score of {risk_score}/10.")
        elif risk_score >= 4.0:
            summary_parts.append(f"MODERATE RISK patient with overall risk score of {risk_score}/10.")
        else:
            summary_parts.append(f"LOW RISK patient with overall risk score of {risk_score}/10.")
        
        if conditions:
            cond_names = [c.text for c in conditions[:5]]
            summary_parts.append(f"Diagnosed conditions include: {', '.join(cond_names)}.")
        
        if symptoms:
            summary_parts.append(f"Presenting with {len(symptoms)} documented symptoms.")
        
        if medications:
            summary_parts.append(f"Currently on {len(medications)} medications.")
        
        return ' '.join(summary_parts)

    def _calculate_summary_stats(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        if not entities:
            return {
                'total_entities': 0,
                'unique_diseases': 0,
                'unique_medications': 0,
                'high_confidence_entities': 0,
                'critical_findings': 0,
                'negated_findings': 0
            }

        diseases = set()
        medications = set()
        critical_count = 0
        negated_count = 0

        for entity in entities:
            if entity.label in ['DISEASE', 'CONDITION', 'DIAGNOSIS']:
                diseases.add(self._norm_text_lower(entity))
            elif entity.label == 'MEDICATION':
                medications.add(self._norm_text_lower(entity))
            
            if entity.negated:
                negated_count += 1
            
            # Check if critical
            if self._is_high_risk_entity(entity):
                critical_count += 1

        high_confidence = sum(1 for e in entities if e.confidence >= 0.8)

        return {
            'total_entities': len(entities),
            'unique_diseases': len(diseases),
            'unique_medications': len(medications),
            'high_confidence_entities': high_confidence,
            'critical_findings': critical_count,
            'negated_findings': negated_count,
            'confidence_rate': (high_confidence / len(entities) * 100) if entities else 0
        }

    def _analyze_entity_patterns(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Analyze patterns in entity extraction with clinical insights"""
        if not entities:
            return {
                'negated_ratio': 0.0,
                'uncertain_ratio': 0.0,
                'conditional_ratio': 0.0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'entity_type_distribution': {},
                'severity_distribution': {},
                'temporality_distribution': {},
                'subject_distribution': {},
                'average_confidence': 0.0
            }

        # Negation and context analysis
        negated_count = sum(1 for e in entities if e.negated)
        negated_ratio = negated_count / len(entities)
        uncertain_count = sum(1 for e in entities if getattr(e, 'uncertain', False))
        conditional_count = sum(1 for e in entities if getattr(e, 'conditional', False))
        uncertain_ratio = uncertain_count / len(entities)
        conditional_ratio = conditional_count / len(entities)

        # Confidence distribution
        high_conf = sum(1 for e in entities if e.confidence >= 0.8)
        medium_conf = sum(1 for e in entities if 0.5 <= e.confidence < 0.8)
        low_conf = sum(1 for e in entities if e.confidence < 0.5)

        total = len(entities)
        confidence_distribution = {
            'high': round((high_conf / total) * 100, 1) if total > 0 else 0,
            'medium': round((medium_conf / total) * 100, 1) if total > 0 else 0,
            'low': round((low_conf / total) * 100, 1) if total > 0 else 0
        }

        # Entity type distribution
        entity_types = Counter(e.label for e in entities)
        entity_type_distribution = {k: v for k, v in entity_types.most_common()}

        # Severity distribution
        severity_counts = {'severe': 0, 'moderate': 0, 'mild': 0, 'unspecified': 0}
        for entity in entities:
            sev = entity.severity if hasattr(entity, 'severity') and entity.severity else 'unspecified'
            if sev in severity_counts:
                severity_counts[sev] += 1
            else:
                severity_counts['unspecified'] += 1

        # Temporality and subject distributions
        temporality_counts = defaultdict(int)
        subject_counts = defaultdict(int)
        for e in entities:
            temporality = (getattr(e, 'temporality', 'current') or 'current').lower()
            subject = (getattr(e, 'subject', 'patient') or 'patient').lower()
            temporality_counts[temporality] += 1
            subject_counts[subject] += 1

        return {
            'negated_ratio': round(negated_ratio, 3),
            'uncertain_ratio': round(uncertain_ratio, 3),
            'conditional_ratio': round(conditional_ratio, 3),
            'confidence_distribution': confidence_distribution,
            'entity_type_distribution': entity_type_distribution,
            'severity_distribution': severity_counts,
            'temporality_distribution': dict(temporality_counts),
            'subject_distribution': dict(subject_counts),
            'average_confidence': round(sum(e.confidence for e in entities) / len(entities), 3)
        }

    def _calculate_comorbidity_multiplier(self, conditions: List[str]) -> float:
        """Calculate risk multiplier based on comorbidity interactions"""
        multiplier = 1.0
        condition_set = set(conditions)
        
        for (cond1, cond2), mult in self.comorbidity_multipliers.items():
            if cond1 in condition_set and cond2 in condition_set:
                multiplier *= mult
        
        return multiplier

    def _get_risk_level_detailed(self, score: float) -> str:
        """Convert numeric score to detailed risk level"""
        if score >= 80:
            return 'critical'
        elif score >= 60:
            return 'high'
        elif score >= 35:
            return 'moderate'
        else:
            return 'low'

    def _norm_text_lower(self, entity: MedicalEntity) -> str:
        """Safely return a lowercase normalized text for an entity."""
        val = getattr(entity, 'normalized_form', None)
        if isinstance(val, str) and val:
            return val.lower()
        # Some extractors may store a dict with {'canonical': ...}
        if isinstance(val, dict):
            can = val.get('canonical')
            if isinstance(can, str) and can:
                return can.lower()
        txt = getattr(entity, 'text', '') or ''
        return txt.lower()

    def _get_entity_severity_score(self, entity: MedicalEntity) -> float:
        """Calculate severity score for an entity"""
        base_score = 1.0
        
        if hasattr(entity, 'severity') and entity.severity:
            severity_map = {'severe': 1.5, 'moderate': 1.2, 'mild': 0.8}
            base_score *= severity_map.get(entity.severity, 1.0)
        
        if hasattr(entity, 'certainty') and entity.certainty:
            certainty_map = {'definite': 1.3, 'probable': 1.1, 'possible': 0.9}
            base_score *= certainty_map.get(entity.certainty, 1.0)
        
        # Check if it's a high-risk condition
        entity_text = self._norm_text_lower(entity)
        if entity_text in self.high_risk_conditions:
            condition_score = self.high_risk_conditions[entity_text]['score'] / 100
            base_score *= condition_score
        
        return base_score

    def _is_high_risk_entity(self, entity: MedicalEntity) -> bool:
        """Determine if entity represents a high-risk finding"""
        entity_text = self._norm_text_lower(entity)
        
        # Check against critical symptoms and conditions
        if entity.label == 'SYMPTOM':
            for symptom in self.critical_symptoms.keys():
                if symptom in entity_text and not entity.negated:
                    return True
        
        # Check high-risk conditions
        if entity_text in self.high_risk_conditions:
            if self.high_risk_conditions[entity_text]['score'] >= 75:
                return True

        return False

    def _get_severity_multiplier(self, entity: MedicalEntity) -> float:
        """Get severity multiplier based on entity attributes"""
        multiplier = 1.0

        if hasattr(entity, 'severity') and entity.severity:
            severity_map = {
                'severe': 1.5,
                'moderate': 1.2,
                'mild': 0.8,
                'critical': 1.8
            }
            multiplier *= severity_map.get(entity.severity, 1.0)

        if hasattr(entity, 'certainty') and entity.certainty:
            certainty_map = {
                'definite': 1.3,
                'probable': 1.1,
                'possible': 0.8,
                'uncertain': 0.6
            }
            multiplier *= certainty_map.get(entity.certainty, 1.0)

        return multiplier

    def _get_condition_risk_score(self, entity: MedicalEntity) -> float:
        """Get base risk score for specific conditions"""
        if entity.label in ['DISEASE', 'CONDITION', 'DIAGNOSIS']:
            condition_name = (entity.normalized_form or entity.text).lower()
            if condition_name in self.high_risk_conditions:
                return self.high_risk_conditions[condition_name]['score'] * 0.1
        return 0

    def _get_symptom_risk_score(self, entity: MedicalEntity) -> float:
        """Get risk score for critical symptoms"""
        if entity.label == 'SYMPTOM' and not entity.negated:
            entity_text = entity.text.lower()
            for symptom, severity in self.critical_symptoms.items():
                if symptom in entity_text:
                    return severity * 0.01 * entity.confidence
        return 0

    def _get_medication_risk_score(self, entity: MedicalEntity) -> float:
        """Get risk score for high-risk medications"""
        if entity.label == 'MEDICATION':
            entity_text = entity.text.lower()
            for med, risk_score in self.high_risk_medications.items():
                if med in entity_text:
                    return risk_score * entity.confidence
        return 0

    def _get_normal_indicator_reduction(self, entity: MedicalEntity) -> float:
        """Get risk score reduction for normal/healthy indicators"""
        entity_text = entity.text.lower()

        max_reduction = 0.0
        for indicator, reduction_value in self.normal_indicators.items():
            if indicator in entity_text:
                max_reduction = max(max_reduction, reduction_value)

        return max_reduction * entity.confidence

    def _determine_trend(self, entities: List[MedicalEntity]) -> str:
        """Determine risk trend based on entity characteristics"""
        chronic_indicators = ['chronic', 'long-term', 'persistent', 'ongoing', 'longstanding']
        acute_indicators = ['acute', 'sudden', 'recent', 'new', 'emergency']
        improving_indicators = ['improved', 'improving', 'resolved', 'better', 'stable']

        chronic_count = sum(1 for e in entities if any(ind in e.text.lower() for ind in chronic_indicators))
        acute_count = sum(1 for e in entities if any(ind in e.text.lower() for ind in acute_indicators))
        improving_count = sum(1 for e in entities if any(ind in e.text.lower() for ind in improving_indicators))

        if improving_count > max(chronic_count, acute_count):
            return 'decreasing'
        elif acute_count > chronic_count:
            return 'increasing'
        else:
            return 'stable'

    def _determine_condition_severity(self, entities: List[MedicalEntity]) -> str:
        """Determine overall severity of a condition based on its mentions"""
        if not entities:
            return 'Unknown'
        
        severity_scores = []
        for entity in entities:
            if hasattr(entity, 'severity') and entity.severity:
                severity_map = {'severe': 3, 'moderate': 2, 'mild': 1}
                severity_scores.append(severity_map.get(entity.severity, 1))
            else:
                severity_scores.append(1)
        
        avg_severity = sum(severity_scores) / len(severity_scores)
        
        if avg_severity >= 2.5:
            return 'Severe'
        elif avg_severity >= 1.5:
            return 'Moderate'
        else:
            return 'Mild'

    def _assess_clinical_significance(self, condition_info: Dict, frequency: int) -> str:
        """Assess clinical significance of a condition"""
        score = condition_info.get('score', 50)
        urgency = condition_info.get('urgency', 'moderate')
        
        if score >= 85 or urgency == 'critical':
            return 'Critical - Requires immediate attention'
        elif score >= 70 or urgency == 'high':
            return 'High - Requires prompt evaluation'
        elif frequency > 2:
            return 'Significant - Multiple occurrences noted'
        elif score >= 50:
            return 'Moderate - Warrants clinical monitoring'
        else:
            return 'Low - Routine follow-up appropriate'

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'overall_risk_score': 0,
            'risk_stratification': {
                'category': 'No Data',
                'description': 'Insufficient data for risk assessment',
                'color': '#9CA3AF',
                'action_required': False
            },
            'risk_assessment': [],
            'common_conditions': [],
            'comorbidity_clusters': [],
            'recommendations': [],
            'summary_stats': {
                'total_entities': 0,
                'unique_diseases': 0,
                'unique_medications': 0,
                'high_confidence_entities': 0,
                'critical_findings': 0,
                'negated_findings': 0
            },
            'entity_patterns': {
                'negated_ratio': 0.0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'entity_type_distribution': {},
                'severity_distribution': {}
            },
            'medication_risks': {
                'total_medications': 0,
                'high_risk_medications': [],
                'polypharmacy_risk': 'Low'
            },
            'critical_symptoms': [],
            'clinical_summary': 'No clinical data available for analysis.'
        }

    def generate_graphs_from_analysis(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate enhanced clinical graphs from analysis data

        Args:
            analysis: The comprehensive risk analysis dictionary

        Returns:
            Dictionary with graph names and base64-encoded PNG images
        """
        graphs = {}

        # Set professional medical style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        # Graph 1: Enhanced Risk Assessment
        risk_factors = analysis.get('risk_assessment', [])
        if risk_factors:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            names = [rf.name.replace(' Risk', '') for rf in risk_factors]
            percentages = [rf.percentage for rf in risk_factors]
            colors = [self._get_risk_color(rf.risk_level) for rf in risk_factors]

            bars = ax.barh(names, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Risk Percentage (%)', fontsize=11, fontweight='bold')
            ax.set_title('Clinical Risk Factor Assessment', fontsize=13, fontweight='bold', pad=15)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # Add value labels
            for bar, percentage, rf in zip(bars, percentages, risk_factors):
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                        f'{percentage:.1f}%', va='center', fontsize=9, fontweight='bold')
            
            # Add risk level indicators
            for i, rf in enumerate(risk_factors):
                ax.text(-5, i, f'●', color=self._get_risk_color(rf.risk_level), 
                       fontsize=16, ha='right', va='center')

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            graphs['risk_assessment'] = base64.b64encode(buf.read()).decode('utf-8')

        # Graph 2: Common Conditions
        common_conditions = analysis.get('common_conditions', [])
        if common_conditions:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            names = [cc.name[:30] + '...' if len(cc.name) > 30 else cc.name for cc in common_conditions[:8]]
            cases = [cc.cases for cc in common_conditions[:8]]
            colors = [self._get_risk_color(cc.risk_level) for cc in common_conditions[:8]]

            bars = ax.barh(names, cases, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Most Common Medical Conditions', fontsize=13, fontweight='bold', pad=15)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # Add value labels
            for bar, case_count in zip(bars, cases):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{case_count}', va='center', fontsize=9, fontweight='bold')

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            graphs['common_conditions'] = base64.b64encode(buf.read()).decode('utf-8')

        # Graph 3: Risk Score Gauge
        overall_score = analysis.get('overall_risk_score', 0)
        if overall_score > 0:
            fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
            
            # Create gauge chart
            theta = np.linspace(0, np.pi, 100)
            radii = np.ones(100)
            
            # Color zones
            colors_zones = ['#059669', '#10B981', '#F59E0B', '#EA580C', '#DC2626']
            zone_boundaries = [0, 2, 4, 6, 8, 10]
            
            for i in range(len(zone_boundaries)-1):
                start = zone_boundaries[i] / 10 * np.pi
                end = zone_boundaries[i+1] / 10 * np.pi
                theta_zone = np.linspace(start, end, 20)
                ax.fill_between(theta_zone, 0, 1, color=colors_zones[i], alpha=0.3)
            
            # Plot needle
            needle_angle = (overall_score / 10) * np.pi
            ax.plot([needle_angle, needle_angle], [0, 0.8], 'k-', linewidth=3)
            ax.plot(needle_angle, 0.8, 'ko', markersize=10)
            
            ax.set_ylim(0, 1)
            ax.set_xlim(0, np.pi)
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['0', '2.5', '5.0', '7.5', '10'])
            ax.set_yticks([])
            ax.set_title(f'Overall Risk Score: {overall_score}/10', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            graphs['risk_gauge'] = base64.b64encode(buf.read()).decode('utf-8')

        return graphs

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        color_map = {
            'critical': '#DC2626',
            'high': '#EA580C',
            'moderate': '#F59E0B',
            'low': '#10B981'
        }
        return color_map.get(risk_level, '#6B7280')

# Utility functions for HTML generation
def format_risk_level_color(risk_level: str) -> str:
    """Format risk level with appropriate color"""
    colors = {
        'critical': '#DC2626',
        'high': '#EA580C',
        'moderate': '#F59E0B',
        'low': '#10B981'
    }
    return colors.get(risk_level, '#6B7280')

def format_trend_icon(trend: str) -> str:
    """Format trend with appropriate icon"""
    icons = {
        'increasing': '📈',
        'stable': '➡️',
        'decreasing': '📉'
    }
    return icons.get(trend, '➡️')

def create_risk_progress_bar_html(risk_factor: RiskFactor) -> str:
    """Create HTML for risk progress bar"""
    color = format_risk_level_color(risk_factor.risk_level)
    icon = format_trend_icon(risk_factor.trend)

    return f"""
    <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #1f2937; font-weight: 600;">{risk_factor.name}</h4>
            <span style="color: {color}; font-weight: 700; font-size: 18px;">{risk_factor.percentage}%</span>
        </div>
        <div style="width: 100%; height: 10px; background: #e2e8f0; border-radius: 5px; margin-bottom: 0.5rem; overflow: hidden;">
            <div style="width: {risk_factor.percentage}%; height: 100%; background: {color}; border-radius: 5px; transition: width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
            <small style="color: #6b7280;">{risk_factor.description}</small>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <span style="background: {color}20; color: {color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase;">{risk_factor.risk_level}</span>
            </div>
        </div>
    </div>
    """

def create_condition_card_html(condition: CommonCondition) -> str:
    """Create HTML for condition card"""
    color = format_risk_level_color(condition.risk_level)

    return f"""
    <div style="margin: 0.5rem; padding: 1.2rem; border: 1px solid #e2e8f0; border-radius: 10px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h5 style="margin: 0 0 0.8rem 0; color: #1f2937; font-weight: 600; font-size: 15px;">{condition.name}</h5>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: #6b7280; font-size: 13px;">Occurrences: <strong>{condition.cases}</strong></span>
            <span style="background: {color}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;">{condition.risk_level}</span>
        </div>
        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e2e8f0;">
            <small style="color: #6b7280; font-size: 12px;"><strong>Severity:</strong> {condition.severity}</small>
        </div>
    </div>
    """

def create_recommendation_card_html(recommendation: Recommendation) -> str:
    """Create HTML for recommendation card"""
    priority_colors = {
        'critical': '#DC2626',
        'high': '#EA580C',
        'medium': '#F59E0B',
        'low': '#10B981'
    }
    color = priority_colors.get(recommendation.priority, '#6B7280')

    return f"""
    <div style="margin-bottom: 1.2rem; padding: 1.2rem; border: 1px solid #e2e8f0; border-left: 4px solid {color}; border-radius: 8px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <h4 style="margin: 0; color: #1f2937; font-weight: 600; font-size: 16px;">{recommendation.title}</h4>
            <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;">{recommendation.priority}</span>
        </div>
        <p style="margin: 0 0 0.8rem 0; color: #4b5563; font-size: 14px; line-height: 1.6;">{recommendation.description}</p>
        <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid #e2e8f0;">
            <small style="color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px; font-weight: 500;">📋 {recommendation.category}</small>
            <small style="color: #6b7280; font-size: 12px;">⏰ {recommendation.timeframe}</small>
        </div>
    </div>
    """