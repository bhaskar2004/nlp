"""
Medical Risk Analytics Module
Provides comprehensive risk assessment and analysis for medical entities
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from medical_nlp import MedicalEntity

# Add matplotlib imports for graph generation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

@dataclass
class RiskFactor:
    """Represents a risk factor with its assessment"""
    name: str
    percentage: float
    risk_level: str  # 'low', 'medium', 'high'
    trend: str  # 'increasing', 'stable', 'decreasing'
    description: str

@dataclass
class CommonCondition:
    """Represents a commonly identified condition"""
    name: str
    cases: int
    risk_level: str

@dataclass
class Recommendation:
    """Represents a clinical recommendation"""
    title: str
    description: str
    category: str
    priority: str  # 'high', 'medium', 'low'

class MedicalRiskAnalyzer:
    """
    Advanced medical risk analyzer that assesses patient risk based on extracted entities
    """

    def __init__(self):
        # Risk scoring weights for different entity types
        self.risk_weights = {
            'DISEASE': 0.8,
            'SYMPTOM': 0.6,
            'MEDICATION': 0.4,
            'TEST': 0.3,
            'PROCEDURE': 0.5,
            'BODY_PART': 0.2
        }

        # High-risk conditions and their base scores
        self.high_risk_conditions = {
            'myocardial infarction': 90,
            'heart failure': 85,
            'stroke': 88,
            'cancer': 95,
            'pneumonia': 75,
            'sepsis': 92,
            'diabetes': 70,
            'hypertension': 65,
            'copd': 78,
            'kidney disease': 80
        }

        # Risk factor definitions - concerning keywords
        self.risk_factor_definitions = {
            'Cardiovascular Risk': ['myocardial infarction', 'heart failure', 'hypertension', 'cholesterol', 'arrhythmia', 'angina'],
            'Respiratory Risk': ['pneumonia', 'copd', 'asthma', 'respiratory failure', 'pulmonary embolism'],
            'Metabolic Risk': ['diabetes', 'hyperglycemia', 'insulin resistance', 'obesity', 'metabolic syndrome'],
            'Neurological Risk': ['stroke', 'seizure', 'neurological deficit', 'headache', 'migraine'],
            'Oncological Risk': ['cancer', 'tumor', 'malignancy', 'carcinoma', 'metastasis']
        }

        # Normal/healthy indicators that reduce risk
        self.normal_indicators = {
            'normal': 0.3,
            'stable': 0.4,
            'unremarkable': 0.5,
            'within normal range': 0.6,
            'no acute distress': 0.5,
            'clear': 0.4,
            'regular': 0.3,
            'healthy': 0.7,
            'good general health': 0.8,
            'balanced diet': 0.4,
            'regular physical activity': 0.5,
            'optimal range': 0.6,
            'no evidence of': 0.5
        }

    def generate_comprehensive_analysis(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """
        Generate comprehensive risk analysis from medical entities

        Args:
            entities: List of extracted medical entities

        Returns:
            Dictionary containing comprehensive risk analysis
        """
        if not entities:
            return self._empty_analysis()

        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(entities)

        # Generate risk assessment factors
        risk_assessment = self._generate_risk_assessment(entities)

        # Identify common conditions
        common_conditions = self._identify_common_conditions(entities)

        # Generate clinical recommendations
        recommendations = self._generate_recommendations(entities, overall_risk_score)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(entities)

        # Analyze entity patterns
        entity_patterns = self._analyze_entity_patterns(entities)

        return {
            'overall_risk_score': overall_risk_score,
            'risk_assessment': risk_assessment,
            'common_conditions': common_conditions,
            'recommendations': recommendations,
            'summary_stats': summary_stats,
            'entity_patterns': entity_patterns
        }

    def analyze(self, entities: List[MedicalEntity]) -> List[Dict[str, Any]]:
        """
        Simplified analysis method for basic risk assessment

        Args:
            entities: List of extracted medical entities

        Returns:
            List of risk analysis results
        """
        if not entities:
            return []

        results = []

        # Group entities by type
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity.label] += 1

        # Calculate risk scores for each category
        for entity_type, count in entity_counts.items():
            risk_score = min(100, count * self.risk_weights.get(entity_type, 0.5) * 20)

            results.append({
                'name': f'{entity_type} Risk',
                'level': self._get_risk_level(risk_score),
                'score': risk_score,
                'factors': [f'{count} {entity_type.lower()} entities detected']
            })

        return results

    def _calculate_overall_risk_score(self, entities: List[MedicalEntity]) -> float:
        """Calculate overall risk score from entities"""
        if not entities:
            return 0.0

        total_score = 0.0
        normal_score_reduction = 0.0
        max_possible_score = 0.0

        for entity in entities:
            # Base score from entity type
            base_score = self.risk_weights.get(entity.label, 0.5)

            # Adjust for confidence
            confidence_multiplier = entity.confidence

            # Adjust for negation (negated entities reduce risk)
            negation_multiplier = 0.3 if entity.negated else 1.0

            # Adjust for severity indicators
            severity_multiplier = self._get_severity_multiplier(entity)

            # Check for high-risk conditions
            condition_score = self._get_condition_risk_score(entity)

            # Check for normal/healthy indicators that reduce risk
            normal_reduction = self._get_normal_indicator_reduction(entity)

            entity_score = (base_score * confidence_multiplier * negation_multiplier * severity_multiplier) + condition_score
            total_score += entity_score
            normal_score_reduction += normal_reduction
            max_possible_score += 1.0

        # Apply normal indicator reductions
        adjusted_score = max(0, total_score - normal_score_reduction)

        # Normalize to 0-100 scale
        if max_possible_score > 0:
            normalized_score = (adjusted_score / max_possible_score) * 100
            return min(100.0, max(0.0, normalized_score))
        else:
            return 0.0

    def _generate_risk_assessment(self, entities: List[MedicalEntity]) -> List[RiskFactor]:
        """Generate detailed risk factor assessment"""
        risk_factors = []

        for factor_name, keywords in self.risk_factor_definitions.items():
            # Count relevant entities
            relevant_entities = []
            for entity in entities:
                entity_text = entity.text.lower()
                if any(keyword in entity_text for keyword in keywords):
                    relevant_entities.append(entity)

            if relevant_entities:
                # Calculate risk percentage
                base_score = len(relevant_entities) * 15
                confidence_avg = sum(e.confidence for e in relevant_entities) / len(relevant_entities)
                percentage = min(100, base_score * confidence_avg)

                risk_level = self._get_risk_level(percentage)
                trend = self._determine_trend(relevant_entities)

                description = f"Found {len(relevant_entities)} relevant entities with average confidence {confidence_avg:.2f}"

                risk_factors.append(RiskFactor(
                    name=factor_name,
                    percentage=round(percentage, 1),
                    risk_level=risk_level,
                    trend=trend,
                    description=description
                ))

        # Sort by percentage descending
        risk_factors.sort(key=lambda x: x.percentage, reverse=True)

        return risk_factors

    def _identify_common_conditions(self, entities: List[MedicalEntity]) -> List[CommonCondition]:
        """Identify commonly occurring medical conditions"""
        condition_counts = defaultdict(int)

        for entity in entities:
            if entity.label == 'DISEASE':
                condition_name = entity.normalized_form or entity.text.lower()
                condition_counts[condition_name] += 1

        common_conditions = []
        for condition, count in condition_counts.items():
            if count >= 1:  # Include conditions that appear at least once
                risk_score = self.high_risk_conditions.get(condition, 50)
                risk_level = self._get_risk_level(risk_score)

                common_conditions.append(CommonCondition(
                    name=condition.title(),
                    cases=count,
                    risk_level=risk_level
                ))

        # Sort by case count descending
        common_conditions.sort(key=lambda x: x.cases, reverse=True)

        return common_conditions[:10]  # Return top 10

    def _generate_recommendations(self, entities: List[MedicalEntity], overall_risk: float) -> List[Recommendation]:
        """Generate clinical recommendations based on entities and risk score"""
        recommendations = []

        # High-risk recommendations
        if overall_risk >= 70:
            recommendations.extend([
                Recommendation(
                    title="Immediate Clinical Review",
                    description="Patient shows high-risk indicators requiring immediate clinical assessment",
                    category="urgent",
                    priority="high"
                ),
                Recommendation(
                    title="Specialist Consultation",
                    description="Consider referral to appropriate specialist based on identified conditions",
                    category="referral",
                    priority="high"
                )
            ])

        # Cardiovascular recommendations
        cardio_entities = [e for e in entities if 'heart' in e.text.lower() or 'cardiac' in e.text.lower()]
        if cardio_entities:
            recommendations.append(Recommendation(
                title="Cardiovascular Assessment",
                description="Perform comprehensive cardiovascular evaluation including ECG and cardiac markers",
                category="diagnostic",
                priority="high" if len(cardio_entities) > 2 else "medium"
            ))

        # Medication review
        medication_entities = [e for e in entities if e.label == 'MEDICATION']
        if medication_entities:
            recommendations.append(Recommendation(
                title="Medication Review",
                description="Review current medications for interactions, adherence, and therapeutic effectiveness",
                category="medication",
                priority="medium"
            ))

        # Lifestyle recommendations
        if any(e.label == 'DISEASE' for e in entities):
            recommendations.append(Recommendation(
                title="Lifestyle Modification",
                description="Provide counseling on diet, exercise, and lifestyle modifications",
                category="preventive",
                priority="medium"
            ))

        # Follow-up recommendations
        recommendations.append(Recommendation(
            title="Regular Follow-up",
            description="Schedule appropriate follow-up based on risk assessment and clinical findings",
            category="followup",
            priority="low"
        ))

        return recommendations

    def _calculate_summary_stats(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Calculate summary statistics from entities"""
        if not entities:
            return {
                'total_entities': 0,
                'unique_diseases': 0,
                'unique_medications': 0,
                'high_confidence_entities': 0
            }

        # Count unique diseases and medications
        diseases = set()
        medications = set()

        for entity in entities:
            if entity.label == 'DISEASE':
                diseases.add(entity.normalized_form or entity.text.lower())
            elif entity.label == 'MEDICATION':
                medications.add(entity.normalized_form or entity.text.lower())

        # Count high confidence entities
        high_confidence = sum(1 for e in entities if e.confidence >= 0.8)

        return {
            'total_entities': len(entities),
            'unique_diseases': len(diseases),
            'unique_medications': len(medications),
            'high_confidence_entities': high_confidence
        }

    def _analyze_entity_patterns(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Analyze patterns in entity extraction"""
        if not entities:
            return {
                'negated_ratio': 0.0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }

        # Calculate negation ratio
        negated_count = sum(1 for e in entities if e.negated)
        negated_ratio = negated_count / len(entities)

        # Confidence distribution
        high_conf = sum(1 for e in entities if e.confidence >= 0.8)
        medium_conf = sum(1 for e in entities if 0.5 <= e.confidence < 0.8)
        low_conf = sum(1 for e in entities if e.confidence < 0.5)

        total = len(entities)
        confidence_distribution = {
            'high': (high_conf / total) * 100 if total > 0 else 0,
            'medium': (medium_conf / total) * 100 if total > 0 else 0,
            'low': (low_conf / total) * 100 if total > 0 else 0
        }

        return {
            'negated_ratio': negated_ratio,
            'confidence_distribution': confidence_distribution
        }

    def _get_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level"""
        if score >= 70:
            return 'high'
        elif score >= 40:
            return 'medium'
        else:
            return 'low'

    def _get_severity_multiplier(self, entity: MedicalEntity) -> float:
        """Get severity multiplier based on entity attributes"""
        multiplier = 1.0

        if entity.severity:
            if entity.severity == 'severe':
                multiplier = 1.5
            elif entity.severity == 'moderate':
                multiplier = 1.2
            elif entity.severity == 'mild':
                multiplier = 0.8

        if entity.certainty:
            if entity.certainty == 'definite':
                multiplier *= 1.2
            elif entity.certainty == 'possible':
                multiplier *= 0.8

        return multiplier

    def _get_condition_risk_score(self, entity: MedicalEntity) -> float:
        """Get base risk score for specific conditions"""
        if entity.label == 'DISEASE':
            condition_name = entity.normalized_form or entity.text.lower()
            return self.high_risk_conditions.get(condition_name, 0) * 0.1
        return 0

    def _get_normal_indicator_reduction(self, entity: MedicalEntity) -> float:
        """Get risk score reduction for normal/healthy indicators"""
        entity_text = entity.text.lower()

        # Check for normal indicators in the entity text
        max_reduction = 0.0
        for indicator, reduction_value in self.normal_indicators.items():
            if indicator in entity_text:
                max_reduction = max(max_reduction, reduction_value)

        # Apply confidence multiplier to the reduction
        return max_reduction * entity.confidence

    def _determine_trend(self, entities: List[MedicalEntity]) -> str:
        """Determine risk trend based on entity characteristics"""
        # Simple heuristic: more recent/chronic conditions suggest stable/increasing
        chronic_indicators = ['chronic', 'long-term', 'persistent', 'ongoing']
        acute_indicators = ['acute', 'sudden', 'recent', 'new']

        chronic_count = sum(1 for e in entities if any(ind in e.text.lower() for ind in chronic_indicators))
        acute_count = sum(1 for e in entities if any(ind in e.text.lower() for ind in acute_indicators))

        if chronic_count > acute_count:
            return 'stable'
        elif acute_count > chronic_count:
            return 'increasing'
        else:
            return 'stable'

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'overall_risk_score': 0,
            'risk_assessment': [],
            'common_conditions': [],
            'recommendations': [],
            'summary_stats': {
                'total_entities': 0,
                'unique_diseases': 0,
                'unique_medications': 0,
                'high_confidence_entities': 0
            },
            'entity_patterns': {
                'negated_ratio': 0.0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
        }

    def generate_graphs_from_analysis(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate graphs (as base64-encoded PNG images) from analysis data.

        Args:
            analysis: The comprehensive risk analysis dictionary.

        Returns:
            Dictionary with keys as graph names and values as base64-encoded PNG images.
        """
        graphs = {}

        # Graph 1: Risk Assessment Bar Chart
        risk_factors = analysis.get('risk_assessment', [])
        if risk_factors:
            names = [rf.name for rf in risk_factors]
            percentages = [rf.percentage for rf in risk_factors]
            colors = [format_risk_level_color(rf.risk_level) for rf in risk_factors]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(names, percentages, color=colors)
            ax.set_xlabel('Risk Percentage')
            ax.set_title('Risk Assessment Factors')
            ax.invert_yaxis()  # Highest risk on top
            ax.set_xlim(0, 100)

            for bar, percentage in zip(bars, percentages):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{percentage}%', va='center')

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            graphs['risk_assessment'] = base64.b64encode(buf.read()).decode('utf-8')

        # Graph 2: Common Conditions Bar Chart
        common_conditions = analysis.get('common_conditions', [])
        if common_conditions:
            names = [cc.name for cc in common_conditions]
            cases = [cc.cases for cc in common_conditions]
            colors = [format_risk_level_color(cc.risk_level) for cc in common_conditions]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(names, cases, color=colors)
            ax.set_xlabel('Number of Cases')
            ax.set_title('Common Medical Conditions')
            ax.invert_yaxis()
            max_cases = max(cases) if cases else 1
            ax.set_xlim(0, max_cases + 1)

            for bar, case_count in zip(bars, cases):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{case_count}', va='center')

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            graphs['common_conditions'] = base64.b64encode(buf.read()).decode('utf-8')

        return graphs

# Utility functions for HTML generation (used in Streamlit UI)
def format_risk_level_color(risk_level: str) -> str:
    """Format risk level with appropriate color"""
    colors = {
        'high': '#dc2626',
        'medium': '#ea580c',
        'low': '#059669'
    }
    return colors.get(risk_level, '#6b7280')

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
    <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #1f2937;">{risk_factor.name}</h4>
            <span style="color: {color}; font-weight: 600;">{risk_factor.percentage}%</span>
        </div>
        <div style="width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; margin-bottom: 0.5rem;">
            <div style="width: {risk_factor.percentage}%; height: 100%; background: {color}; border-radius: 4px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <small style="color: #6b7280;">{risk_factor.description}</small>
            <span style="font-size: 1.2rem;">{icon}</span>
        </div>
    </div>
    """

def create_condition_card_html(condition: CommonCondition) -> str:
    """Create HTML for condition card"""
    color = format_risk_level_color(condition.risk_level)

    return f"""
    <div style="display: inline-block; margin: 0.5rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;">
        <h5 style="margin: 0 0 0.5rem 0; color: #1f2937;">{condition.name}</h5>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #6b7280;">Cases: {condition.cases}</span>
            <span style="color: {color}; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">{condition.risk_level}</span>
        </div>
    </div>
    """

def create_recommendation_card_html(recommendation: Recommendation) -> str:
    """Create HTML for recommendation card"""
    priority_colors = {
        'high': '#dc2626',
        'medium': '#ea580c',
        'low': '#059669'
    }
    color = priority_colors.get(recommendation.priority, '#6b7280')

    return f"""
    <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #1f2937;">{recommendation.title}</h4>
            <span style="color: {color}; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">{recommendation.priority}</span>
        </div>
        <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">{recommendation.description}</p>
        <small style="color: #9ca3af; text-transform: uppercase; letter-spacing: 0.025em;">{recommendation.category}</small>
    </div>
    """
