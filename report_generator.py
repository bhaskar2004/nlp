from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, KeepTogether, Image)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from io import BytesIO
import datetime
from typing import List, Dict, Any
from collections import Counter

class NumberedCanvas(canvas.Canvas):
    """Custom canvas for adding page numbers and headers/footers"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        # Footer with page numbers
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#7B8794'))
        self.drawRightString(letter[0] - 0.75*inch, 0.5*inch, 
                            f"Page {self._pageNumber} of {page_count}")
        self.drawString(0.75*inch, 0.5*inch, 
                       "MedNLP Clinical Intelligence Report | CONFIDENTIAL")
        
        # Header line
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor('#E8ECF0'))
            self.setLineWidth(0.5)
            self.line(0.75*inch, letter[1] - 0.6*inch, 
                     letter[0] - 0.75*inch, letter[1] - 0.6*inch)


class MedicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create comprehensive custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7B8794'),
            spaceAfter=25,
            fontName='Helvetica-Oblique'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#0066CC'),
            spaceBefore=18,
            spaceAfter=10,
            fontName='Helvetica-Bold',
            leftIndent=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#1F2933'),
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#3E4C59'),
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#1F2933'),
            backColor=colors.HexColor('#EFF6FF'),
            borderPadding=10,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='WarningBox',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#7C2D12'),
            backColor=colors.HexColor('#FEF2F2'),
            borderPadding=10,
            fontName='Helvetica'
        ))

    def _create_entity_distribution_chart(self, entities):
        """Create a pie chart showing entity type distribution"""
        entity_counts = Counter([e.label for e in entities])
        
        if not entity_counts:
            return None
            
        # Take top 8 categories
        top_entities = entity_counts.most_common(8)
        labels = [item[0].replace('_', ' ') for item in top_entities]
        data = [item[1] for item in top_entities]
        
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 120
        pie.height = 120
        pie.data = data
        pie.labels = labels
        pie.slices.strokeWidth = 0.5
        
        # Color palette
        colors_palette = [
            colors.HexColor('#0066CC'), colors.HexColor('#10B981'),
            colors.HexColor('#F59E0B'), colors.HexColor('#EF4444'),
            colors.HexColor('#8B5CF6'), colors.HexColor('#EC4899'),
            colors.HexColor('#06B6D4'), colors.HexColor('#84CC16')
        ]
        
        for i, color in enumerate(colors_palette[:len(data)]):
            pie.slices[i].fillColor = color
            
        drawing.add(pie)
        return drawing

    def _create_confidence_distribution_chart(self, entities):
        """Create a bar chart showing confidence level distribution"""
        if not entities:
            return None
            
        # Categorize by confidence levels
        high = sum(1 for e in entities if e.confidence >= 0.8)
        medium = sum(1 for e in entities if 0.6 <= e.confidence < 0.8)
        low = sum(1 for e in entities if e.confidence < 0.6)
        
        drawing = Drawing(400, 180)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 100
        bc.width = 300
        bc.data = [[high, medium, low]]
        bc.categoryAxis.categoryNames = ['High\n(≥80%)', 'Medium\n(60-80%)', 'Low\n(<60%)']
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(high, medium, low) + 5
        bc.bars[0].fillColor = colors.HexColor('#10B981')
        bc.bars[1].fillColor = colors.HexColor('#F59E0B')
        bc.bars[2].fillColor = colors.HexColor('#EF4444')
        
        drawing.add(bc)
        return drawing

    def _analyze_clinical_patterns(self, entities):
        """Analyze clinical patterns and extract meaningful insights"""
        patterns = {
            'chronic_conditions': [],
            'acute_symptoms': [],
            'diagnostic_tests': [],
            'medications': [],
            'procedures': [],
            'anatomical_focus': []
        }
        
        chronic_keywords = ['diabetes', 'hypertension', 'copd', 'asthma', 'arthritis', 
                           'heart disease', 'kidney disease', 'chronic']
        acute_keywords = ['acute', 'sudden', 'severe', 'emergency', 'critical']
        
        for e in entities:
            text_lower = e.text.lower()
            
            if e.label in ['DISEASE', 'CONDITION']:
                if any(keyword in text_lower for keyword in chronic_keywords):
                    patterns['chronic_conditions'].append(e.text)
                elif any(keyword in text_lower for keyword in acute_keywords):
                    patterns['acute_symptoms'].append(e.text)
                    
            elif e.label == 'TEST':
                patterns['diagnostic_tests'].append(e.text)
            elif e.label == 'MEDICATION':
                patterns['medications'].append(e.text)
            elif e.label == 'PROCEDURE':
                patterns['procedures'].append(e.text)
            elif e.label == 'ANATOMY':
                patterns['anatomical_focus'].append(e.text)
        
        return patterns

    def _calculate_clinical_complexity_score(self, entities, risk_analysis):
        """Calculate a clinical complexity score based on multiple factors"""
        score_components = {
            'num_conditions': min(len([e for e in entities if e.label in ['DISEASE', 'CONDITION']]) / 5, 3),
            'num_medications': min(len([e for e in entities if e.label == 'MEDICATION']) / 10, 3),
            'num_procedures': min(len([e for e in entities if e.label == 'PROCEDURE']) / 5, 2),
            'risk_score': risk_analysis.get('overall_risk_score', 0) / 10 * 2,
        }
        
        total_score = sum(score_components.values())
        complexity_level = 'Low' if total_score < 3 else 'Moderate' if total_score < 6 else 'High'
        
        return {
            'score': total_score,
            'max_score': 10.0,
            'level': complexity_level,
            'components': score_components
        }

    def _create_executive_summary(self, story, entities, risk_analysis, patterns, complexity):
        """Create an executive summary with key insights"""
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1.5, 
                               color=colors.HexColor('#0066CC'), spaceAfter=15))
        
        # Summary statistics
        summary_data = [
            ['Total Entities Identified', str(len(entities))],
            ['Unique Conditions', str(len(set([e.text for e in entities if e.label in ['DISEASE', 'CONDITION']])))],
            ['Active Medications', str(len(set([e.text for e in entities if e.label == 'MEDICATION'])))],
            ['Clinical Complexity', f"{complexity['level']} ({complexity['score']:.1f}/10)"],
            ['Overall Risk Level', risk_analysis.get('risk_stratification', {}).get('category', 'Unknown').upper()]
        ]
        
        summary_table = Table(summary_data, colWidths=[200, 280])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2933')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8ECF0')),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # Key clinical insights
        story.append(Paragraph("<b>Key Clinical Insights:</b>", self.styles['SubsectionHeader']))
        
        insights = []
        if patterns['chronic_conditions']:
            insights.append(f"• Patient presents with {len(set(patterns['chronic_conditions']))} chronic condition(s) requiring long-term management")
        if patterns['acute_symptoms']:
            insights.append(f"• {len(set(patterns['acute_symptoms']))} acute finding(s) identified requiring immediate attention")
        if patterns['medications']:
            insights.append(f"• Current medication regimen includes {len(set(patterns['medications']))} agent(s) - review for interactions recommended")
        if patterns['diagnostic_tests']:
            insights.append(f"• {len(set(patterns['diagnostic_tests']))} diagnostic test(s) referenced - ensure results are reviewed")
        if complexity['score'] >= 6:
            insights.append(f"• High clinical complexity detected - multidisciplinary care approach recommended")
        
        if not insights:
            insights.append("• Standard complexity case - routine follow-up protocols apply")
        
        for insight in insights[:5]:  # Top 5 insights
            story.append(Paragraph(insight, self.styles['NormalText']))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))

    def _create_header_section(self, story):
        """Create professional header with metadata"""
        story.append(Paragraph("CLINICAL INTELLIGENCE REPORT", self.styles['ReportTitle']))
        story.append(Paragraph(
            "Advanced Medical Natural Language Processing Analysis", 
            self.styles['Subtitle']
        ))
        
        story.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#0066CC'),
            spaceBefore=5,
            spaceAfter=20
        ))
        
        # Metadata
        report_id = f'MED-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        metadata = [
            ['Report ID:', report_id],
            ['Generation Date:', datetime.datetime.now().strftime('%B %d, %Y')],
            ['Generation Time:', datetime.datetime.now().strftime('%I:%M:%S %p %Z')],
            ['Report Version:', 'v2.0'],
            ['Classification:', 'CONFIDENTIAL MEDICAL INFORMATION']
        ]
        
        meta_table = Table(metadata, colWidths=[140, 340])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#3E4C59')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#E8ECF0')),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 25))

    def _create_risk_assessment_section(self, story, risk_analysis, complexity):
        """Enhanced risk assessment with complexity analysis"""
        story.append(Paragraph("COMPREHENSIVE RISK ASSESSMENT", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=15))
        
        risk_score = risk_analysis.get('overall_risk_score', 0)
        risk_strat = risk_analysis.get('risk_stratification', {})
        risk_level = risk_strat.get('category', 'Unknown')
        risk_color = risk_strat.get('color', '#7B8794')
        
        # Risk metrics table
        risk_data = [
            [Paragraph("<b>Risk Metric</b>", self.styles['Normal']), 
             Paragraph("<b>Value</b>", self.styles['Normal']),
             Paragraph("<b>Interpretation</b>", self.styles['Normal'])],
            [Paragraph("Overall Risk Score", self.styles['Normal']), 
             Paragraph(f"<b>{risk_score:.1f}</b>/10.0", 
                      ParagraphStyle('Score', parent=self.styles['Normal'], 
                                   fontSize=12, textColor=colors.HexColor(risk_color))),
             risk_level.upper()],
            [Paragraph("Clinical Complexity", self.styles['Normal']),
             Paragraph(f"<b>{complexity['score']:.1f}</b>/10.0", self.styles['Normal']),
             complexity['level'].upper()],
            [Paragraph("Confidence Level", self.styles['Normal']),
             "High" if risk_score > 0 else "N/A",
             "Based on extracted clinical entities"]
        ]
        
        risk_table = Table(risk_data, colWidths=[160, 140, 180])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8ECF0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.HexColor('#F8FAFC'), colors.white]),
        ]))
        
        story.append(risk_table)
        story.append(Spacer(1, 15))
        
        # Risk interpretation box
        if risk_score >= 7:
            interpretation = f"<b>HIGH RISK ALERT:</b> This patient profile indicates elevated clinical risk (score: {risk_score:.1f}/10). Immediate clinical review and intervention planning recommended. Consider specialist consultation and enhanced monitoring protocols."
            box_style = self.styles['WarningBox']
        elif risk_score >= 4:
            interpretation = f"<b>MODERATE RISK:</b> Patient presents moderate clinical complexity (score: {risk_score:.1f}/10). Regular monitoring and standard care protocols apply. Review care plan for optimization opportunities."
            box_style = self.styles['HighlightBox']
        else:
            interpretation = f"<b>LOW RISK:</b> Patient profile suggests lower clinical risk (score: {risk_score:.1f}/10). Routine follow-up care appropriate. Maintain preventive health measures."
            box_style = self.styles['HighlightBox']
        
        story.append(Paragraph(interpretation, box_style))
        story.append(Spacer(1, 20))

    def _create_clinical_patterns_section(self, story, patterns):
        """Create clinical patterns analysis section"""
        story.append(Paragraph("CLINICAL PATTERN ANALYSIS", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=12))
        
        pattern_data = []
        
        if patterns['chronic_conditions']:
            unique_chronic = list(set(patterns['chronic_conditions']))[:5]
            pattern_data.append([
                Paragraph("<b>Chronic Conditions</b>", self.styles['Normal']),
                Paragraph(", ".join(unique_chronic), self.styles['NormalText']),
                Paragraph("Long-term management required", 
                         ParagraphStyle('Note', parent=self.styles['Normal'], 
                                      fontSize=9, textColor=colors.HexColor('#7B8794')))
            ])
        
        if patterns['acute_symptoms']:
            unique_acute = list(set(patterns['acute_symptoms']))[:5]
            pattern_data.append([
                Paragraph("<b>Acute Findings</b>", self.styles['Normal']),
                Paragraph(", ".join(unique_acute), self.styles['NormalText']),
                Paragraph("Requires immediate attention", 
                         ParagraphStyle('Note', parent=self.styles['Normal'], 
                                      fontSize=9, textColor=colors.HexColor('#DC2626')))
            ])
        
        if patterns['medications']:
            unique_meds = list(set(patterns['medications']))[:8]
            pattern_data.append([
                Paragraph("<b>Current Medications</b>", self.styles['Normal']),
                Paragraph(", ".join(unique_meds), self.styles['NormalText']),
                Paragraph(f"Total: {len(unique_meds)} agents", 
                         ParagraphStyle('Note', parent=self.styles['Normal'], 
                                      fontSize=9, textColor=colors.HexColor('#7B8794')))
            ])
        
        if patterns['diagnostic_tests']:
            unique_tests = list(set(patterns['diagnostic_tests']))[:5]
            pattern_data.append([
                Paragraph("<b>Diagnostic Tests</b>", self.styles['Normal']),
                Paragraph(", ".join(unique_tests), self.styles['NormalText']),
                Paragraph("Ensure results reviewed", 
                         ParagraphStyle('Note', parent=self.styles['Normal'], 
                                      fontSize=9, textColor=colors.HexColor('#7B8794')))
            ])
        
        if patterns['anatomical_focus']:
            unique_anatomy = list(set(patterns['anatomical_focus']))[:5]
            pattern_data.append([
                Paragraph("<b>Anatomical Areas</b>", self.styles['Normal']),
                Paragraph(", ".join(unique_anatomy), self.styles['NormalText']),
                Paragraph("Focus areas identified", 
                         ParagraphStyle('Note', parent=self.styles['Normal'], 
                                      fontSize=9, textColor=colors.HexColor('#7B8794')))
            ])
        
        if pattern_data:
            pattern_table = Table(pattern_data, colWidths=[130, 240, 110])
            pattern_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2933')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8ECF0')),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), 
                 [colors.HexColor('#F8FAFC'), colors.white]),
            ]))
            story.append(pattern_table)
        else:
            story.append(Paragraph("No specific clinical patterns identified in the analyzed data.", 
                                 self.styles['NormalText']))
        
        story.append(Spacer(1, 20))

    def _create_recommendations_section(self, story, risk_analysis, patterns, complexity):
        """Enhanced recommendations with clinical context"""
        story.append(Paragraph("CLINICAL RECOMMENDATIONS & ACTION ITEMS", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=12))
        
        recommendations = risk_analysis.get('recommendations', [])
        
        # Add context-aware recommendations
        contextual_recs = []
        
        if complexity['score'] >= 6:
            contextual_recs.append({
                'title': 'Multidisciplinary care team review recommended due to high clinical complexity',
                'priority': 'high'
            })
        
        if len(patterns['medications']) > 5:
            contextual_recs.append({
                'title': 'Comprehensive medication reconciliation and interaction screening advised',
                'priority': 'medium'
            })
        
        if patterns['acute_symptoms']:
            contextual_recs.append({
                'title': 'Follow-up assessment for acute findings within 24-48 hours',
                'priority': 'high'
            })
        
        all_recommendations = contextual_recs + [
            {
                'title': rec.title if hasattr(rec, 'title') else rec.get('title', ''),
                'priority': rec.priority if hasattr(rec, 'priority') else rec.get('priority', 'medium')
            }
            for rec in recommendations
        ]
        
        if not all_recommendations:
            story.append(Paragraph("Standard care protocols recommended. Continue routine monitoring.", 
                                 self.styles['NormalText']))
            story.append(Spacer(1, 20))
            return
        
        # Priority categorization
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x['priority'].lower(), 3))
        
        priority_colors = {
            'critical': '#991B1B',
            'high': '#DC2626',
            'medium': '#F59E0B',
            'low': '#10B981'
        }
        
        priority_icons = {
            'critical': '⚠',
            'high': '●',
            'medium': '▲',
            'low': '○'
        }
        
        rec_data = [['Priority', 'Recommendation', 'Action Required']]
        
        for idx, rec in enumerate(all_recommendations[:12], 1):
            priority = rec['priority'].lower()
            color = priority_colors.get(priority, '#7B8794')
            icon = priority_icons.get(priority, '•')
            
            action_text = 'Immediate' if priority in ['critical', 'high'] else \
                         'Within 7 days' if priority == 'medium' else 'Routine follow-up'
            
            rec_data.append([
                Paragraph(f"{icon} <b>{rec['priority'].upper()}</b>", 
                         ParagraphStyle('Priority', parent=self.styles['Normal'],
                                      textColor=colors.HexColor(color),
                                      fontSize=9)),
                Paragraph(f"{idx}. {rec['title']}", self.styles['NormalText']),
                Paragraph(action_text, 
                         ParagraphStyle('Action', parent=self.styles['Normal'],
                                      fontSize=9, textColor=colors.HexColor('#7B8794')))
            ])
        
        rec_table = Table(rec_data, colWidths=[75, 310, 95])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8ECF0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.HexColor('#F8FAFC'), colors.white]),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 20))

    def _create_data_visualization_section(self, story, entities):
        """Create data visualization section"""
        story.append(PageBreak())
        story.append(Paragraph("DATA ANALYTICS & VISUALIZATION", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=15))
        
        # Entity distribution chart
        story.append(Paragraph("<b>Entity Type Distribution</b>", self.styles['SubsectionHeader']))
        chart1 = self._create_entity_distribution_chart(entities)
        if chart1:
            story.append(chart1)
        story.append(Spacer(1, 20))
        
        # Confidence distribution chart
        story.append(Paragraph("<b>Confidence Level Distribution</b>", self.styles['SubsectionHeader']))
        chart2 = self._create_confidence_distribution_chart(entities)
        if chart2:
            story.append(chart2)
        story.append(Spacer(1, 20))
        
        # Statistical summary
        story.append(Paragraph("<b>Statistical Summary</b>", self.styles['SubsectionHeader']))
        
        if entities:
            avg_confidence = sum(e.confidence for e in entities) / len(entities)
            high_conf = sum(1 for e in entities if e.confidence >= 0.8)
            
            stats_data = [
                ['Total Entities', str(len(entities))],
                ['Average Confidence', f"{avg_confidence:.1%}"],
                ['High Confidence Entities', f"{high_conf} ({high_conf/len(entities)*100:.1f}%)"],
                ['Unique Entity Types', str(len(set(e.label for e in entities)))]
            ]
            
            stats_table = Table(stats_data, colWidths=[200, 280])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8ECF0')),
            ]))
            story.append(stats_table)
        
        story.append(Spacer(1, 20))

    def _create_detailed_entities_section(self, story, entities):
        """Create comprehensive detailed entities section"""
        story.append(PageBreak())
        story.append(Paragraph("DETAILED ENTITY CATALOG", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=12))
        
        story.append(Paragraph(
            f"Complete listing of {len(entities)} identified clinical entities, sorted by confidence level. "
            "This catalog provides granular detail for clinical documentation and quality assurance.",
            self.styles['NormalText']
        ))
        story.append(Spacer(1, 15))
        
        # Group entities by category
        entity_by_category = {}
        for e in entities:
            if e.label not in entity_by_category:
                entity_by_category[e.label] = []
            entity_by_category[e.label].append(e)
        
        # Sort each category by confidence
        for label in entity_by_category:
            entity_by_category[label].sort(key=lambda x: x.confidence, reverse=True)
        
        # Display by category
        priority_labels = ['DISEASE', 'CONDITION', 'MEDICATION', 'SYMPTOM', 
                          'PROCEDURE', 'TEST', 'ANATOMY']
        
        for label in priority_labels:
            if label in entity_by_category:
                self._add_entity_category_table(story, label, entity_by_category[label])
        
        # Remaining categories
        for label in sorted(entity_by_category.keys()):
            if label not in priority_labels:
                self._add_entity_category_table(story, label, entity_by_category[label])

    def _add_entity_category_table(self, story, label, entities):
        """Add a table for a specific entity category"""
        story.append(Paragraph(f"<b>{label.replace('_', ' ').title()} ({len(entities)})</b>", 
                             self.styles['SubsectionHeader']))
        
        # Limit to top 20 per category to avoid overly long PDFs
        display_entities = entities[:20]
        
        table_data = [['#', 'Entity Text', 'Confidence', 'Context']]
        
        for idx, entity in enumerate(display_entities, 1):
            confidence_color = '#10B981' if entity.confidence >= 0.8 else \
                             '#F59E0B' if entity.confidence >= 0.6 else '#DC2626'
            
            # Add context if available (you can expand this based on your entity structure)
            context = "Primary finding" if entity.confidence >= 0.9 else \
                     "Secondary finding" if entity.confidence >= 0.7 else "Low confidence"
            
            table_data.append([
                str(idx),
                Paragraph(entity.text[:60] + "..." if len(entity.text) > 60 else entity.text,
                         self.styles['NormalText']),
                Paragraph(f"<b>{entity.confidence:.0%}</b>",
                         ParagraphStyle('Conf', parent=self.styles['Normal'],
                                      textColor=colors.HexColor(confidence_color),
                                      fontSize=9)),
                Paragraph(context, 
                         ParagraphStyle('Context', parent=self.styles['Normal'],
                                      fontSize=8, textColor=colors.HexColor('#7B8794')))
            ])
        
        entity_table = Table(table_data, colWidths=[25, 270, 70, 115])
        entity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F2933')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('ALIGN', (3, 0), (3, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#E8ECF0')),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor('#F0F0F0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.white, colors.HexColor('#FAFAFA')]),
        ]))
        
        story.append(entity_table)
        
        if len(entities) > 20:
            story.append(Paragraph(
                f"<i>Showing top 20 of {len(entities)} total {label.lower()} entities</i>",
                ParagraphStyle('Note', parent=self.styles['Normal'],
                             fontSize=8, textColor=colors.HexColor('#7B8794'),
                             spaceAfter=15)
            ))
        else:
            story.append(Spacer(1, 15))

    def _create_clinical_summary_section(self, story, risk_analysis):
        """Create enhanced clinical summary"""
        story.append(Paragraph("CLINICAL NARRATIVE SUMMARY", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=12))
        
        insights = risk_analysis.get('clinical_summary', 
                                    'Comprehensive clinical analysis performed on provided medical documentation. '
                                    'Natural language processing algorithms have extracted and categorized clinical entities '
                                    'to support clinical decision-making and documentation quality assurance.')
        
        story.append(Paragraph(insights, self.styles['NormalText']))
        story.append(Spacer(1, 20))

    def _create_methodology_section(self, story):
        """Add methodology and technical notes"""
        story.append(PageBreak())
        story.append(Paragraph("METHODOLOGY & TECHNICAL NOTES", self.styles['SectionHeader']))
        story.append(HRFlowable(width="100%", thickness=1, 
                               color=colors.HexColor('#E8ECF0'), spaceAfter=12))
        
        methodology_text = """
        <b>Analysis Framework:</b> This report was generated using advanced medical Natural Language Processing (NLP) 
        technology designed to extract, categorize, and analyze clinical entities from unstructured medical text. 
        The system employs machine learning models trained on extensive medical literature and clinical documentation.
        <br/><br/>
        <b>Entity Recognition:</b> Clinical entities are identified using named entity recognition (NER) algorithms 
        with confidence scores reflecting the model's certainty in each classification. High confidence entities 
        (≥80%) represent clear, unambiguous clinical findings, while lower confidence entities may require human validation.
        <br/><br/>
        <b>Risk Assessment:</b> The risk stratification model considers multiple factors including diagnosis complexity, 
        medication burden, procedural history, and identified clinical patterns. Risk scores should be interpreted 
        within the broader clinical context and validated by qualified healthcare professionals.
        <br/><br/>
        <b>Limitations:</b> This automated analysis supplements but does not replace clinical judgment. The system 
        may not capture contextual nuances, temporal relationships, or negated findings without explicit markers. 
        All findings should be validated against original source documentation.
        """
        
        story.append(Paragraph(methodology_text, self.styles['NormalText']))
        story.append(Spacer(1, 20))

    def _create_disclaimer(self, story):
        """Enhanced medical disclaimer with legal protections"""
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=2, 
                               color=colors.HexColor('#DC2626'), spaceAfter=10))
        
        disclaimer = """
        <b>IMPORTANT MEDICAL AND LEGAL DISCLAIMER:</b><br/><br/>
        
        This report is generated by an automated Natural Language Processing (NLP) system and is provided for 
        <b>informational, research, and quality assurance purposes only</b>. This document does NOT constitute:
        <br/>• Medical advice, diagnosis, or treatment recommendations
        <br/>• A substitute for professional medical judgment or clinical examination
        <br/>• A complete or verified medical record
        <br/>• Legal or regulatory documentation for patient care decisions
        <br/><br/>
        
        <b>Healthcare Provider Responsibilities:</b> All clinical decisions must be made by qualified, licensed 
        healthcare professionals based on comprehensive patient evaluation, current evidence-based guidelines, and 
        professional clinical judgment. The risk assessments, recommendations, and clinical patterns identified in 
        this report are algorithmic estimates that require validation and interpretation by medical professionals.
        <br/><br/>
        
        <b>Data Accuracy:</b> While this system employs advanced medical NLP technology, automated analysis may 
        contain errors, omissions, or misclassifications. Users must verify all findings against original source 
        documentation and clinical records before taking any action.
        <br/><br/>
        
        <b>Confidentiality:</b> This report contains confidential medical information protected under applicable 
        privacy laws including HIPAA. Unauthorized disclosure, distribution, or use is strictly prohibited.
        <br/><br/>
        
        <b>No Warranty:</b> This analysis is provided "as is" without warranties of any kind, express or implied. 
        The developers and distributors of this system assume no liability for decisions made based on this report.
        <br/><br/>
        
        <i>Report Version 2.0 | MedNLP Clinical Intelligence System</i>
        """
        
        story.append(Paragraph(disclaimer, 
                             ParagraphStyle('Disclaimer', parent=self.styles['NormalText'],
                                          fontSize=7, textColor=colors.HexColor('#3E4C59'),
                                          alignment=TA_JUSTIFY, leading=10)))

    def generate_report(self, entities: List[Any], risk_analysis: Dict[str, Any]) -> bytes:
        """Generate a comprehensive, meaningful clinical PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title="MedNLP Clinical Intelligence Report",
            author="MedNLP System",
            subject="Clinical Analysis Report"
        )

        story = []

        # Analyze clinical patterns and complexity
        patterns = self._analyze_clinical_patterns(entities)
        complexity = self._calculate_clinical_complexity_score(entities, risk_analysis)

        # Build comprehensive report sections
        self._create_header_section(story)
        self._create_executive_summary(story, entities, risk_analysis, patterns, complexity)
        self._create_risk_assessment_section(story, risk_analysis, complexity)
        self._create_clinical_summary_section(story, risk_analysis)
        self._create_clinical_patterns_section(story, patterns)
        self._create_recommendations_section(story, risk_analysis, patterns, complexity)
        self._create_data_visualization_section(story, entities)
        self._create_detailed_entities_section(story, entities)
        self._create_methodology_section(story)
        self._create_disclaimer(story)
        
        # Build PDF with custom canvas for professional page layout
        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()