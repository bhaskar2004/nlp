from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import datetime
from typing import List, Dict, Any
import pandas as pd

class MedicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1F2933'),
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#0066CC'),
            spaceBefore=20,
            spaceAfter=10,
            borderPadding=(0, 0, 5, 0),
            borderWidth=1,
            borderColor=colors.HexColor('#E8ECF0'),
            borderRadius=None
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#3E4C59'),
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskBadge',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.white,
            backColor=colors.HexColor('#7B8794'),
            alignment=TA_CENTER,
            borderPadding=5,
            borderRadius=4
        ))

    def generate_report(self, entities: List[Any], risk_analysis: Dict[str, Any]) -> bytes:
        """Generate a PDF report and return it as bytes"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        story = []

        # --- Header ---
        story.append(Paragraph("MedNLP Clinical Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                             self.styles['Normal']))
        story.append(Spacer(1, 20))

        # --- Clinical Summary ---
        story.append(Paragraph("Clinical Summary", self.styles['SectionHeader']))
        
        insights = risk_analysis.get('clinical_summary', 'No summary available.')
        story.append(Paragraph(insights, self.styles['NormalText']))
        story.append(Spacer(1, 20))

        # --- Risk Assessment ---
        story.append(Paragraph("Risk Assessment", self.styles['SectionHeader']))
        
        risk_score = risk_analysis.get('overall_risk_score', 0)
        risk_strat = risk_analysis.get('risk_stratification', {})
        risk_level = risk_strat.get('category', 'Unknown')
        risk_color = risk_strat.get('color', '#7B8794')
        
        # Create a visual risk indicator table
        risk_data = [
            [Paragraph("<b>Overall Risk Score</b>", self.styles['Normal']), 
             f"{risk_score:.1f}/10"],
            [Paragraph("<b>Risk Category</b>", self.styles['Normal']), 
             Paragraph(f"<b>{risk_level.upper()}</b>", 
                       ParagraphStyle('RiskLevel', parent=self.styles['Normal'], textColor=colors.HexColor(risk_color)))]
        ]
        
        risk_table = Table(risk_data, colWidths=[200, 200])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2933')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E8ECF0')),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 20))

        # --- Key Findings ---
        story.append(Paragraph("Key Findings", self.styles['SectionHeader']))
        
        # Conditions
        conditions = [e.text for e in entities if e.label in ['DISEASE', 'CONDITION']]
        if conditions:
            story.append(Paragraph("<b>Identified Conditions:</b>", self.styles['Normal']))
            story.append(Paragraph(", ".join(set(conditions)), self.styles['NormalText']))
            story.append(Spacer(1, 10))
            
        # Medications
        medications = [e.text for e in entities if e.label == 'MEDICATION']
        if medications:
            story.append(Paragraph("<b>Current Medications:</b>", self.styles['Normal']))
            story.append(Paragraph(", ".join(set(medications)), self.styles['NormalText']))
            story.append(Spacer(1, 10))

        # --- Recommendations ---
        recommendations = risk_analysis.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))
            for rec in recommendations:
                # Handle both object and dict access for recommendations
                title = rec.title if hasattr(rec, 'title') else rec.get('title', '')
                priority = rec.priority if hasattr(rec, 'priority') else rec.get('priority', '')
                
                rec_text = f"• <b>[{priority.upper()}]</b> {title}"
                story.append(Paragraph(rec_text, self.styles['NormalText']))
                story.append(Spacer(1, 5))
            story.append(Spacer(1, 20))

        # --- Detailed Entities Table ---
        story.append(Paragraph("Detailed Entity List", self.styles['SectionHeader']))
        
        # Prepare data for table
        table_data = [['Entity', 'Type', 'Confidence']]
        for entity in entities[:50]:  # Limit to top 50 to avoid huge PDFs
            table_data.append([
                entity.text[:40] + "..." if len(entity.text) > 40 else entity.text,
                entity.label,
                f"{entity.confidence:.1%}"
            ])
            
        t = Table(table_data, colWidths=[250, 150, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#3E4C59')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E8ECF0')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(t)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
