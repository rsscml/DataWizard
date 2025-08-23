"""
Enhanced Template Manager for Multi-Worksheet Support
===================================================

Extended to support templates with multiple worksheets/CSV files
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ColumnDefinition:
    """Definition of a column in a template"""
    name: str
    business_name: str
    description: str
    data_type: str  # 'numeric', 'categorical', 'datetime', 'text', 'boolean'
    business_type: str  # 'metric', 'dimension', 'identifier', 'timestamp'
    unit: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    expected_values: Optional[List[str]] = None
    relationships: Optional[List[str]] = None  # Related column names
    
@dataclass
class WorksheetDefinition:
    """Definition of a worksheet/file within a template"""
    name: str  # Worksheet name or file pattern
    description: str
    columns: List[ColumnDefinition]
    is_required: bool = True  # Whether this worksheet is required for template match
    file_patterns: Optional[List[str]] = None  # For CSV templates - filename patterns
    relationships: Optional[List[str]] = None  # Related worksheets
    primary_key: Optional[str] = None  # Column that serves as primary key
    
@dataclass
class TemplateMetrics:
    """Business metrics and KPIs for a template"""
    name: str
    description: str
    formula: str  # Python expression using worksheet.column notation
    category: str  # 'financial', 'operational', 'performance', etc.
    display_format: str = "%.2f"
    worksheets_involved: Optional[List[str]] = None  # Which worksheets this metric uses

@dataclass
class TemplateVisualization:
    """Suggested visualizations for a template"""
    name: str
    description: str
    chart_type: str  # 'bar', 'line', 'pie', 'scatter', etc.
    x_axis: str  # worksheet.column format for multi-worksheet
    y_axis: str
    group_by: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    worksheets_involved: Optional[List[str]] = None

@dataclass
class WorksheetRelationship:
    """Defines relationships between worksheets"""
    name: str
    description: str
    from_worksheet: str
    to_worksheet: str
    from_column: str
    to_column: str
    relationship_type: str  # 'one_to_many', 'many_to_one', 'one_to_one'

@dataclass
class TemplateDefinition:
    """Complete template definition with multi-worksheet support"""
    id: str
    name: str
    description: str
    version: str
    worksheets: List[WorksheetDefinition]  # Changed from columns to worksheets
    business_context: str
    common_analyses: List[str]
    metrics: List[TemplateMetrics]
    visualizations: List[TemplateVisualization]
    worksheet_relationships: Optional[List[WorksheetRelationship]] = None
    detection_rules: Dict[str, Any] = None
    domain: str = 'general'  # 'sales', 'finance', 'hr', 'operations', etc.
    template_type: str = 'single_worksheet'  # 'single_worksheet', 'multi_worksheet', 'csv_collection'
    created_date: str = ''
    last_modified: str = ''
    
    # Legacy support - auto-generate for backward compatibility
    @property
    def columns(self) -> List[ColumnDefinition]:
        """For backward compatibility - returns columns from first worksheet"""
        if self.worksheets and len(self.worksheets) > 0:
            return self.worksheets[0].columns
        return []

class EnhancedTemplateManager:
    """Enhanced template manager with multi-worksheet support"""
    
    def __init__(self, templates_dir: str = "data_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates: Dict[str, TemplateDefinition] = {}
        self.load_templates()
        
    def load_templates(self):
        """Load all templates from the templates directory"""
        try:
            template_files = list(self.templates_dir.glob("*.json"))
            for template_file in template_files:
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                    
                    template = self._dict_to_template(template_data)
                    self.templates[template.id] = template
                    logger.info(f"Loaded template: {template.name} (ID: {template.id}, Type: {template.template_type})")
                    
                except Exception as e:
                    logger.error(f"Error loading template {template_file}: {e}")
                    
            logger.info(f"Loaded {len(self.templates)} templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _dict_to_template(self, data: Dict) -> TemplateDefinition:
        """Convert dictionary to TemplateDefinition with backward compatibility"""
        
        # Handle legacy single-worksheet templates
        if 'columns' in data and 'worksheets' not in data:
            # Convert legacy format to new format
            columns = [ColumnDefinition(**col_data) for col_data in data.get('columns', [])]
            worksheets = [WorksheetDefinition(
                name="Main",
                description="Main worksheet",
                columns=columns,
                is_required=True
            )]
            template_type = 'single_worksheet'
        else:
            # New multi-worksheet format
            worksheets = []
            for ws_data in data.get('worksheets', []):
                columns = [ColumnDefinition(**col_data) for col_data in ws_data.get('columns', [])]
                worksheets.append(WorksheetDefinition(
                    name=ws_data['name'],
                    description=ws_data['description'],
                    columns=columns,
                    is_required=ws_data.get('is_required', True),
                    file_patterns=ws_data.get('file_patterns'),
                    relationships=ws_data.get('relationships'),
                    primary_key=ws_data.get('primary_key')
                ))
            template_type = data.get('template_type', 'multi_worksheet')
        
        # Convert metrics
        metrics = []
        for metric_data in data.get('metrics', []):
            metrics.append(TemplateMetrics(**metric_data))
        
        # Convert visualizations
        visualizations = []
        for viz_data in data.get('visualizations', []):
            visualizations.append(TemplateVisualization(**viz_data))
            
        # Convert worksheet relationships
        relationships = []
        for rel_data in data.get('worksheet_relationships', []):
            relationships.append(WorksheetRelationship(**rel_data))
        
        return TemplateDefinition(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            version=data['version'],
            worksheets=worksheets,
            business_context=data['business_context'],
            common_analyses=data.get('common_analyses', []),
            metrics=metrics,
            visualizations=visualizations,
            worksheet_relationships=relationships,
            detection_rules=data.get('detection_rules', {}),
            domain=data.get('domain', 'general'),
            template_type=template_type,
            created_date=data.get('created_date', datetime.now().isoformat()),
            last_modified=data.get('last_modified', datetime.now().isoformat())
        )
    
    def detect_template(self, data_source, filename: str = None) -> Optional[TemplateDefinition]:
        """
        Enhanced template detection for multiple data sources
        
        Args:
            data_source: Either pd.DataFrame (single) or Dict[str, pd.DataFrame] (multi-worksheet)
            filename: Original filename for pattern matching
            
        Returns:
            Best matching template or None
        """
        # Determine if we have single or multiple worksheets
        if isinstance(data_source, pd.DataFrame):
            # Single worksheet/CSV
            worksheet_data = {"Main": data_source}
            is_multi_sheet = False
        elif isinstance(data_source, dict):
            # Multiple worksheets
            worksheet_data = data_source
            is_multi_sheet = True
        else:
            logger.error(f"Invalid data_source type: {type(data_source)}")
            return None
        
        best_match = None
        best_score = 0
        
        for template_id, template in self.templates.items():
            score = self._calculate_multi_worksheet_match_score(
                worksheet_data, template, filename, is_multi_sheet
            )
            logger.debug(f"Template {template.name} match score: {score:.3f}")
            
            if score > best_score and score >= 0.6:  # Minimum 60% match for multi-worksheet
                best_match = template
                best_score = score
        
        if best_match:
            logger.info(f"Detected template: {best_match.name} (score: {best_score:.2f})")
        else:
            logger.info("No template detected for this data")
            
        return best_match
    
    def _calculate_multi_worksheet_match_score(self, worksheet_data: Dict[str, pd.DataFrame], 
                                             template: TemplateDefinition, filename: str = None,
                                             is_multi_sheet: bool = False) -> float:
        """Calculate template match score for multi-worksheet data"""
        
        # Template type compatibility check
        if template.template_type == 'single_worksheet' and is_multi_sheet:
            # Single worksheet template but multi-sheet data - check main worksheet only
            if "Main" not in worksheet_data and len(worksheet_data) > 0:
                # Use the first worksheet as main
                main_worksheet = list(worksheet_data.values())[0]
                return self._calculate_single_worksheet_score(main_worksheet, template.worksheets[0], filename)
            elif "Main" in worksheet_data:
                return self._calculate_single_worksheet_score(worksheet_data["Main"], template.worksheets[0], filename)
        
        elif template.template_type in ['multi_worksheet', 'csv_collection'] and not is_multi_sheet:
            # Multi-worksheet template but single sheet data - partial match only
            if len(template.worksheets) == 1:
                return self._calculate_single_worksheet_score(worksheet_data["Main"], template.worksheets[0], filename)
            else:
                # Can't match multi-worksheet template with single worksheet
                return 0.0
        
        # Multi-worksheet to multi-worksheet matching
        if template.template_type in ['multi_worksheet', 'csv_collection'] and is_multi_sheet:
            return self._calculate_multi_worksheet_score_full(worksheet_data, template, filename)
        
        # Single to single matching
        if template.template_type == 'single_worksheet' and not is_multi_sheet:
            return self._calculate_single_worksheet_score(worksheet_data["Main"], template.worksheets[0], filename)
        
        return 0.0
    
    def _calculate_multi_worksheet_score_full(self, worksheet_data: Dict[str, pd.DataFrame],
                                            template: TemplateDefinition, filename: str = None) -> float:
        """Calculate score for full multi-worksheet matching"""
        total_score = 0.0
        total_weight = 0.0
        required_worksheets_found = 0
        required_worksheets_total = sum(1 for ws in template.worksheets if ws.is_required)
        
        # Score each worksheet
        for template_ws in template.worksheets:
            best_ws_score = 0.0
            
            # Try to match this template worksheet with actual worksheets
            for actual_ws_name, actual_ws_data in worksheet_data.items():
                ws_score = self._calculate_worksheet_match_score(actual_ws_data, template_ws, actual_ws_name)
                if ws_score > best_ws_score:
                    best_ws_score = ws_score
            
            # Weight by requirement
            weight = 1.0 if template_ws.is_required else 0.5
            total_score += best_ws_score * weight
            total_weight += weight
            
            if template_ws.is_required and best_ws_score > 0.5:
                required_worksheets_found += 1
        
        # Penalty for missing required worksheets
        required_penalty = 1.0
        if required_worksheets_total > 0:
            required_penalty = required_worksheets_found / required_worksheets_total
        
        # Filename bonus
        filename_bonus = 0.0
        if filename and template.detection_rules and 'filename_patterns' in template.detection_rules:
            filename_patterns = template.detection_rules['filename_patterns']
            if any(pattern.lower() in filename.lower() for pattern in filename_patterns):
                filename_bonus = 0.1
        
        base_score = total_score / total_weight if total_weight > 0 else 0.0
        final_score = (base_score * required_penalty) + filename_bonus
        
        return min(final_score, 1.0)
    
    def _calculate_worksheet_match_score(self, df: pd.DataFrame, template_ws: WorksheetDefinition, 
                                       worksheet_name: str = None) -> float:
        """Calculate match score between a DataFrame and template worksheet"""
        score = 0.0
        
        # Column name matching
        template_columns = {col.name for col in template_ws.columns}
        actual_columns = set(df.columns)
        
        if template_columns:
            matched_columns = len(template_columns.intersection(actual_columns))
            column_score = matched_columns / len(template_columns)
            score += column_score * 0.7
        
        # Worksheet name matching (for Excel files)
        if worksheet_name and template_ws.name != "Main":
            name_similarity = self._calculate_name_similarity(worksheet_name, template_ws.name)
            score += name_similarity * 0.2
        
        # File pattern matching (for CSV collections)
        if worksheet_name and template_ws.file_patterns:
            pattern_match = any(pattern.lower() in worksheet_name.lower() 
                              for pattern in template_ws.file_patterns)
            if pattern_match:
                score += 0.1
        
        # Data type validation bonus
        type_bonus = self._validate_worksheet_data_types(df, template_ws)
        score += type_bonus * 0.1
        
        return min(score, 1.0)
    
    def _calculate_single_worksheet_score(self, df: pd.DataFrame, template_ws: WorksheetDefinition, 
                                        filename: str = None) -> float:
        """Calculate score for single worksheet (backward compatibility)"""
        return self._calculate_worksheet_match_score(df, template_ws, filename)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        name1_lower = name1.lower().replace('_', ' ').replace('-', ' ')
        name2_lower = name2.lower().replace('_', ' ').replace('-', ' ')
        
        if name1_lower == name2_lower:
            return 1.0
        elif name2_lower in name1_lower or name1_lower in name2_lower:
            return 0.8
        else:
            # Simple word overlap
            words1 = set(name1_lower.split())
            words2 = set(name2_lower.split())
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                return overlap / max(len(words1), len(words2))
        
        return 0.0
    
    def _validate_worksheet_data_types(self, df: pd.DataFrame, template_ws: WorksheetDefinition) -> float:
        """Validate data types for a specific worksheet"""
        matches = 0
        total = 0
        
        for col_def in template_ws.columns:
            if col_def.name in df.columns:
                total += 1
                expected_type = col_def.data_type
                if (expected_type == 'numeric' and pd.api.types.is_numeric_dtype(df[col_def.name])) or \
                   (expected_type == 'categorical' and pd.api.types.is_object_dtype(df[col_def.name])) or \
                   (expected_type == 'datetime' and pd.api.types.is_datetime64_any_dtype(df[col_def.name])):
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def generate_enhanced_context(self, data_source, template: TemplateDefinition, base_context: str) -> str:
        """Generate enhanced context for multi-worksheet templates"""
        
        # Determine data structure
        if isinstance(data_source, pd.DataFrame):
            worksheet_data = {"Main": data_source}
        else:
            worksheet_data = data_source
        
        context_parts = [
            "",
            f"DATASET DOMAIN SPECIFIC INFORMATION:",
            template.business_context,
            "",
        ]
        
        # Multi-worksheet information
        if template.template_type in ['multi_worksheet', 'csv_collection']:
            context_parts.extend([
                f"Multi-Worksheet Template Structure:",
                f"   - Total Worksheets Defined: {len(template.worksheets)}",
                f"   - Available Data Worksheets: {len(worksheet_data)}",
                "",
            ])
            
            # Add worksheet-specific information
            for i, template_ws in enumerate(template.worksheets, 1):
                context_parts.append(f" Worksheet {i}: {template_ws.name}")
                context_parts.append(f"      - Description: {template_ws.description}")
                context_parts.append(f"      - Required: {'Yes' if template_ws.is_required else 'No'}")
                context_parts.append(f"      - Columns: {len(template_ws.columns)}")
                if template_ws.primary_key:
                    context_parts.append(f"      - Primary Key: {template_ws.primary_key}")
                
                # Show column details for this worksheet
                context_parts.append(f"      - Column Details:")
                for col_def in template_ws.columns[:5]:  # Show first 5 columns
                    context_parts.append(f"        * {col_def.name}: {col_def.description}")
                if len(template_ws.columns) > 5:
                    context_parts.append(f"        * ... and {len(template_ws.columns) - 5} more columns")
                context_parts.append("")
            
            # Add worksheet relationships
            if template.worksheet_relationships:
                context_parts.extend([
                    f" Worksheet Relationships:",
                ])
                for rel in template.worksheet_relationships:
                    context_parts.append(f"   - {rel.name}: {rel.description}")
                    context_parts.append(f"     - {rel.from_worksheet}.{rel.from_column} → {rel.to_worksheet}.{rel.to_column}")
                    context_parts.append(f"     - Type: {rel.relationship_type}")
                    context_parts.append("")
        
        else:
            # Single worksheet template
            template_ws = template.worksheets[0]
            context_parts.extend([
                f" Enhanced Column Definitions:",
            ])
            for col_def in template_ws.columns:
                context_parts.append(f"   - {col_def.name}:")
                context_parts.append(f"     - Business Name: {col_def.business_name}")
                context_parts.append(f"     - Description: {col_def.description}")
                context_parts.append(f"     - Business Type: {col_def.business_type}")
                if col_def.unit:
                    context_parts.append(f"     - Unit: {col_def.unit}")
                if col_def.relationships:
                    context_parts.append(f"     - Related to: {', '.join(col_def.relationships)}")
                context_parts.append("")
        
        # Add business metrics
        if template.metrics:
            context_parts.extend([
                f" Available Business Metrics:",
            ])
            for metric in template.metrics:
                context_parts.append(f"   - {metric.name} ({metric.category}):")
                context_parts.append(f"     - Description: {metric.description}")
                context_parts.append(f"     - Formula: {metric.formula}")
                if metric.worksheets_involved:
                    context_parts.append(f"     - Worksheets: {', '.join(metric.worksheets_involved)}")
                context_parts.append("")
        
        # Add suggested analyses
        if template.common_analyses:
            context_parts.extend([
                f" Common Analysis Patterns:",
            ])
            for analysis in template.common_analyses:
                context_parts.append(f"   • {analysis}")
            context_parts.append("")
        
        # Add visualization suggestions
        if template.visualizations:
            context_parts.extend([
                f" Suggested Visualizations:",
            ])
            for viz in template.visualizations:
                context_parts.append(f"   - {viz.name}: {viz.description}")
                context_parts.append(f"     - Chart Type: {viz.chart_type}")
                context_parts.append(f"     - X-axis: {viz.x_axis}, Y-axis: {viz.y_axis}")
                if viz.group_by:
                    context_parts.append(f"     - Group by: {viz.group_by}")
                if viz.worksheets_involved:
                    context_parts.append(f"     - Worksheets: {', '.join(viz.worksheets_involved)}")
                context_parts.append("")
        
        # Add guidance for multi-worksheet queries
        if template.template_type in ['multi_worksheet', 'csv_collection']:
            context_parts.extend([
                f" Multi-Worksheet Query Guidance:",
                f"   • Use worksheet_data['worksheet_name'] to access specific worksheets",
                f"   • Use merge_worksheets(worksheet_data) for cross-worksheet analysis",
                f"   • Consider relationships when joining data across worksheets",
                f"   • Template metrics may span multiple worksheets",
                "",
            ])
        
        # Add the original context
        context_parts.extend([
            " STANDARD DATA PROFILE:",
            base_context
        ])
        
        return "\n".join(context_parts)
    
    # Keep existing methods for backward compatibility
    def save_template(self, template: TemplateDefinition):
        """Save a template to disk"""
        template_file = self.templates_dir / f"{template.id}.json"
        
        # Convert to dict
        template_dict = asdict(template)
        
        # Update modification time
        template_dict['last_modified'] = datetime.now().isoformat()
        
        try:
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, indent=2, ensure_ascii=False)
            
            # Update in-memory templates
            self.templates[template.id] = template
            logger.info(f"Saved template: {template.name}")
            
        except Exception as e:
            logger.error(f"Error saving template {template.id}: {e}")
            raise
    
    def get_template_suggestions(self, data_source) -> List[Tuple[str, float]]:
        """Get template suggestions with match scores"""
        suggestions = []
        
        for template_id, template in self.templates.items():
            score = self._calculate_multi_worksheet_match_score(
                data_source if isinstance(data_source, dict) else {"Main": data_source},
                template,
                None,
                isinstance(data_source, dict)
            )
            if score > 0.3:  # Only suggest if at least 30% match
                suggestions.append((template.name, score))
        
        return sorted(suggestions, key=lambda x: x[1], reverse=True)
    
    def get_all_templates(self) -> List[TemplateDefinition]:
        """Get all available templates"""
        return list(self.templates.values())
    
    def get_template_by_id(self, template_id: str) -> Optional[TemplateDefinition]:
        """Get template by ID"""
        return self.templates.get(template_id)

# Create enhanced template manager instance
template_manager = EnhancedTemplateManager("data_templates")

# Backward compatibility alias
TemplateManager = EnhancedTemplateManager