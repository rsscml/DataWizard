"""
Enhanced Data Analysis with Multi-Worksheet Template Support
===========================================================

This module extends the existing analyze_data function to support template detection
and enhanced context generation for both single and multi-worksheet templates.
"""

import pandas as pd
import numpy as np
from template_manager import template_manager, TemplateDefinition
from enhanced_data_profiler import generate_enhanced_data_summary
import re
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def analyze_data_with_template_support(data_source: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                                     filename: str = None) -> Tuple[Dict[str, Any], Optional[TemplateDefinition]]:
    """
    Enhanced version of analyze_data that includes template detection for multi-worksheet data.
    
    Args:
        data_source: Either pd.DataFrame (single) or Dict[str, pd.DataFrame] (multi-worksheet)
        filename: Original filename for pattern matching
        
    Returns:
        Tuple of (analysis_summary, detected_template)
    """
    # Determine if we have single or multiple worksheets
    if isinstance(data_source, pd.DataFrame):
        # Single worksheet analysis
        standard_summary = analyze_data_standard(data_source)
        main_data = data_source
        is_multi_worksheet = False
    else:
        # Multi-worksheet analysis - use active worksheet for primary analysis
        worksheet_data = data_source
        # Get the first worksheet as main for analysis summary
        main_data = list(worksheet_data.values())[0] if worksheet_data else pd.DataFrame()
        standard_summary = analyze_data_standard(main_data)
        is_multi_worksheet = True
        
        # Add multi-worksheet information to summary
        standard_summary['multi_worksheet_info'] = {
            'total_worksheets': len(worksheet_data),
            'worksheet_names': list(worksheet_data.keys()),
            'worksheet_shapes': {name: list(df.shape) for name, df in worksheet_data.items()},
            'is_multi_worksheet': True
        }
    
    # Attempt template detection with appropriate data structure
    detected_template = template_manager.detect_template(data_source, filename)
    
    if detected_template:
        logger.info(f"Template detected: {detected_template.name} (Type: {detected_template.template_type})")
        
        # Add template information to summary
        standard_summary['template_info'] = {
            'template_id': detected_template.id,
            'template_name': detected_template.name,
            'template_domain': detected_template.domain,
            'template_version': detected_template.version,
            'template_type': detected_template.template_type,
            'enhanced_context_available': True,
            'multi_worksheet_template': detected_template.template_type in ['multi_worksheet', 'csv_collection']
        }
        
        # Add template-specific multi-worksheet information
        if is_multi_worksheet and detected_template.template_type in ['multi_worksheet', 'csv_collection']:
            standard_summary['template_info']['worksheet_mapping'] = _map_worksheets_to_template(
                worksheet_data, detected_template
            )
            
    else:
        logger.info("No template detected, using standard analysis")
        standard_summary['template_info'] = {
            'enhanced_context_available': False,
            'multi_worksheet_template': False
        }
    
    return standard_summary, detected_template

def _map_worksheets_to_template(worksheet_data: Dict[str, pd.DataFrame], 
                               template: TemplateDefinition) -> Dict[str, Any]:
    """Map actual worksheets to template worksheet definitions"""
    mapping = {
        'template_worksheets': len(template.worksheets),
        'actual_worksheets': len(worksheet_data),
        'mappings': []
    }
    
    for template_ws in template.worksheets:
        best_match = None
        best_score = 0.0
        
        for actual_name, actual_df in worksheet_data.items():
            # Calculate match score between template worksheet and actual worksheet
            score = template_manager._calculate_worksheet_match_score(actual_df, template_ws, actual_name)
            if score > best_score:
                best_match = actual_name
                best_score = score
        
        mapping['mappings'].append({
            'template_worksheet': template_ws.name,
            'template_description': template_ws.description,
            'matched_worksheet': best_match,
            'match_score': best_score,
            'is_required': template_ws.is_required,
            'columns_expected': len(template_ws.columns),
            'columns_matched': _count_matched_columns(
                worksheet_data.get(best_match, pd.DataFrame()), template_ws
            ) if best_match else 0
        })
    
    return mapping

def _count_matched_columns(df: pd.DataFrame, template_ws) -> int:
    """Count how many template columns are present in the actual worksheet"""
    if df.empty:
        return 0
    
    template_columns = {col.name for col in template_ws.columns}
    actual_columns = set(df.columns)
    return len(template_columns.intersection(actual_columns))


def detect_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect temporal patterns in both column values and column names.
    Handles business-specific date formats like 'Q1 22', '2024-26', etc.
    """
    temporal_analysis = {
        'temporal_value_columns': [],
        'temporal_name_columns': [],
        'recommendations': [],
        'summary': {
            'total_temporal_columns': 0,
            'value_based_temporal': 0,
            'name_based_temporal': 0,
            'requires_sorting_attention': []
        }
    }

    # Define comprehensive temporal patterns
    temporal_patterns = {
        'quarter': {
            'patterns': [
                r'Q[1-4]\s*[-_]?\s*(\d{2}|\d{4})',  # Q1 22, Q1-2022, Q1_22
                r'Quarter\s*[1-4]\s*[-_]?\s*(\d{2}|\d{4})',  # Quarter 1 2022
                r'(\d{2}|\d{4})[-_]?Q[1-4]',  # 2022-Q1, 22Q1
                r'[1-4]Q\s*[-_]?\s*(\d{2}|\d{4})',  # 1Q 22, 1Q-2022
            ],
            'description': 'Quarterly periods',
            'sort_guidance': 'Sort by year then quarter (Q1, Q2, Q3, Q4)'
        },
        'year_week': {
            'patterns': [
                r'(\d{2}|\d{4})[-_]W?\d{1,2}',  # 2024-26, 2024W26, 24-26
                r'W\d{1,2}[-_](\d{2}|\d{4})',  # W26-2024
                r'Week\s*\d{1,2}\s*[-_]?\s*(\d{2}|\d{4})',  # Week 26 2024
                r'(\d{2}|\d{4})WK\d{1,2}',  # 2024WK26
            ],
            'description': 'Year-Week periods',
            'sort_guidance': 'Sort by year then week number (1-53)'
        },
        'year_month': {
            'patterns': [
                r'(\d{2}|\d{4})[-_]M?\d{1,2}',  # 2024-01, 2024M01
                r'M\d{1,2}[-_](\d{2}|\d{4})',  # M01-2024
                r'(\d{2}|\d{4})[-_](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # 2024-Jan
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_](\d{2}|\d{4})',  # Jan-2024
                r'(\d{2}|\d{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # 2024Jan
            ],
            'description': 'Year-Month periods',
            'sort_guidance': 'Sort by year then month (Jan-Dec or 01-12)'
        },
        'fiscal_year': {
            'patterns': [
                r'FY\s*(\d{2}|\d{4})',  # FY2024, FY 24
                r'(\d{2}|\d{4})\s*FY',  # 2024FY, 24 FY
                r'Fiscal\s*(\d{2}|\d{4})',  # Fiscal 2024
            ],
            'description': 'Fiscal Year periods',
            'sort_guidance': 'Sort by fiscal year chronologically'
        },
        'half_year': {
            'patterns': [
                r'H[1-2]\s*[-_]?\s*(\d{2}|\d{4})',  # H1 2024, H1-2024
                r'(\d{2}|\d{4})[-_]H[1-2]',  # 2024-H1
                r'(1st|2nd)\s*Half\s*[-_]?\s*(\d{2}|\d{4})',  # 1st Half 2024
            ],
            'description': 'Half-Year periods',
            'sort_guidance': 'Sort by year then half (H1, H2)'
        },
        'year_only': {
            'patterns': [
                r'^\d{4}$',  # 2024
                r'^(\d{2})$',  # 24 (assuming year)
            ],
            'description': 'Year periods',
            'sort_guidance': 'Sort chronologically by year'
        }
    }

    # Check column values for temporal patterns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        try:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                continue

            # Sample values for pattern matching (use more samples for better detection)
            sample_size = min(50, len(col_data))
            sample_values = col_data.head(sample_size).tolist()

            # Clean sample values (strip whitespace)
            sample_values = [str(val).strip() for val in sample_values if str(val).strip()]

            if not sample_values:
                continue

            for pattern_type, pattern_info in temporal_patterns.items():
                pattern_matches = 0
                matched_values = []

                for pattern in pattern_info['patterns']:
                    for value in sample_values:
                        if re.match(pattern, value, re.IGNORECASE):
                            pattern_matches += 1
                            matched_values.append(value)
                            # Remove this value to avoid double counting
                            sample_values = [v for v in sample_values if v != value]
                            break

                # If significant portion matches temporal pattern
                match_ratio = pattern_matches / sample_size if sample_size > 0 else 0

                if match_ratio >= 0.6:  # 60% of sampled values match
                    # Check for sorting issues
                    unique_values = col_data.unique()[:20]  # Check first 20 unique values
                    is_properly_sorted = _check_temporal_sorting(unique_values, pattern_type)

                    temporal_column_info = {
                        'column': str(col),
                        'pattern_type': pattern_type,
                        'description': pattern_info['description'],
                        'match_ratio': round(match_ratio, 3),
                        'sample_matches': matched_values[:5],
                        'detection_method': 'value_pattern',
                        'sort_guidance': pattern_info['sort_guidance'],
                        'granularity': pattern_type,
                        'is_properly_sorted': is_properly_sorted,
                        'unique_value_count': len(col_data.unique()),
                        'total_non_null_values': len(col_data)
                    }

                    temporal_analysis['temporal_value_columns'].append(temporal_column_info)

                    if not is_properly_sorted:
                        temporal_analysis['summary']['requires_sorting_attention'].append(str(col))

                    break  # Don't check other patterns for this column

        except Exception as e:
            logger.warning(f"Error analyzing temporal patterns in column {col}: {e}")
            continue

    # Check column names for temporal patterns
    for col in df.columns:
        col_str = str(col).strip()

        for pattern_type, pattern_info in temporal_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.match(pattern, col_str, re.IGNORECASE):
                    # Additional validation - make sure this isn't just a random number
                    if pattern_type == 'year_only' and len(col_str) == 4:
                        try:
                            year = int(col_str)
                            if year < 1900 or year > 2100:  # Reasonable year range
                                continue
                        except ValueError:
                            continue

                    temporal_column_info = {
                        'column': str(col),
                        'pattern_type': pattern_type,
                        'description': pattern_info['description'],
                        'column_name': col_str,
                        'detection_method': 'column_name',
                        'interpretation': f'Column represents data for {pattern_info["description"]}: {col_str}',
                        'granularity': pattern_type,
                        'data_type': str(df[col].dtype),
                        'non_null_count': int(df[col].count()),
                        'unique_value_count': int(df[col].nunique())
                    }

                    temporal_analysis['temporal_name_columns'].append(temporal_column_info)
                    break

    # Generate recommendations
    recommendations = []

    if temporal_analysis['temporal_value_columns']:
        recommendations.append(
            "TEMPORAL VALUE COLUMNS DETECTED: Use proper chronological sorting for time series analysis, "
            "forecasting, and trend visualization."
        )

        for col_info in temporal_analysis['temporal_value_columns']:
            if not col_info['is_properly_sorted']:
                recommendations.append(
                    f"Column '{col_info['column']}' ({col_info['description']}) may need chronological sorting. "
                    f"Guidance: {col_info['sort_guidance']}"
                )

    if temporal_analysis['temporal_name_columns']:
        recommendations.append(
            "TEMPORAL COLUMN NAMES DETECTED: These columns represent time period data. "
            "Consider reshaping for time series analysis if analyzing trends across these periods."
        )

    temporal_analysis['recommendations'] = recommendations

    # Update summary
    temporal_analysis['summary'].update({
        'total_temporal_columns': len(temporal_analysis['temporal_value_columns']) + len(
            temporal_analysis['temporal_name_columns']),
        'value_based_temporal': len(temporal_analysis['temporal_value_columns']),
        'name_based_temporal': len(temporal_analysis['temporal_name_columns'])
    })

    return temporal_analysis


def _check_temporal_sorting(values, pattern_type):
    """
    Check if temporal values appear to be in chronological order.
    This is a heuristic check - not perfect but gives guidance.
    """
    try:
        if len(values) < 3:  # Need at least 3 values to determine order
            return True

        # Convert to string and clean
        str_values = [str(v).strip() for v in values if str(v).strip()]

        if pattern_type == 'quarter':
            # Simple check: see if quarters progress logically
            quarters = []
            for val in str_values[:10]:  # Check first 10 values
                q_match = re.search(r'Q([1-4])', val, re.IGNORECASE)
                if q_match:
                    quarters.append(int(q_match.group(1)))

            if len(quarters) >= 3:
                # Check if quarters are in reasonable order (allowing for year changes)
                is_sorted = True
                for i in range(1, len(quarters)):
                    if quarters[i] < quarters[i - 1] and quarters[i] != 1:  # Q1 can follow Q4
                        is_sorted = False
                        break
                return is_sorted

        elif pattern_type == 'year_week':
            # Extract years and weeks
            year_weeks = []
            for val in str_values[:10]:
                # Try to extract year and week
                match = re.search(r'(\d{2,4})[-_]?W?(\d{1,2})', val)
                if match:
                    year = int(match.group(1))
                    if year < 100:  # Convert 2-digit year
                        year += 2000 if year < 50 else 1900
                    week = int(match.group(2))
                    year_weeks.append((year, week))

            if len(year_weeks) >= 3:
                # Check chronological order
                for i in range(1, len(year_weeks)):
                    curr_year, curr_week = year_weeks[i]
                    prev_year, prev_week = year_weeks[i - 1]

                    if curr_year < prev_year:
                        return False
                    elif curr_year == prev_year and curr_week < prev_week:
                        return False
                return True

        # For other types, do a basic alphabetical vs numeric check
        return True  # Default to assuming it's sorted

    except Exception:
        return True  # If we can't determine, assume it's okay


def analyze_data_standard(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standard data analysis function enhanced with temporal pattern detection
    """
    # Debug column information
    for i, col in enumerate(df.columns):
        logger.debug(f"Column {i}: {repr(col)} (type: {type(col)})")

    # Convert dtypes to string representation to avoid JSON serialization issues
    dtypes_dict = {}
    for col, dtype in df.dtypes.items():
        dtypes_dict[str(col)] = str(dtype)

    # Convert missing values to regular int
    missing_values = {}
    for col, count in df.isnull().sum().items():
        missing_values[str(col)] = int(count)

    # Get memory usage as regular int
    memory_usage = int(df.memory_usage(deep=True).sum())

    # Convert sample data safely
    sample_data = {}
    for col in df.columns:
        sample_data[col] = [convert_pandas_types(val) for val in df[col].head().tolist()]

    summary = {
        'shape': list(df.shape),
        'columns': [str(col) for col in df.columns],
        'dtypes': dtypes_dict,
        'missing_values': missing_values,
        'numeric_columns': [str(col) for col in df.select_dtypes(include=[np.number]).columns],
        'categorical_columns': [str(col) for col in df.select_dtypes(include=['object', 'string']).columns],
        'memory_usage': memory_usage,
        'sample_data': sample_data
    }

    # Add basic statistics for numeric columns
    summary['numeric_stats'] = {}
    if summary['numeric_columns']:
        try:
            stats_df = df[summary['numeric_columns']].describe()
            for col in stats_df.columns:
                numeric_stats[str(col)] = {}
                for stat in stats_df.index:
                    numeric_stats[col][stat] = convert_pandas_types(stats_df.loc[stat, col])
            summary['numeric_stats'] = numeric_stats
        except Exception as e:
            logger.warning(f"Could not generate numeric stats: {e}")
            summary['numeric_stats'] = {}

    # Add value counts for categorical columns 
    summary['categorical_stats'] = {}
    for col in summary['categorical_columns'][:50]:  # Limit to first 50 categorical columns
        try:
            value_counts = df[col].value_counts().head(50)  # Limit to top 50 unique values
            summary['categorical_stats'][str(col)] = {
                str(k): convert_pandas_types(v) for k, v in value_counts.to_dict().items()
            }
        except Exception as e:
            logger.warning(f"Could not generate categorical stats for {col}: {e}")
            summary['categorical_stats'][str(col)] = {}

    # NEW: Add temporal pattern detection
    try:
        temporal_analysis = detect_temporal_patterns(df)
        summary['temporal_columns_analysis'] = temporal_analysis

        if temporal_analysis['summary']['total_temporal_columns'] > 0:
            print(
                f"Temporal pattern detection: Found {temporal_analysis['summary']['total_temporal_columns']} columns with temporal characteristics")
            if temporal_analysis['summary']['requires_sorting_attention']:
                print(
                    f"  - Columns requiring sorting attention: {temporal_analysis['summary']['requires_sorting_attention']}")

    except Exception as e:
        logger.warning(f"Temporal pattern detection failed: {e}")
        summary['temporal_columns_analysis'] = {
            'temporal_value_columns': [],
            'temporal_name_columns': [],
            'recommendations': [],
            'summary': {'total_temporal_columns': 0, 'value_based_temporal': 0, 'name_based_temporal': 0}
        }

    # NEW: Add enhanced data profiling (keeping existing functionality)
    try:
        from enhanced_data_profiler import generate_enhanced_data_summary
        enhanced_summary = generate_enhanced_data_summary(df)
        # Merge enhanced analysis into the existing summary structure
        summary.update(enhanced_summary)
        summary['enhanced_profiling_available'] = True
        print("Enhanced data profiling completed successfully")
    except Exception as e:
        print(f"Enhanced profiling failed, using basic analysis: {e}")
        summary['enhanced_profiling_available'] = False

    return summary

def convert_pandas_types(obj):
    """Convert pandas types to JSON serializable types (existing function)"""
    # Handle strings first - they should NOT be converted to character lists
    if isinstance(obj, str):
        return obj

    # Handle DataFrames and Series first (before pd.isna check)
    if isinstance(obj, pd.DataFrame):
        try:
            return obj.to_dict('records')
        except:
            return str(obj)
    elif isinstance(obj, pd.Series):
        try:
            return obj.tolist()
        except:
            return str(obj)
    elif isinstance(obj, (pd.Index, pd.MultiIndex)):
        try:
            return obj.tolist()
        except:
            return str(obj)

    # Handle None and NaN values
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass

    # Handle array-like objects (but NOT strings)
    if hasattr(obj, '__len__') and len(obj) > 1 and not isinstance(obj, str):
        if isinstance(obj, dict):
            return {k: convert_pandas_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_pandas_types(v) for v in obj]
        else:
            try:
                return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
            except:
                return str(obj)

    # Handle scalar values with dtypes
    elif hasattr(obj, 'dtype'):
        if str(obj.dtype) in ['Int64', 'Int32', 'Int16', 'Int8']:
            return int(obj) if pd.notna(obj) else None
        elif str(obj.dtype) in ['Float64', 'Float32']:
            return float(obj) if pd.notna(obj) else None
        elif str(obj.dtype) == 'boolean':
            return bool(obj) if pd.notna(obj) else None
        elif str(obj.dtype) == 'string':
            return str(obj) if pd.notna(obj) else None
        elif pd.api.types.is_integer_dtype(obj.dtype):
            return int(obj) if pd.notna(obj) else None
        elif pd.api.types.is_float_dtype(obj.dtype):
            return float(obj) if pd.notna(obj) else None
        elif pd.api.types.is_bool_dtype(obj.dtype):
            return bool(obj) if pd.notna(obj) else None
        else:
            return str(obj) if pd.notna(obj) else None

    # Handle numpy types
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)

    # Handle remaining dictionaries and lists
    elif isinstance(obj, dict):
        return {k: convert_pandas_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_pandas_types(v) for v in obj]

    # Default case
    else:
        return obj

def create_enhanced_context_for_query(data_context: str, user_query: str, 
                                     template: TemplateDefinition = None,
                                     worksheet_data: Dict[str, pd.DataFrame] = None) -> str:
    """
    Create enhanced context for query processing, incorporating multi-worksheet template information
    """
    if template is None:
        # Return original context for non-template files
        return data_context
    
    # Determine data structure for context generation
    if worksheet_data and len(worksheet_data) > 1:
        # Multi-worksheet data
        data_source = worksheet_data
    elif worksheet_data and len(worksheet_data) == 1:
        # Single worksheet in dict format
        data_source = list(worksheet_data.values())[0]
    else:
        # Fallback - no specific data source
        data_source = None

    # Explicitly check for None instead of using ambiguous boolean evaluation
    context_data = data_source if data_source is not None else worksheet_data
    
    # Generate enhanced context using template
    enhanced_context = template_manager.generate_enhanced_context(
        context_data, # Pass the determined data structure
        template,
        data_context
    )
    
    # Add query-specific guidance based on template and data structure
    query_guidance = generate_query_specific_guidance(user_query, template, worksheet_data)
    if query_guidance:
        enhanced_context += f"\n\n QUERY-SPECIFIC GUIDANCE:\n{query_guidance}"
    
    return enhanced_context

def generate_query_specific_guidance(user_query: str, template: TemplateDefinition, 
                                   worksheet_data: Dict[str, pd.DataFrame] = None) -> str:
    """Generate specific guidance based on the user query, template, and data structure"""
    query_lower = user_query.lower()
    guidance_parts = []
    
    # Multi-worksheet specific guidance
    if template.template_type in ['multi_worksheet', 'csv_collection'] and worksheet_data and len(worksheet_data) > 1:
        
        # Check for cross-worksheet queries
        cross_worksheet_keywords = ['across', 'between', 'join', 'merge', 'combine', 'all worksheets', 'all sheets']
        if any(keyword in query_lower for keyword in cross_worksheet_keywords):
            guidance_parts.append(" Cross-Worksheet Analysis Detected:")
            guidance_parts.append(f"   - Use merge_worksheets(worksheet_data) to combine all worksheets")
            guidance_parts.append(f"   - Available worksheets: {', '.join(worksheet_data.keys())}")
            if template.worksheet_relationships:
                guidance_parts.append(f"   - Consider these relationships when joining:")
                for rel in template.worksheet_relationships:
                    guidance_parts.append(f"     - {rel.name}: {rel.from_worksheet}.{rel.from_column} → {rel.to_worksheet}.{rel.to_column}")
        
        # Check for worksheet-specific queries
        worksheet_names = list(worksheet_data.keys())
        mentioned_worksheets = [ws for ws in worksheet_names if ws.lower() in query_lower]
        if mentioned_worksheets:
            guidance_parts.append(f" Worksheet-Specific Analysis:")
            guidance_parts.append(f"   - Mentioned worksheets: {', '.join(mentioned_worksheets)}")
            guidance_parts.append(f"   - Use worksheet_data['{mentioned_worksheets[0]}'] to access specific data")
    
    # Check for metric-related queries
    for metric in template.metrics:
        metric_keywords = metric.name.lower().split()
        if any(keyword in query_lower for keyword in metric_keywords):
            guidance_parts.append(f" Template Metric: {metric.name}")
            guidance_parts.append(f"   Description: {metric.description}")
            guidance_parts.append(f"   Formula: {metric.formula}")
            guidance_parts.append(f"   Category: {metric.category}")
            if metric.worksheets_involved:
                guidance_parts.append(f"   Worksheets: {', '.join(metric.worksheets_involved)}")
    
    # Check for visualization-related queries
    viz_keywords = ['chart', 'plot', 'graph', 'visualize', 'show', 'display']
    if any(keyword in query_lower for keyword in viz_keywords):
        guidance_parts.append(" Template Visualization Suggestions:")
        relevant_viz = []
        
        for viz in template.visualizations:
            # Check if visualization is relevant to query
            viz_keywords_in_query = any(word in query_lower for word in viz.name.lower().split())
            if viz_keywords_in_query or len(relevant_viz) < 3:  # Show top 3 if no specific match
                relevant_viz.append(viz)
        
        for viz in relevant_viz[:3]:  # Limit to 3 suggestions
            guidance_parts.append(f"   - {viz.name}: {viz.description}")
            guidance_parts.append(f"     Chart Type: {viz.chart_type}, X: {viz.x_axis}, Y: {viz.y_axis}")
            if viz.worksheets_involved:
                guidance_parts.append(f"     Worksheets: {', '.join(viz.worksheets_involved)}")
    
    # Check for common analysis patterns
    for analysis in template.common_analyses:
        analysis_keywords = analysis.lower().split()[:3]  # First 3 words
        if any(keyword in query_lower for keyword in analysis_keywords):
            guidance_parts.append(f" Template Analysis Pattern: {analysis}")
            break
    
    return "\n".join(guidance_parts) if guidance_parts else ""

def get_template_suggestions_for_data(data_source: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                                    filename: str = None) -> list:
    """Get template suggestions for uploaded data (multi-worksheet aware)"""
    suggestions = template_manager.get_template_suggestions(data_source)
    return [{"name": name, "score": score} for name, score in suggestions]

def apply_template_manually(data_source: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                          template_id: str) -> Tuple[Dict[str, Any], TemplateDefinition]:
    """Manually apply a template to data (multi-worksheet aware)"""
    template = template_manager.get_template_by_id(template_id)
    if not template:
        raise ValueError(f"Template {template_id} not found")
    
    # Get standard analysis based on data type
    if isinstance(data_source, pd.DataFrame):
        standard_summary = analyze_data_standard(data_source)
        is_multi = False
    else:
        # Multi-worksheet - analyze primary worksheet
        main_data = list(data_source.values())[0] if data_source else pd.DataFrame()
        standard_summary = analyze_data_standard(main_data)
        is_multi = True
        
        # Add multi-worksheet info
        standard_summary['multi_worksheet_info'] = {
            'total_worksheets': len(data_source),
            'worksheet_names': list(data_source.keys()),
            'worksheet_shapes': {name: list(df.shape) for name, df in data_source.items()},
            'is_multi_worksheet': True
        }
    
    # Add template info
    standard_summary['template_info'] = {
        'template_id': template.id,
        'template_name': template.name,
        'template_domain': template.domain,
        'template_version': template.version,
        'template_type': template.template_type,
        'enhanced_context_available': True,
        'manually_applied': True,
        'multi_worksheet_template': template.template_type in ['multi_worksheet', 'csv_collection']
    }
    
    # Add worksheet mapping for multi-worksheet templates
    if is_multi and template.template_type in ['multi_worksheet', 'csv_collection']:
        standard_summary['template_info']['worksheet_mapping'] = _map_worksheets_to_template(
            data_source, template
        )
    
    return standard_summary, template

def validate_data_against_template(data_source: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                                 template: TemplateDefinition) -> Dict[str, Any]:
    """Validate uploaded data against template expectations (multi-worksheet aware)"""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'worksheet_validations': []
    }
    
    # Determine data structure
    if isinstance(data_source, pd.DataFrame):
        worksheet_data = {"Main": data_source}
    else:
        worksheet_data = data_source
    
    # Validate each template worksheet
    for template_ws in template.worksheets:
        ws_validation = {
            'worksheet_name': template_ws.name,
            'is_required': template_ws.is_required,
            'found': False,
            'matched_worksheet': None,
            'missing_columns': [],
            'unexpected_columns': [],
            'data_quality_issues': [],
            'warnings': []
        }
        
        # Find best matching worksheet
        best_match = None
        best_score = 0.0
        
        for actual_name, actual_df in worksheet_data.items():
            score = template_manager._calculate_worksheet_match_score(actual_df, template_ws, actual_name)
            if score > best_score:
                best_match = actual_name
                best_score = score
        
        if best_match and best_score > 0.5:  # Reasonable match found
            ws_validation['found'] = True
            ws_validation['matched_worksheet'] = best_match
            df = worksheet_data[best_match]
            
            # Validate columns for this worksheet
            template_columns = {col.name for col in template_ws.columns}
            data_columns = set(df.columns)
            
            missing_cols = template_columns - data_columns
            unexpected_cols = data_columns - template_columns
            
            if missing_cols:
                ws_validation['missing_columns'] = list(missing_cols)
                ws_validation['warnings'].append(f"Missing expected columns: {', '.join(missing_cols)}")
            
            if unexpected_cols:
                ws_validation['unexpected_columns'] = list(unexpected_cols)
                ws_validation['warnings'].append(f"Unexpected columns found: {', '.join(unexpected_cols)}")
            
            # Validate data quality for existing columns
            for col_def in template_ws.columns:
                if col_def.name in df.columns:
                    col_data = df[col_def.name]
                    
                    # Check data type expectations
                    if col_def.data_type == 'numeric' and not pd.api.types.is_numeric_dtype(col_data):
                        ws_validation['data_quality_issues'].append(
                            f"Column '{col_def.name}' expected to be numeric but found {col_data.dtype}"
                        )
                    
                    # Check valid ranges for numeric columns
                    if col_def.valid_range and pd.api.types.is_numeric_dtype(col_data):
                        min_val, max_val = col_def.valid_range
                        out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
                        if not out_of_range.empty:
                            ws_validation['data_quality_issues'].append(
                                f"Column '{col_def.name}' has {len(out_of_range)} values outside expected range [{min_val}, {max_val}]"
                            )
                    
                    # Check expected values for categorical columns
                    if col_def.expected_values:
                        unexpected_values = set(col_data.unique()) - set(col_def.expected_values)
                        if unexpected_values:
                            ws_validation['warnings'].append(
                                f"Column '{col_def.name}' has unexpected values: {', '.join(map(str, list(unexpected_values)[:5]))}"
                            )
        
        else:
            # Worksheet not found or poor match
            if template_ws.is_required:
                ws_validation['warnings'].append(f"Required worksheet '{template_ws.name}' not found or poorly matched")
                validation_results['errors'].append(f"Required worksheet '{template_ws.name}' missing")
            else:
                ws_validation['warnings'].append(f"Optional worksheet '{template_ws.name}' not found")
        
        validation_results['worksheet_validations'].append(ws_validation)
        
        # Aggregate warnings and errors
        validation_results['warnings'].extend(ws_validation['warnings'])
        if ws_validation['data_quality_issues']:
            validation_results['warnings'].extend(ws_validation['data_quality_issues'])
    
    # Determine overall validity
    if validation_results['errors']:
        validation_results['is_valid'] = False
    
    # Add summary
    total_worksheets = len(template.worksheets)
    found_worksheets = sum(1 for ws in validation_results['worksheet_validations'] if ws['found'])
    required_worksheets = sum(1 for ws in template.worksheets if ws.is_required)
    found_required = sum(1 for ws in validation_results['worksheet_validations'] 
                        if ws['found'] and ws['is_required'])
    
    validation_results['summary'] = {
        'total_template_worksheets': total_worksheets,
        'found_worksheets': found_worksheets,
        'required_worksheets': required_worksheets,
        'found_required_worksheets': found_required,
        'match_percentage': (found_worksheets / total_worksheets * 100) if total_worksheets > 0 else 0
    }
    
    return validation_results