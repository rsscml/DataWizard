"""
Enhanced Data Profiler for DataAnalysisAgent - ROOT CAUSE FIXES + SPECIAL NUMERIC DETECTION
Provides comprehensive data profiling to give LLM rich context about data structure, quality, and relationships.
Includes detection of special numeric columns that require careful handling (percentages, ratios, etc.)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def convert_to_json_serializable(obj):
    """Convert pandas/numpy types to JSON serializable types"""
    if isinstance(obj, str):
        return obj

    # Handle collections FIRST - before pd.isna() check
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (pd.Series, pd.Index)):
        return [convert_to_json_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')

    # Handle pandas/numpy scalar types
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)

    # Now it's safe to check pd.isna() on scalar values
    elif pd.isna(obj):
        return None

    # Handle pandas extension dtypes
    elif hasattr(obj, 'dtype'):
        if str(obj.dtype) in ['Int64', 'Int32', 'Int16', 'Int8']:
            return int(obj) if pd.notna(obj) else None
        elif str(obj.dtype) in ['Float64', 'Float32']:
            return float(obj) if pd.notna(obj) else None
        elif str(obj.dtype) == 'boolean':
            return bool(obj) if pd.notna(obj) else None
        elif str(obj.dtype) == 'string':
            return str(obj) if pd.notna(obj) else None

    # Default case - try conversion or return as string
    try:
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj
    except:
        return str(obj)

def safe_scalar(value):
    """Ensure a value is a Python scalar, not a pandas/numpy object"""
    if hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (pd.Series, pd.Index)) and len(value) == 1:
        return value.iloc[0]
    elif isinstance(value, (np.ndarray)) and value.size == 1:
        return value.item()
    else:
        return value

def safe_bool_check(condition):
    """Safely check boolean condition that might be a pandas Series"""
    if isinstance(condition, (pd.Series, np.ndarray)):
        return condition.any() if len(condition) > 0 else False
    else:
        return bool(condition)

def detect_special_numeric_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect numeric columns that require special handling for calculations/aggregations.
    
    Identifies columns containing percentages, ratios, currencies, coordinates, etc.
    that cannot be treated like regular numeric data for aggregation purposes.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict containing categorized special numeric columns with operation guidance
    """
    special_columns = {
        'percentage_columns': [],
        'ratio_columns': [],
        'currency_columns': [],
        'coordinate_columns': [],
        'year_columns': [],
        'id_like_columns': [],
        'score_columns': [],
        'index_columns': [],
        'normalized_value_columns': [],
        'rate_columns': [],
        'summary': {
            'total_special_columns': 0,
            'requires_careful_aggregation': [],
            'avoid_simple_arithmetic': [],
            'categories_found': []
        }
    }
    
    try:
        # Get numeric columns only (including object columns that might contain numeric data with symbols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Check object columns for numeric data with special symbols (like %)
        potential_numeric_object_cols = []
        for col in object_cols:
            try:
                sample_data = df[col].dropna().head(20)
                if len(sample_data) > 0:
                    # Check if most values look like numbers with symbols
                    numeric_with_symbol_count = 0
                    for val in sample_data:
                        val_str = str(val).strip()
                        # Check for percentage, currency, or other numeric patterns
                        if (re.match(r'^[\d.,]+%$', val_str) or  # 25.5%
                            re.match(r'^\$[\d.,]+$', val_str) or  # $1,000
                            re.match(r'^[\d.,]+\$?$', val_str)):  # 1000$ or just numbers
                            numeric_with_symbol_count += 1
                    
                    if numeric_with_symbol_count / len(sample_data) > 0.5:  # More than 50% match pattern
                        potential_numeric_object_cols.append(col)
            except:
                continue
        
        all_cols_to_check = numeric_cols + potential_numeric_object_cols
        
        for col in all_cols_to_check:
            try:
                col_name = str(col).lower()
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                column_info = {
                    'column': str(col),
                    'detection_reasons': [],
                    'safe_operations': [],
                    'avoid_operations': [],
                    'special_considerations': [],
                    'data_type': str(df[col].dtype)
                }
                
                # 1. PERCENTAGE DETECTION
                percentage_indicators = ['%', 'percent', 'percentage', 'pct', 'perc', 'bps', 'market value share',
                                         'market volume share', 'market value growth', 'market volume growth', 'uvg',
                                         'upg', 'usg', 'underlying operating margin', 'tdp share',
                                         'tbca', 'dbca', 'attractive pack', 'attractive_pack' ,'brand power score',
                                         'brand_power_score', 'slp - absolute', 'sustainable living purpose',
                                         'msval', 'msvol', 'relative penetration', 'penetration' ,'market share by price tier',
                                         'market_share_by_price_tier', 'better quality', 'volume share', 'value share']
                
                # Check column name
                name_has_percentage = any(indicator in col_name for indicator in percentage_indicators)
                
                # Check for percentage values (% symbol in data)
                values_have_percentage = False
                if col in potential_numeric_object_cols:
                    sample_str_values = col_data.astype(str).head(50)
                    percentage_count = sum(1 for val in sample_str_values if '%' in str(val))
                    values_have_percentage = percentage_count > len(sample_str_values) * 0.3  # 30% threshold

                if name_has_percentage or values_have_percentage:
                    if name_has_percentage:
                        column_info['detection_reasons'].append('column name contains percentage term or matches KPI names which are measured in percentages')
                    if values_have_percentage:
                        column_info['detection_reasons'].append('values_contain_percentage_symbol')

                    column_info['safe_operations'] = ['count', 'mean', 'min', 'max', 'median', 'mode', 'value_counts']
                    column_info['avoid_operations'] = ['sum', 'add', 'subtract']
                    column_info['special_considerations'] = [
                        'Use weighted averages instead of simple mean where possible',
                        'Consider if values are in percentage format (0-100) or decimal (0-1)',
                        'Percentage changes require special calculation methods'
                    ]
                    special_columns['percentage_columns'].append(column_info.copy())
                    continue
                
                # Skip further checks for object columns that aren't percentage
                if col in potential_numeric_object_cols:
                    continue

                # 2. INDEX DETECTION
                index_indicators = ['sos/som', 'index', '_index', 'indicator', 'average price index', 'perceived price']
                if any(indicator in col_name for indicator in index_indicators):
                    column_info['detection_reasons'].append('appears to be index or indicator')
                    column_info['safe_operations'] = ['count', 'mean', 'median', 'min', 'max', 'trend_analysis']
                    column_info['avoid_operations'] = ['sum', 'add', 'subtract']
                    column_info['special_considerations'] = [
                            'Averaging indices may require weighting by underlying values',
                            'Focus on percentage changes rather than absolute values']
                    special_columns['index_columns'].append(column_info.copy())
                    continue

                # 3. RATE DETECTION
                rate_indicators = ['_rate', 'price', 'frequency', 'per_second', 'per_minute', 'per_hour', 'per_day']
                if any(indicator in col_name for indicator in rate_indicators):
                    column_info['detection_reasons'].append('column name indicates rate measurement')
                    column_info['safe_operations'] = ['count', 'min', 'max', 'median', 'mean', 'harmonic_mean', 'weighted_average']
                    column_info['avoid_operations'] = ['simple_sum', 'arithmetic_mean']
                    column_info['special_considerations'] = [
                        'Rates often require harmonic mean rather than arithmetic mean',
                        'Sum of rates is usually not meaningful',
                        'Consider time-weighting for temporal rates'
                    ]
                    special_columns['rate_columns'].append(column_info.copy())
                    continue
                

                # 4. COORDINATE DETECTION
                coordinate_indicators = ['lat', 'lng', 'long', 'latitude', 'longitude', 'coord', 'x_coord', 'y_coord']
                if any(indicator in col_name for indicator in coordinate_indicators):
                    # Additional validation for coordinate ranges
                    try:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        
                        is_latitude = ('lat' in col_name and min_val >= -90 and max_val <= 90)
                        is_longitude = (('lng' in col_name or 'long' in col_name) and min_val >= -180 and max_val <= 180)
                        is_general_coord = 'coord' in col_name
                        
                        if is_latitude or is_longitude or is_general_coord:
                            coord_type = 'latitude' if is_latitude else 'longitude' if is_longitude else 'coordinate'
                            column_info['detection_reasons'].append(f'geographic_{coord_type}_pattern')
                            column_info['safe_operations'] = ['count', 'min', 'max', 'range']
                            column_info['avoid_operations'] = ['sum', 'mean']
                            column_info['special_considerations'] = [
                                'Averaging coordinates may not represent meaningful geographic location',
                                'Use centroid calculation for geographic center',
                                'Consider map projection system for distance calculations',
                                'Coordinate operations require geographic context'
                            ]
                            special_columns['coordinate_columns'].append(column_info.copy())
                            continue
                    except:
                        pass
                
                # 5. YEAR DETECTION
                year_indicators = ['year', 'yr', 'yyyy']
                if any(indicator in col_name for indicator in year_indicators):
                    # Check if values look like years (4-digit numbers in reasonable range)
                    try:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        
                        if (min_val >= 1900 and max_val <= 2100 and
                            all(val == int(val) for val in col_data.head(20) if pd.notna(val))):
                            column_info['detection_reasons'].append('appears_to_be_year_values')
                            column_info['safe_operations'] = ['count', 'min', 'max', 'mode', 'range']
                            column_info['avoid_operations'] = ['sum', 'mean']
                            column_info['special_considerations'] = [
                                'Averaging years rarely provides meaningful insight',
                                'Use median year, mode, or year ranges instead',
                                'Consider converting to age or time periods for analysis',
                                'Year differences might be more meaningful than year averages'
                            ]
                            special_columns['year_columns'].append(column_info.copy())
                            continue
                    except:
                        pass
                
                # 6. ID-LIKE NUMERIC COLUMNS
                id_indicators = ['id', 'key', 'code', 'number', 'num', '#', 'ref', 'seq']
                if any(indicator in col_name for indicator in id_indicators):
                    # Check if this looks like an identifier (high uniqueness)
                    try:
                        uniqueness_ratio = col_data.nunique() / len(col_data)
                        if uniqueness_ratio > 0.8:  # High uniqueness suggests ID
                            column_info['detection_reasons'].append('high_uniqueness_suggests_identifier')
                            column_info['safe_operations'] = ['count', 'nunique', 'value_counts']
                            column_info['avoid_operations'] = ['sum', 'mean', 'min', 'max', 'all_arithmetic']
                            column_info['special_considerations'] = [
                                'This appears to be an identifier, not a measurable quantity',
                                'Only count and uniqueness operations are meaningful',
                                'Avoid all arithmetic operations - they have no business meaning',
                                'Use for grouping and joining operations instead'
                            ]
                            special_columns['id_like_columns'].append(column_info.copy())
                            continue
                    except:
                        pass
                
                # 7. RANK DETECTION
                score_indicators = ['grade', 'rating', 'rank']
                if any(indicator in col_name for indicator in score_indicators):
                    column_info['detection_reasons'].append('appears_to_be_score_rating_or_grade')
                    column_info['safe_operations'] = ['count', 'mean', 'median', 'min', 'max', 'mode']
                    column_info['avoid_operations'] = ['sum']
                    column_info['special_considerations'] = [
                        'Scores may need weighted averages based on different sample sizes',
                        'Consider if scale is ordinal (rankings) vs interval (test scores)',
                        'Sum of scores rarely meaningful unless representing total points',
                        'Be aware of different scoring scales when combining'
                    ]
                    special_columns['score_columns'].append(column_info.copy())
                    continue

            except Exception as e:
                continue
        
        # Calculate summary statistics
        all_special_columns = []
        categories_found = []
        
        for category, columns in special_columns.items():
            if isinstance(columns, list) and len(columns) > 0:
                category_name = category.replace('_columns', '').replace('_', ' ').title()
                categories_found.append(category_name)
                all_special_columns.extend([col_info['column'] for col_info in columns])
        
        special_columns['summary']['total_special_columns'] = len(all_special_columns)
        special_columns['summary']['categories_found'] = categories_found
        
        # Identify columns that require careful aggregation
        careful_aggregation = []
        avoid_arithmetic = []
        
        for category, columns in special_columns.items():
            if isinstance(columns, list):
                for col_info in columns:
                    col_name = col_info['column']
                    avoid_ops = col_info.get('avoid_operations', [])
                    
                    if any(op in avoid_ops for op in ['sum', 'simple_mean', 'mean', 'simple_sum']):
                        careful_aggregation.append(col_name)
                    
                    if len(avoid_ops) >= 3 or 'all_arithmetic' in avoid_ops:
                        avoid_arithmetic.append(col_name)
        
        special_columns['summary']['requires_careful_aggregation'] = list(set(careful_aggregation))
        special_columns['summary']['avoid_simple_arithmetic'] = list(set(avoid_arithmetic))
        
        # Log detection results
        if special_columns['summary']['total_special_columns'] > 0:
            print(f"Detected {special_columns['summary']['total_special_columns']} special numeric columns requiring careful handling")
            for category in categories_found:
                category_key = category.lower().replace(' ', '_') + '_columns'
                count = len(special_columns.get(category_key, []))
                print(f"  - {category}: {count} columns")
        else:
            print("No special numeric columns detected - all numeric columns can use standard operations")
    
    except Exception as e:
        print(f"Warning in detect_special_numeric_columns: {e}")
        # Return empty structure on error
        special_columns = {
            'percentage_columns': [], 'ratio_columns': [], 'currency_columns': [],
            'coordinate_columns': [], 'year_columns': [], 'id_like_columns': [],
            'score_columns': [], 'index_columns': [], 'normalized_value_columns': [],
            'rate_columns': [],
            'summary': {
                'total_special_columns': 0, 'requires_careful_aggregation': [],
                'avoid_simple_arithmetic': [], 'categories_found': []
            }
        }
    
    return convert_to_json_serializable(special_columns)

def detect_long_text_columns(df: pd.DataFrame, min_avg_length: int = 100, min_sample_size: int = 10) -> List[str]:
    """
    Detect columns that contain long free-form text (paragraphs, multiple lines)
    These columns should be excluded from most analyses as they're not categorical data
    """
    long_text_columns = []
    
    # Only check string/object columns
    text_columns = df.select_dtypes(include=['object', 'string']).columns
    
    for col in text_columns:
        try:
            col_data = df[col].dropna().astype(str)
            
            if len(col_data) < min_sample_size:
                continue
            
            # Calculate average text length
            text_lengths = col_data.str.len()
            avg_length = text_lengths.mean()
            
            # Check for indicators of long free-form text
            indicators = {
                'avg_length': avg_length,
                'has_newlines': False,
                'has_multiple_sentences': False,
                'high_uniqueness_with_long_text': False
            }
            
            if avg_length >= min_avg_length:
                # Check for newlines (multi-line text)
                newline_count = col_data.str.contains('\n|\\n', regex=True, na=False).sum()
                if newline_count > len(col_data) * 0.1:  # 10% have newlines
                    indicators['has_newlines'] = True
                
                # Check for multiple sentences (look for multiple periods, exclamation marks, question marks)
                sentence_pattern = r'[.!?]\s+[A-Z]'
                multiple_sentences = col_data.str.contains(sentence_pattern, regex=True, na=False).sum()
                if multiple_sentences > len(col_data) * 0.2:  # 20% have multiple sentences
                    indicators['has_multiple_sentences'] = True
                
                # Check for high uniqueness combined with long text (indicates unique content, not categories)
                uniqueness_ratio = col_data.nunique() / len(col_data)
                if uniqueness_ratio > 0.8 and avg_length > 50:
                    indicators['high_uniqueness_with_long_text'] = True
                
                # Determine if this is a long text column
                is_long_text = (
                    avg_length >= min_avg_length * 2 or  # Very long average length
                    indicators['has_newlines'] or  # Multi-line text
                    indicators['has_multiple_sentences'] or  # Multiple sentences
                    indicators['high_uniqueness_with_long_text']  # Unique long content
                )
                
                if is_long_text:
                    long_text_columns.append(col)
                    print(f"Detected long text column '{col}': avg_length={avg_length:.1f}, "
                          f"newlines={indicators['has_newlines']}, "
                          f"multi_sentences={indicators['has_multiple_sentences']}, "
                          f"high_unique_long={indicators['high_uniqueness_with_long_text']}")
        
        except Exception as e:
            continue
    
    return long_text_columns


def detect_column_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Data-driven column classification based on statistical patterns and behaviors"""
    patterns = {
        'naming_conventions': {},
        'data_format_patterns': {},
        'column_groups': {},
        'potential_identifiers': [],
        'dimensions': [],
        'measures': []
    }

    try:
        # Keep existing naming conventions and grouping logic (useful metadata)
        column_names = df.columns.tolist()

        snake_case_count = sum(1 for col in column_names if '_' in str(col) and str(col).islower())
        camel_case_count = sum(1 for col in column_names if re.match(r'^[a-z]+([A-Z][a-z]*)*$', str(col)))
        space_separated = sum(1 for col in column_names if ' ' in str(col))

        total_cols = len(column_names)
        patterns['naming_conventions'] = {
            'snake_case_ratio': snake_case_count / total_cols if total_cols > 0 else 0,
            'camel_case_ratio': camel_case_count / total_cols if total_cols > 0 else 0,
            'space_separated_ratio': space_separated / total_cols if total_cols > 0 else 0,
            'dominant_style': 'snake_case' if snake_case_count > total_cols / 2 else
            'camel_case' if camel_case_count > total_cols / 2 else
            'space_separated' if space_separated > total_cols / 2 else 'mixed'
        }

        # Column grouping by naming patterns
        column_prefixes = {}
        column_suffixes = {}

        for col in column_names:
            try:
                col_str = str(col).lower()
                if '_' in col_str:
                    prefix = col_str.split('_')[0]
                    column_prefixes[prefix] = column_prefixes.get(prefix, []) + [col]
                    suffix = col_str.split('_')[-1]
                    column_suffixes[suffix] = column_suffixes.get(suffix, []) + [col]
                elif ' ' in col_str:
                    prefix = col_str.split(' ')[0]
                    column_prefixes[prefix] = column_prefixes.get(prefix, []) + [col]
            except:
                continue

        patterns['column_groups']['by_prefix'] = {k: v for k, v in column_prefixes.items() if len(v) >= 2}
        patterns['column_groups']['by_suffix'] = {k: v for k, v in column_suffixes.items() if len(v) >= 2}

        # DATA-DRIVEN ANALYSIS STARTS HERE
        # First, collect basic stats for all columns
        column_stats = {}
        numeric_columns = []
        categorical_columns = []

        for col in df.columns:
            try:
                col_dtype = str(df[col].dtype).lower()

                # Basic statistics
                unique_count = int(df[col].nunique())
                total_count = int(len(df))
                null_count = int(df[col].isnull().sum())
                uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
                null_ratio = null_count / total_count if total_count > 0 else 0

                stats = {
                    'dtype': col_dtype,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'null_count': null_count,
                    'uniqueness_ratio': uniqueness_ratio,
                    'null_ratio': null_ratio,
                    'is_numeric': any(num_type in col_dtype for num_type in ['int', 'float']),
                    'is_categorical': any(cat_type in col_dtype for cat_type in ['object', 'string', 'category'])
                }

                # Analyze data patterns for non-null values
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # String pattern analysis for categorical columns
                    if stats['is_categorical']:
                        col_str_data = col_data.astype(str)

                        # Length consistency (important for IDs)
                        lengths = [len(str(val)) for val in col_str_data]
                        if lengths:
                            stats['avg_length'] = sum(lengths) / len(lengths)
                            stats['length_variance'] = sum((l - stats['avg_length']) ** 2 for l in lengths) / len(
                                lengths)
                            stats['length_consistency'] = 1 / (1 + stats['length_variance'])  # Higher = more consistent

                        # Character pattern analysis
                        alphanumeric_count = sum(1 for val in col_str_data if re.match(r'^[A-Za-z0-9_-]+$', str(val)))
                        stats['alphanumeric_ratio'] = alphanumeric_count / len(col_str_data)

                        # Word count (to detect free-form text)
                        word_counts = [len(str(val).split()) for val in col_str_data]
                        stats['avg_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0

                        # Character diversity (identifiers often have consistent character patterns)
                        all_chars = ''.join(col_str_data.astype(str))
                        unique_chars = len(set(all_chars))
                        stats['char_diversity'] = unique_chars / len(all_chars) if len(all_chars) > 0 else 0

                    # Numeric pattern analysis
                    elif stats['is_numeric']:
                        stats['min_value'] = float(col_data.min())
                        stats['max_value'] = float(col_data.max())
                        stats['mean_value'] = float(col_data.mean())
                        stats['median_value'] = float(col_data.median())
                        stats['std_value'] = float(col_data.std())
                        stats['range_value'] = stats['max_value'] - stats['min_value']

                        # Check for sequential patterns (common in IDs)
                        if len(col_data) > 1 and col_data.dtype in ['int64', 'int32', 'int16', 'int8']:
                            sorted_values = col_data.sort_values().reset_index(drop=True)
                            diffs = sorted_values.diff().dropna()
                            if len(diffs) > 0:
                                # Sequential if most differences are 1
                                sequential_ratio = (diffs == 1).sum() / len(diffs)
                                stats['sequential_ratio'] = float(sequential_ratio)

                                # Monotonic if always increasing
                                stats['is_monotonic'] = bool((diffs > 0).all())

                        # Distribution analysis
                        if stats['std_value'] > 0:
                            stats['coefficient_of_variation'] = stats['std_value'] / abs(stats['mean_value']) if stats[
                                                                                                                     'mean_value'] != 0 else float(
                                'inf')
                            stats['skewness'] = float(col_data.skew())

                column_stats[col] = stats

                if stats['is_numeric']:
                    numeric_columns.append(col)
                elif stats['is_categorical']:
                    categorical_columns.append(col)

            except Exception as e:
                continue

        # IDENTIFIER DETECTION (Data-driven)
        for col, stats in column_stats.items():
            try:
                identifier_score = 0
                identifier_reasons = []

                # High uniqueness suggests potential identifier
                if stats['uniqueness_ratio'] >= 0.97:
                    identifier_score += 0.5
                    identifier_reasons.append('very_high_uniqueness')
                elif stats['uniqueness_ratio'] >= 0.95:
                    identifier_score += 0.3
                    identifier_reasons.append('high_uniqueness')
                elif stats['uniqueness_ratio'] >= 0.9 and stats['total_count'] > 100:
                    identifier_score += 0.2
                    identifier_reasons.append('high_uniqueness_large_dataset')

                # Low null rate is good for identifiers
                if stats['null_ratio'] == 0:
                    identifier_score += 0.1
                    identifier_reasons.append('no_nulls')
                elif stats['null_ratio'] < 0.05:
                    identifier_score += 0.05
                    identifier_reasons.append('few_nulls')

                # For categorical columns: consistent length and clean format suggests ID
                if stats['is_categorical']:
                    if stats.get('length_consistency', 0) > 0.9:
                        identifier_score += 0.1
                        identifier_reasons.append('consistent_length')

                    if stats.get('alphanumeric_ratio', 0) > 0.9:
                        identifier_score += 0.1
                        identifier_reasons.append('clean_format')

                    # Low word count (not descriptive text)
                    if stats.get('avg_word_count', 0) <= 1.0:
                        identifier_score += 0.1
                        identifier_reasons.append('single_token')

                # For numeric columns: sequential or monotonic patterns suggest ID
                if stats['is_numeric']:
                    if stats.get('sequential_ratio', 0) > 0.8:
                        identifier_score += 0.3
                        identifier_reasons.append('sequential_pattern')
                    elif stats.get('is_monotonic', False):
                        identifier_score += 0.2
                        identifier_reasons.append('monotonic_pattern')

                    # Integer types more likely to be IDs than floats
                    if 'int' in stats['dtype']:
                        identifier_score += 0.1
                        identifier_reasons.append('integer_type')

                # Threshold for identifier classification
                if identifier_score >= 0.5:
                    confidence = 'high' if identifier_score > 0.7 else 'medium' if identifier_score > 0.6 else 'low'

                    # Determine identifier type based on patterns
                    if stats['is_numeric'] and stats.get('sequential_ratio', 0) > 0.5:
                        id_type = 'sequential_numeric'
                    elif stats['is_categorical'] and stats.get('alphanumeric_ratio', 0) > 0.9:
                        id_type = 'structured_code'
                    elif stats.get('avg_length', 0) > 20:
                        id_type = 'long_identifier'
                    else:
                        id_type = 'generic'

                    patterns['potential_identifiers'].append({
                        'column': str(col),
                        'identifier_score': float(identifier_score),
                        'confidence': confidence,
                        'identifier_type': id_type,
                        'uniqueness_ratio': float(stats['uniqueness_ratio']),
                        'data_type': stats['dtype'],
                        'reasons': identifier_reasons
                    })

            except Exception as e:
                continue

        # DIMENSION DETECTION (Data-driven)
        # Exclude already identified identifiers
        identifier_columns = [id_info['column'] for id_info in patterns['potential_identifiers']]

        for col, stats in column_stats.items():
            try:
                if col in identifier_columns or not stats['is_categorical']:
                    continue

                dimension_score = 0
                dimension_reasons = []

                # Good cardinality range for dimensions (not too few, not too many)
                if 2 <= stats['unique_count'] <= 1000:
                    if stats['unique_count'] <= 20:
                        dimension_score += 0.4
                        dimension_reasons.append('very_low_cardinality')
                    elif stats['unique_count'] <= 100:
                        dimension_score += 0.3
                        dimension_reasons.append('low_cardinality')
                    else:
                        dimension_score += 0.2
                        dimension_reasons.append('high_cardinality')
                else:
                    continue  # Skip if cardinality is outside reasonable range

                # Moderate uniqueness (values repeat, good for grouping)
                if stats['uniqueness_ratio'] <= 0.5:
                    dimension_score += 0.3
                    dimension_reasons.append('values_repeat')
                elif stats['uniqueness_ratio'] <= 0.7:
                    dimension_score += 0.2
                    dimension_reasons.append('some_repetition')

                # Not too many nulls
                if stats['null_ratio'] < 0.3:
                    dimension_score += 0.2
                    dimension_reasons.append('low_nulls')
                elif stats['null_ratio'] < 0.5:
                    dimension_score += 0.1
                    dimension_reasons.append('moderate_nulls')

                # Single words/tokens are better dimensions than long text
                if stats.get('avg_word_count', 0) <= 2:
                    dimension_score += 0.1
                    dimension_reasons.append('concise_values')
                elif stats.get('avg_word_count', 0) > 5:
                    dimension_score -= 0.3  # Penalty for long text
                    dimension_reasons.append('verbose_values_penalty')

                # Check if this column creates meaningful groups when used with numeric columns
                # (This is a key insight: good dimensions create groups with different measure values)
                if numeric_columns:
                    try:
                        # Sample a numeric column to test grouping effectiveness
                        test_numeric_col = numeric_columns[0]
                        grouped = df.groupby(col)[test_numeric_col].agg(['count', 'mean', 'std']).dropna()

                        if len(grouped) > 1:
                            # Check if groups have different means (indicates dimension is meaningful)
                            mean_variance = grouped['mean'].var()
                            if mean_variance > 0:
                                dimension_score += 0.2
                                dimension_reasons.append('creates_meaningful_groups')

                            # Check if groups have reasonable sizes (not all tiny groups)
                            min_group_size = grouped['count'].min()
                            if min_group_size >= 3:
                                dimension_score += 0.1
                                dimension_reasons.append('reasonable_group_sizes')
                    except:
                        pass

                if dimension_score >= 0.4:
                    col_data = df[col].dropna()
                    dimension_info = {
                        'column': str(col),
                        'dimension_score': float(dimension_score),
                        'unique_count': stats['unique_count'],
                        'uniqueness_ratio': float(stats['uniqueness_ratio']),
                        'null_ratio': float(stats['null_ratio']),
                        'cardinality_level': (
                            'very low' if stats['unique_count'] <= 20 else
                            'low' if stats['unique_count'] <= 100 else
                            'medium' if stats['unique_count'] <= 1000 else
                            'high'
                        ),
                        'reasons': dimension_reasons
                    }

                    # Add sample values
                    if len(col_data) > 0:
                        value_counts = col_data.value_counts()
                        sample_values = value_counts.head(10).index.tolist()
                        dimension_info['sample_values'] = [str(val) for val in sample_values if pd.notna(val)]
                        dimension_info['most_frequent_value'] = str(sample_values[0]) if sample_values else None
                        dimension_info['most_frequent_count'] = int(value_counts.iloc[0]) if len(
                            value_counts) > 0 else 0

                    patterns['dimensions'].append(dimension_info)

            except Exception as e:
                continue

        # MEASURE DETECTION (Data-driven)
        for col, stats in column_stats.items():
            try:
                if col in identifier_columns or not stats['is_numeric']:
                    continue

                measure_score = 0
                measure_reasons = []

                # Reasonable cardinality for measures (should have variety)
                if stats['unique_count'] > 10:
                    measure_score += 0.3
                    measure_reasons.append('good_cardinality')
                elif stats['unique_count'] > 5:
                    measure_score += 0.2
                    measure_reasons.append('moderate_cardinality')
                elif stats['unique_count'] <= 3:
                    measure_score -= 0.2  # Likely categorical even if numeric
                    measure_reasons.append('low_cardinality_penalty')

                # Wide range suggests continuous measure
                if stats.get('range_value', 0) > 1:
                    measure_score += 0.2
                    measure_reasons.append('wide_range')

                # Check coefficient of variation (measures often have meaningful variation)
                cv = stats.get('coefficient_of_variation', 0)
                if 0.1 < cv < 10:  # Reasonable variation, not constant or wildly variable
                    measure_score += 0.2
                    measure_reasons.append('meaningful_variation')
                elif cv < 0.05:
                    measure_score -= 0.1
                    measure_reasons.append('low_variation')

                # Non-sequential numeric data is more likely a measure
                if stats.get('sequential_ratio', 0) < 0.3:
                    measure_score += 0.2
                    measure_reasons.append('non_sequential')
                else:
                    measure_score -= 0.1
                    measure_reasons.append('sequential_penalty')

                # Floating point values suggest continuous measures
                if 'float' in stats['dtype']:
                    measure_score += 0.1
                    measure_reasons.append('float_type')

                # Check if this correlates with other potential measures (business logic)
                if len(numeric_columns) > 1:
                    try:
                        correlations = df[numeric_columns].corr()[col].abs()
                        correlations = correlations.drop(col)

                        # Remove identifier columns from correlation analysis
                        correlations = correlations.drop([c for c in identifier_columns if c in correlations.index])

                        if len(correlations) > 0:
                            max_correlation = correlations.max()
                            if max_correlation > 0.3:  # Some correlation with other measures is good
                                measure_score += 0.15
                                measure_reasons.append('correlates_with_measures')
                    except:
                        pass

                # Test if aggregating this column makes sense by checking group variance
                if patterns['dimensions']:
                    try:
                        # Use the first dimension to test aggregation meaningfulness
                        test_dim = patterns['dimensions'][0]['column']
                        grouped_means = df.groupby(test_dim)[col].mean()

                        if len(grouped_means) > 1 and grouped_means.var() > 0:
                            measure_score += 0.2
                            measure_reasons.append('aggregation_meaningful')
                    except:
                        pass

                if measure_score >= 0.4:
                    measure_info = {
                        'column': str(col),
                        'measure_score': float(measure_score),
                        'unique_count': stats['unique_count'],
                        'uniqueness_ratio': float(stats['uniqueness_ratio']),
                        'null_ratio': float(stats['null_ratio']),
                        'data_type': stats['dtype'],
                        'can_be_summed': True,
                        'measure_type': 'continuous' if stats['uniqueness_ratio'] > 0.1 else 'discrete',
                        'reasons': measure_reasons
                    }

                    # Add statistical info
                    measure_info.update({
                        'min_value': float(stats['min_value']),
                        'max_value': float(stats['max_value']),
                        'mean_value': float(stats['mean_value']),
                        'median_value': float(stats['median_value']),
                        'std_value': float(stats['std_value']),
                        'range_value': float(stats['range_value']),
                        'has_negatives': stats['min_value'] < 0,
                        'has_decimals': 'float' in stats['dtype']
                    })

                    patterns['measures'].append(measure_info)

            except Exception as e:
                continue

        # Keep existing format detection for special patterns
        for col in df.columns:
            try:
                col_sample = df[col].dropna().astype(str).head(50)
                sample_size = len(col_sample)
                if sample_size == 0:
                    continue

                format_info = {}

                # Email pattern
                email_count = sum(1 for val in col_sample
                                  if re.match(r'^[^@]+@[^@]+\.[^@]+$', str(val)))
                if email_count > sample_size * 0.8:
                    format_info['likely_email'] = True

                # Phone pattern
                phone_count = sum(1 for val in col_sample
                                  if re.match(r'[\d\-\(\)\+\s]{10,}', str(val)))
                if phone_count > sample_size * 0.7:
                    format_info['likely_phone'] = True

                # Date pattern
                date_count = 0
                for val in col_sample:
                    try:
                        if pd.notna(val) and str(val).strip() != '':
                            pd.to_datetime(val, errors='raise')
                            date_count += 1
                    except:
                        continue
                if date_count > sample_size * 0.7:
                    format_info['likely_date'] = True

                # URL pattern
                url_count = sum(1 for val in col_sample
                                if str(val).startswith(('http://', 'https://', 'www.')))
                if url_count > sample_size * 0.7:
                    format_info['likely_url'] = True

                if format_info:
                    patterns['data_format_patterns'][str(col)] = format_info

            except Exception as e:
                continue

    except Exception as e:
        print(f"Warning in detect_column_patterns: {e}")

    return convert_to_json_serializable(patterns)


def analyze_data_quality(df: pd.DataFrame, excluded_columns: List[str] = None) -> Dict[str, Any]:
    """Comprehensive data quality analysis with fixed boolean array issues"""
    quality_metrics = {
        'completeness': {},
        'consistency': {},
        'case_inconsistencies': {},
        'outliers': {},
        'duplicates': {},
        'overall_score': 0
    }
    
    if excluded_columns is None:
        excluded_columns = []
    
    try:
        # Completeness analysis - FIXED: ensure scalar calculations
        total_cells = int(df.shape[0]) * int(df.shape[1])  # Ensure scalars
        null_cells = int(df.isnull().sum().sum())  # Ensure scalar
        
        quality_metrics['completeness']['overall_ratio'] = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        quality_metrics['completeness']['columns_with_nulls'] = int(df.isnull().any().sum())
        
        complete_rows = int(df.dropna().shape[0])  # Ensure scalar
        total_rows = int(df.shape[0])  # Ensure scalar
        quality_metrics['completeness']['complete_rows'] = complete_rows
        quality_metrics['completeness']['complete_rows_ratio'] = complete_rows / total_rows if total_rows > 0 else 0
        
        # Column-specific completeness - FIXED
        col_completeness = {}
        for col in df.columns:
            try:
                null_count = int(df[col].isnull().sum())  # Ensure scalar
                total_count = int(len(df))  # Ensure scalar
                completeness = 1 - (null_count / total_count) if total_count > 0 else 0
                
                col_completeness[str(col)] = {
                    'completeness_ratio': float(completeness),
                    'missing_count': null_count,
                    'quality_tier': 'excellent' if completeness >= 0.95 else 
                                   'good' if completeness >= 0.8 else 
                                   'fair' if completeness >= 0.5 else 'poor'
                }
            except:
                col_completeness[str(col)] = {
                    'completeness_ratio': 0.0,
                    'missing_count': len(df),
                    'quality_tier': 'poor'
                }
        quality_metrics['completeness']['by_column'] = col_completeness
        
        # Case inconsistency analysis for text columns - FIXED and MODIFIED to exclude long text columns
        case_inconsistency_info = {}
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        # Filter out excluded columns (like long text columns)
        filtered_text_columns = [col for col in text_columns if col not in excluded_columns]
        
        for col in filtered_text_columns:
            try:
                col_data = df[col].dropna().astype(str)
                if len(col_data) == 0:
                    continue
                
                # Convert to lowercase for comparison - FIXED: ensure scalar comparisons
                lowercase_values = col_data.str.lower()
                original_unique = int(col_data.nunique())  # Ensure scalar
                lowercase_unique = int(lowercase_values.nunique())  # Ensure scalar
                
                # Also check for whitespace inconsistencies
                stripped_values = col_data.str.strip()
                stripped_unique = int(stripped_values.nunique())  # Ensure scalar
                
                # Combined normalization (lowercase + stripped)
                normalized_values = col_data.str.lower().str.strip()
                normalized_unique = int(normalized_values.nunique())  # Ensure scalar
                
                # FIXED: Now comparing scalars, not potentially Series
                inconsistencies_found = original_unique > normalized_unique
                
                if inconsistencies_found:
                    # Found case and/or whitespace inconsistencies
                    case_groups = {}
                    inconsistent_examples = []
                    
                    try:
                        # Group values by their normalized version
                        for orig_val in col_data.unique()[:100]:  # Limit for performance
                            try:
                                normalized_val = str(orig_val).lower().strip()
                                if normalized_val not in case_groups:
                                    case_groups[normalized_val] = []
                                case_groups[normalized_val].append(orig_val)
                            except:
                                continue
                        
                        # Find groups with multiple variations
                        inconsistent_groups = {k: v for k, v in case_groups.items() if len(v) > 1}
                        
                        # Create examples of inconsistencies
                        for normalized_val, variations in list(inconsistent_groups.items())[:5]:
                            inconsistent_examples.append({
                                'normalized_form': normalized_val,
                                'variations': variations[:5],
                                'variation_count': len(variations)
                            })
                        
                        # Count total affected records
                        affected_records = 0
                        try:
                            value_counts = col_data.value_counts()
                            for variations in inconsistent_groups.values():
                                for var in variations[1:]:  # Count all but most frequent
                                    if var in value_counts:
                                        affected_records += int(value_counts[var])  # Ensure scalar
                        except:
                            affected_records = 0
                        
                        case_inconsistency_info[str(col)] = {
                            'original_unique_count': original_unique,
                            'lowercase_unique_count': lowercase_unique,
                            'stripped_unique_count': stripped_unique,
                            'normalized_unique_count': normalized_unique,
                            'inconsistent_groups': len(inconsistent_groups),
                            'potential_duplicates': original_unique - normalized_unique,
                            'case_issues': original_unique - lowercase_unique,
                            'whitespace_issues': original_unique - stripped_unique,
                            'affected_records': affected_records,
                            'affected_records_ratio': float(affected_records / len(col_data)) if len(col_data) > 0 else 0.0,
                            'examples': inconsistent_examples,
                            'severity': 'high' if (affected_records / len(col_data) if len(col_data) > 0 else 0) > 0.1 else 
                                      'medium' if (affected_records / len(col_data) if len(col_data) > 0 else 0) > 0.05 else 'low',
                            'issue_types': []
                        }
                        
                        # Identify specific issue types
                        if original_unique > lowercase_unique:
                            case_inconsistency_info[str(col)]['issue_types'].append('case_variation')
                        if original_unique > stripped_unique:
                            case_inconsistency_info[str(col)]['issue_types'].append('whitespace_variation')
                            
                    except Exception as e:
                        # If detailed analysis fails, provide basic info
                        case_inconsistency_info[str(col)] = {
                            'original_unique_count': original_unique,
                            'normalized_unique_count': normalized_unique,
                            'potential_duplicates': original_unique - normalized_unique,
                            'severity': 'low',
                            'issue_types': ['normalization_needed']
                        }
            except Exception as e:
                continue
        
        quality_metrics['case_inconsistencies'] = case_inconsistency_info
        
        # Duplicate analysis - FIXED: ensure scalar values
        try:
            duplicate_count = int(df.duplicated().sum())  # Ensure scalar
            total_rows = int(len(df))  # Ensure scalar
            quality_metrics['duplicates']['total_duplicates'] = duplicate_count
            quality_metrics['duplicates']['duplicate_ratio'] = float(duplicate_count / total_rows) if total_rows > 0 else 0.0
        except:
            quality_metrics['duplicates']['total_duplicates'] = 0
            quality_metrics['duplicates']['duplicate_ratio'] = 0.0
        
        # Find columns with potential duplicates - FIXED
        duplicate_cols = {}
        for col in df.columns:
            try:
                if df[col].dtype in ['object', 'string']:
                    # Check for whitespace variations
                    cleaned = df[col].astype(str).str.strip().str.lower()
                    original_unique = int(df[col].nunique())  # Ensure scalar
                    cleaned_unique = int(cleaned.nunique())  # Ensure scalar
                    
                    if original_unique > cleaned_unique:
                        duplicate_cols[str(col)] = {
                            'original_unique': original_unique,
                            'cleaned_unique': cleaned_unique,
                            'improvement_ratio': float((original_unique - cleaned_unique) / original_unique) if original_unique > 0 else 0.0
                        }
            except:
                continue
        
        quality_metrics['duplicates']['whitespace_duplicates'] = duplicate_cols
        
        # Outlier detection for numeric columns - FIXED
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            try:
                unique_count = int(df[col].nunique())  # Ensure scalar
                if unique_count > 1:  # Skip constant columns
                    # Remove NaN values for outlier calculation
                    clean_col = df[col].dropna()
                    if len(clean_col) < 3:  # Need at least 3 values
                        continue
                        
                    Q1 = float(clean_col.quantile(0.25))  # Ensure scalar
                    Q3 = float(clean_col.quantile(0.75))  # Ensure scalar
                    IQR = Q3 - Q1
                    
                    if pd.isna(Q1) or pd.isna(Q3) or pd.isna(IQR) or IQR == 0:
                        continue
                        
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # FIXED: Use bitwise operators and handle boolean Series properly
                    outlier_mask = (clean_col < lower_bound) | (clean_col > upper_bound)
                    has_outliers = safe_bool_check(outlier_mask)  # Use safe boolean check
                    
                    if has_outliers:
                        outliers = clean_col[outlier_mask]
                        outlier_count = int(len(outliers))  # Ensure scalar
                        clean_count = int(len(clean_col))  # Ensure scalar
                        
                        outlier_info[str(col)] = {
                            'outlier_count': outlier_count,
                            'outlier_ratio': float(outlier_count / clean_count) if clean_count > 0 else 0.0,
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'extreme_values': [float(x) for x in outliers.head(10).tolist()]
                        }
            except Exception as e:
                continue
        
        quality_metrics['outliers'] = outlier_info
        
        # Overall quality score - FIXED: ensure all values are scalars
        try:
            completeness_score = float(quality_metrics['completeness']['overall_ratio'])
            duplicate_penalty = min(0.3, float(quality_metrics['duplicates']['duplicate_ratio']))
            outlier_penalty = min(0.2, sum(float(info.get('outlier_ratio', 0)) for info in outlier_info.values()) / max(1, len(outlier_info)))
            
            # Case inconsistency penalty
            case_penalty = 0
            if case_inconsistency_info:
                total_case_affected = sum(float(info.get('affected_records_ratio', 0)) for info in case_inconsistency_info.values())
                case_penalty = min(0.15, total_case_affected / len(case_inconsistency_info))
            
            quality_metrics['overall_score'] = float(max(0, completeness_score - duplicate_penalty - outlier_penalty - case_penalty))
        except:
            quality_metrics['overall_score'] = 0.5
        
    except Exception as e:
        print(f"Warning in analyze_data_quality: {e}")
        quality_metrics = {
            'completeness': {'overall_ratio': 0.5, 'complete_rows_ratio': 0.5},
            'case_inconsistencies': {},
            'outliers': {},
            'duplicates': {'duplicate_ratio': 0.0},
            'overall_score': 0.5
        }
    
    return convert_to_json_serializable(quality_metrics)


def analyze_cardinality_and_uniqueness(df: pd.DataFrame, max_unique_values: int = 100, excluded_columns: List[str] = None) -> Dict[str, Any]:
    """Analyze cardinality with fixed boolean array issues - MODIFIED to exclude identifier columns, long text columns, and filter by data type"""
    cardinality_info = {}
    
    if excluded_columns is None:
        excluded_columns = []
    
    # Filter columns to only include categorical/string/object/int types and exclude identifier/long text columns
    valid_columns = []
    for col in df.columns:
        # Skip excluded columns (identifier columns, long text columns)
        if col in excluded_columns:
            continue
            
        # Check if column is of appropriate data type
        col_dtype = str(df[col].dtype).lower()
        is_valid_type = (
            'object' in col_dtype or 
            'string' in col_dtype or 
            'int' in col_dtype or
            'category' in col_dtype
        )
        
        if is_valid_type:
            valid_columns.append(col)
    
    for col in valid_columns:
        try:
            col_data = df[col].dropna()
            unique_count = int(col_data.nunique())  # FIXED: ensure scalar
            total_count = int(len(col_data))  # FIXED: ensure scalar
            
            info = {
                'unique_count': unique_count,
                'total_count': total_count,
                'uniqueness_ratio': float(unique_count / total_count) if total_count > 0 else 0.0,
                'cardinality_type': '',
                'unique_values_sample': [],
                'value_frequency_top10': {}
            }
            
            # Classify cardinality - FIXED: all comparisons with scalars
            if unique_count == 1:
                info['cardinality_type'] = 'constant'
            elif unique_count == total_count and total_count > 100:
                info['cardinality_type'] = 'unique'
            elif unique_count == total_count and total_count <= 100:
                info['cardinality_type'] = 'low_cardinality'
            elif total_count > 100 and (unique_count / total_count) > 0.9:
                info['cardinality_type'] = 'high_cardinality'
            elif unique_count <= 100:
                info['cardinality_type'] = 'low_cardinality'
            else:
                info['cardinality_type'] = 'medium_cardinality'

            # Sample unique values prioritized by frequency (most frequent first)
            try:
                # Get value counts to prioritize by frequency
                value_counts = col_data.value_counts()

                if unique_count <= max_unique_values:
                    # Show all values, ordered by frequency
                    info['unique_values_sample'] = [str(val) for val in value_counts.index[:max_unique_values] if
                                                        pd.notna(val)]
                else:
                    # Show top max_unique_values most frequent values
                    top_frequent_values = value_counts.head(max_unique_values)
                    info['unique_values_sample'] = [str(val) for val in top_frequent_values.index if pd.notna(val)]
                    info['unique_values_sample'].append(f"... and {unique_count - max_unique_values} more")
            except:
                info['unique_values_sample'] = ['Error retrieving values']

            # Top 10 most frequent values
            try:
                value_counts = col_data.value_counts().head(10)
                info['value_frequency_top10'] = {str(k): int(v) for k, v in value_counts.items()}
            except:
                info['value_frequency_top10'] = {}
            
            cardinality_info[str(col)] = info
            
        except Exception as e:
            cardinality_info[str(col)] = {
                'unique_count': 0,
                'total_count': len(df),
                'uniqueness_ratio': 0.0,
                'cardinality_type': 'unknown',
                'unique_values_sample': [],
                'value_frequency_top10': {}
            }
    
    return convert_to_json_serializable(cardinality_info)


def analyze_data_relationships(df: pd.DataFrame, excluded_columns: List[str] = None) -> Dict[str, Any]:
    """Analyze relationships with fixed boolean array issues"""

    if excluded_columns is None:
        excluded_columns = []

    relationships = {
        'correlations': [],
        'potential_foreign_keys': [],
        'functional_dependencies': [],
        'column_associations': {}
    }
    
    try:
        # Potential foreign key relationships - FIXED
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        for col1 in categorical_cols:
            if col1 in excluded_columns:
                continue
            for col2 in categorical_cols:
                if col2 in excluded_columns:
                    continue
                if col1 != col2:
                    try:
                        # Check if values in col1 are subset of values in col2
                        values1 = set(df[col1].dropna().astype(str).unique())
                        values2 = set(df[col2].dropna().astype(str).unique())
                        
                        values1_len = len(values1)  # FIXED: ensure scalar
                        values2_len = len(values2)  # FIXED: ensure scalar
                        
                        if values1_len > 1 and values2_len > 1 and values1.issubset(values2):
                            overlap_ratio = len(values1.intersection(values2)) / values1_len if values1_len > 0 else 0
                            if overlap_ratio > 0.8:
                                relationships['potential_foreign_keys'].append({
                                    'child_column': str(col1),
                                    'parent_column': str(col2),
                                    'overlap_ratio': float(overlap_ratio),
                                    'child_unique_count': values1_len,
                                    'parent_unique_count': values2_len
                                })
                    except Exception as e:
                        continue
        
        # Functional dependencies - FIXED
        for col1 in categorical_cols:
            if col1 in excluded_columns:
                continue
            for col2 in categorical_cols:
                if col2 in excluded_columns:
                    continue
                if col1 != col2:
                    try:
                        # Check if col1 -> col2 (one-to-one or many-to-one relationship)
                        grouped = df.groupby(col1)[col2].nunique()
                        grouped_len = int(len(grouped))  # FIXED: ensure scalar
                        
                        if grouped_len > 0:
                            # FIXED: Use .all() on the boolean Series correctly
                            one_to_one_mask = (grouped == 1)
                            all_one_to_one = safe_bool_check(one_to_one_mask.all())  # Use safe boolean check
                            
                            if all_one_to_one:
                                relationships['functional_dependencies'].append({
                                    'determinant': str(col1),
                                    'dependent': str(col2),
                                    'relationship_type': 'functional_dependency'
                                })
                    except Exception as e:
                        continue

    except Exception as e:
        print(f"Warning in analyze_data_relationships: {e}")
    
    return convert_to_json_serializable(relationships)


def generate_enhanced_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data summary with all enhanced analysis and robust error handling"""
    
    print("Generating enhanced data profile...")
    
    # Start with basic summary
    enhanced_summary = {
        'basic_info': {
            'shape': list(df.shape),
            'columns': [str(col) for col in df.columns],
            'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': int(df.memory_usage(deep=True).sum())
        }
    }
    
    # Detect and exclude long text columns from all analyses
    try:
        print("Detecting long text columns to exclude from analysis...")
        long_text_columns = detect_long_text_columns(df)
        enhanced_summary['excluded_columns'] = {
            'long_text_columns': long_text_columns,
            'reason': 'Contains long free-form text (paragraphs/multiple lines) unsuitable for categorical analysis'
        }
        if long_text_columns:
            print(f"Excluding {len(long_text_columns)} long text columns: {long_text_columns}")
    except Exception as e:
        print(f"Warning: Long text column detection failed: {e}")
        long_text_columns = []
        enhanced_summary['excluded_columns'] = {
            'long_text_columns': [],
            'reason': 'Long text detection failed'
        }
    
    # Track which analyses succeeded
    analysis_results = {}
    
    # 1. Data Quality Analysis (MODIFIED: Now excludes long text columns from case inconsistency analysis)
    try:
        print("Analyzing data quality...")
        enhanced_summary['data_quality'] = analyze_data_quality(df, excluded_columns=long_text_columns)
        analysis_results['data_quality'] = 'success'
    except Exception as e:
        print(f"Warning: Data quality analysis failed: {e}")
        enhanced_summary['data_quality'] = {'overall_score': 0.5}
        analysis_results['data_quality'] = 'failed'
    
    # 2. Column Patterns (MODIFIED: Now filters identifier columns by data type)
    try:
        print("Detecting column patterns...")
        enhanced_summary['column_patterns'] = detect_column_patterns(df)
        analysis_results['column_patterns'] = 'success'
    except Exception as e:
        print(f"Warning: Column pattern analysis failed: {e}")
        enhanced_summary['column_patterns'] = {}
        analysis_results['column_patterns'] = 'failed'

    # get potential identifiers
    potential_identifiers = enhanced_summary['column_patterns'].get('potential_identifiers', [])
    identifier_columns = [id_info['column'] for id_info in potential_identifiers]
    # Combine all columns to exclude: identifiers + long text columns
    all_excluded_columns = list(set(identifier_columns + long_text_columns))

    # 3. NEW: Special Numeric Columns Detection
    try:
        print("Detecting special numeric columns requiring careful handling...")
        enhanced_summary['special_numeric_columns'] = detect_special_numeric_columns(df)
        analysis_results['special_numeric_columns'] = 'success'
    except Exception as e:
        print(f"Warning: Special numeric columns analysis failed: {e}")
        enhanced_summary['special_numeric_columns'] = {}
        analysis_results['special_numeric_columns'] = 'failed'

    # 4. Data Relationships
    try:
        print("Analyzing relationships...")
        enhanced_summary['relationships'] = analyze_data_relationships(df, excluded_columns=all_excluded_columns)
        analysis_results['relationships'] = 'success'
    except Exception as e:
        print(f"Warning: Relationship analysis failed: {e}")
        enhanced_summary['relationships'] = {}
        analysis_results['relationships'] = 'failed'

    # Add analysis status summary
    enhanced_summary['analysis_status'] = analysis_results
    successful_analyses = sum(1 for status in analysis_results.values() if status == 'success')
    total_analyses = len(analysis_results)
    
    print(f"Enhanced data profiling complete! ({successful_analyses}/{total_analyses} analyses successful)")
    
    return convert_to_json_serializable(enhanced_summary)


def format_enhanced_context_for_llm(enhanced_summary: Dict[str, Any], original_context: str) -> str:
    """Format the enhanced summary into a comprehensive context string for the LLM with improved categorization"""

    context_parts = [original_context, "\n", "ENHANCED DATA PROFILE:"]

    # Analysis status
    analysis_status = enhanced_summary.get('analysis_status', {})
    successful_analyses = [name for name, status in analysis_status.items() if status == 'success']
    failed_analyses = [name for name, status in analysis_status.items() if status == 'failed']

    if successful_analyses:
        context_parts.append(f"Successfully completed analyses: {', '.join(successful_analyses)}")
    if failed_analyses:
        context_parts.append(f"Failed analyses (using fallback data): {', '.join(failed_analyses)}")

    # Excluded Columns Information
    excluded_info = enhanced_summary.get('excluded_columns', {})
    long_text_columns = excluded_info.get('long_text_columns', [])
    if long_text_columns:
        context_parts.append(f"\nEXCLUDED FROM ANALYSIS:")
        context_parts.append(f"Long Text Columns ({len(long_text_columns)}): {', '.join(long_text_columns)}")
        context_parts.append(
            f"Reason: {excluded_info.get('reason', 'Contains long free-form text unsuitable for categorical analysis')}")
        context_parts.append(
            "IMPORTANT: These columns contain free-form text (paragraphs, multiple lines) and should be IGNORED in all data analysis tasks.")
        context_parts.append("They are not suitable for categorical analysis, grouping, or statistical operations.")

    # NEW: Special Numeric Columns Section
    special_numeric = enhanced_summary.get('special_numeric_columns', {})
    if special_numeric and special_numeric.get('summary', {}).get('total_special_columns', 0) > 0:
        summary = special_numeric['summary']
        context_parts.append(f"\nSPECIAL NUMERIC COLUMNS REQUIRING CAREFUL HANDLING:")
        context_parts.append(f"Total Special Columns: {summary['total_special_columns']}")
        context_parts.append(f"Categories Found: {', '.join(summary.get('categories_found', []))}")
        
        # Percentage columns
        percentage_cols = special_numeric.get('percentage_columns', [])
        if percentage_cols:
            context_parts.append(f"\nPERCENTAGE COLUMNS ({len(percentage_cols)}) - CANNOT be simply added or averaged:")
            for col_info in percentage_cols:
                reasons = ', '.join(col_info.get('detection_reasons', [])[:2])
                context_parts.append(f"  - {col_info['column']} - {reasons}")
            context_parts.append("  Use weighted averages, never simple sum/mean for percentages!")
        
        # Ratio/Rate columns
        ratio_cols = special_numeric.get('ratio_columns', []) + special_numeric.get('rate_columns', [])
        if ratio_cols:
            context_parts.append(f"\nRATIO/RATE COLUMNS ({len(ratio_cols)}) - Require weighted aggregation:")
            for col_info in ratio_cols:
                context_parts.append(f"  - {col_info['column']} - Use harmonic mean or weighted averages")
        
        # ID-like numeric columns
        id_like_cols = special_numeric.get('id_like_columns', [])
        if id_like_cols:
            context_parts.append(f"\nID-LIKE NUMERIC COLUMNS ({len(id_like_cols)}) - NO arithmetic operations:")
            for col_info in id_like_cols:
                context_parts.append(f"  - {col_info['column']} - Use only for counting, grouping, joining")
        
        # Coordinate columns
        coord_cols = special_numeric.get('coordinate_columns', [])
        if coord_cols:
            context_parts.append(f"\nCOORDINATE COLUMNS ({len(coord_cols)}) - Geographic data:")
            for col_info in coord_cols:
                context_parts.append(f"  - {col_info['column']} - Use centroid calculations, not simple averages")
        
        # Year columns
        year_cols = special_numeric.get('year_columns', [])
        if year_cols:
            context_parts.append(f"\nYEAR COLUMNS ({len(year_cols)}) - Temporal identifiers:")
            for col_info in year_cols:
                context_parts.append(f"  - {col_info['column']} - Use ranges, modes, not averages")

        # Score/Index columns
        score_cols = special_numeric.get('score_columns', []) + special_numeric.get('index_columns', [])
        if score_cols:
            context_parts.append(f"\nSCORE/INDEX COLUMNS ({len(score_cols)}) - Use weighted averages:")
            for col_info in score_cols:
                context_parts.append(f"  - {col_info['column']} - Average OK, sum usually not meaningful")

        # Critical operation warnings
        careful_aggregation = summary.get('requires_careful_aggregation', [])
        avoid_arithmetic = summary.get('avoid_simple_arithmetic', [])
        
        if careful_aggregation:
            context_parts.append(f"\n CRITICAL AGGREGATION WARNINGS:")
            context_parts.append(f"NEVER use simple sum/mean on: {', '.join(careful_aggregation)}")
            
        if avoid_arithmetic:
            context_parts.append(f"AVOID ALL arithmetic on: {', '.join(avoid_arithmetic)}")

    # Data Quality Summary
    if 'data_quality' in enhanced_summary and enhanced_summary['data_quality']:
        quality = enhanced_summary['data_quality']
        context_parts.append(f"\nDATA QUALITY ASSESSMENT:")
        context_parts.append(f"Overall Quality Score: {quality.get('overall_score', 0):.2f}/1.0")

        completeness = quality.get('completeness', {})
        if completeness:
            context_parts.append(f"Data Completeness: {completeness.get('overall_ratio', 0):.1%}")
            context_parts.append(f"Complete Rows: {completeness.get('complete_rows_ratio', 0):.1%}")

        duplicates = quality.get('duplicates', {})
        if duplicates:
            context_parts.append(f"Duplicate Records: {duplicates.get('duplicate_ratio', 0):.1%}")

        # Column quality tiers
        col_quality = completeness.get('by_column', {})
        if col_quality:
            excellent_cols = [col for col, info in col_quality.items() if info.get('quality_tier') == 'excellent']
            poor_cols = [col for col, info in col_quality.items() if info.get('quality_tier') == 'poor']

            if excellent_cols:
                context_parts.append(f"High Quality Columns ({len(excellent_cols)}): {', '.join(excellent_cols)}")
            if poor_cols:
                context_parts.append(f"Low Quality Columns ({len(poor_cols)}): {', '.join(poor_cols)}")

        # Case inconsistency information
        case_issues = quality.get('case_inconsistencies', {})
        if case_issues:
            high_severity_cols = [col for col, info in case_issues.items() if info.get('severity') == 'high']
            medium_severity_cols = [col for col, info in case_issues.items() if info.get('severity') == 'medium']

            if high_severity_cols or medium_severity_cols:
                context_parts.append(f"\nTEXT STANDARDIZATION ISSUES DETECTED:")

                if high_severity_cols:
                    context_parts.append(f"High Impact Columns: {', '.join(high_severity_cols)}")
                if medium_severity_cols:
                    context_parts.append(f"Medium Impact Columns: {', '.join(medium_severity_cols)}")

                # Show examples for the most problematic column
                if case_issues:
                    worst_col = max(case_issues.keys(), key=lambda x: case_issues[x].get('affected_records_ratio', 0))
                    worst_info = case_issues[worst_col]
                    issue_types = worst_info.get('issue_types', [])

                    context_parts.append(f"Example from '{worst_col}' (Issues: {', '.join(issue_types)}):")
                    for example in worst_info.get('examples', [])[:2]:  # Show first 2 examples
                        variations_str = ', '.join([f"'{var}'" for var in example.get('variations', [])[:5]])
                        context_parts.append(f"  • '{example.get('normalized_form', '')}' appears as: {variations_str}")

    # Column Patterns
    if 'column_patterns' in enhanced_summary and enhanced_summary['column_patterns']:
        patterns = enhanced_summary['column_patterns']
        context_parts.append(f"\nCOLUMN PATTERNS:")

        naming = patterns.get('naming_conventions', {})
        if naming:
            context_parts.append(f"Naming Convention: {naming.get('dominant_style', 'unknown')}")

        groups = patterns.get('column_groups', {})
        if groups and groups.get('by_prefix'):
            context_parts.append("Related Column Groups:")
            for prefix, cols in list(groups['by_prefix'].items()):
                context_parts.append(f"  • {prefix}_*: {', '.join(cols)}")

        formats = patterns.get('data_format_patterns', {})
        if formats:
            context_parts.append("Special Data Formats Detected:")
            for col, format_info in list(formats.items()):
                format_types = [k.replace('likely_', '') for k, v in format_info.items() if
                                v and k.startswith('likely_')]
                if format_types:
                    context_parts.append(f"  • {col}: {', '.join(format_types)}")

        # DATA-DRIVEN IDENTIFIER CLASSIFICATION
        identifiers = patterns.get('potential_identifiers', [])
        if identifiers:
            context_parts.append(f"\nIDENTIFIER COLUMNS ({len(identifiers)}) - Detected by data patterns:")

            # Group by confidence level
            high_conf_ids = [id_info for id_info in identifiers if id_info.get('confidence') == 'high']
            medium_conf_ids = [id_info for id_info in identifiers if id_info.get('confidence') == 'medium']
            low_conf_ids = [id_info for id_info in identifiers if id_info.get('confidence') == 'low']

            if high_conf_ids:
                for info in high_conf_ids:
                    reasons = ', '.join(info.get('reasons', [])[:3])  # Show top 3 reasons
                    context_parts.append(f"  • {info['column']} ({info.get('identifier_type', 'generic')}) - {reasons}")

            if medium_conf_ids:
                id_details = [f"{info['column']} ({info.get('identifier_type', 'generic')})" for info in medium_conf_ids]
                context_parts.append(f"Medium Confidence: {', '.join(id_details)}")

            if low_conf_ids:
                id_details = [info['column'] for info in low_conf_ids]
                context_parts.append(f"Low Confidence: {', '.join(id_details)}")

            context_parts.append("NOTE: Identifiers detected by uniqueness, consistency, and correlation patterns.")

        # DATA-DRIVEN DIMENSION CLASSIFICATION
        dimensions = patterns.get('dimensions', [])
        if dimensions:
            context_parts.append(f"\nDIMENSION COLUMNS ({len(dimensions)}) - Detected by grouping patterns:")

            # Group by cardinality level
            very_low_card_dims = [dim for dim in dimensions if dim.get('cardinality_level') == 'very low']
            low_card_dims = [dim for dim in dimensions if dim.get('cardinality_level') == 'low']
            medium_card_dims = [dim for dim in dimensions if dim.get('cardinality_level') == 'medium']
            high_card_dims = [dim for dim in dimensions if dim.get('cardinality_level') == 'high']

            if very_low_card_dims:
                context_parts.append(f"Very Low Cardinality (≤20 values) - Best for grouping:")
                for dim in very_low_card_dims:
                    context_parts.append(f" - {dim['column']}")

            if low_card_dims:
                context_parts.append(f"Low Cardinality (≤100 values) - Great for grouping:")
                for dim in low_card_dims:
                    context_parts.append(f" - {dim['column']}")

            if medium_card_dims:
                context_parts.append(f"Medium Cardinality (100-1000 values) - Good for grouping:")
                for dim in medium_card_dims:  # Show details for first 3
                    context_parts.append(f" - {dim['column']}")

            if high_card_dims:
                dim_names = [f"{dim['column']} ({dim['unique_count']} values)" for dim in high_card_dims]
                context_parts.append(f"High Cardinality: {', '.join(dim_names)}")

        # DATA-DRIVEN MEASURE CLASSIFICATION
        measures = patterns.get('measures', [])
        if measures:
            context_parts.append(f"\nMEASURE COLUMNS ({len(measures)}) - Detected by aggregation patterns:")

            # Group by measure type and show key characteristics
            continuous_measures = [m for m in measures if m.get('measure_type') == 'continuous']
            discrete_measures = [m for m in measures if m.get('measure_type') == 'discrete']

            if continuous_measures:
                context_parts.append(f"Continuous Measures:")
                for measure in continuous_measures:
                    col_name = measure['column']
                    min_val = measure.get('min_value', 0)
                    max_val = measure.get('max_value', 0)
                    mean_val = measure.get('mean_value', 0)
                    median_val = measure.get('median_value', 0)
                    has_negatives = measure.get('has_negatives', False)
                    has_decimals = measure.get('has_decimals', False)
                    top_reasons = measure.get('reasons', [])[:2]

                    details = f"{col_name} [min: {min_val:.3f}, max: {max_val:.3f}, mean: {mean_val: .3f}, median: {median_val: .3f}]"
                    if has_decimals:
                        details += " (decimal)"
                    if has_negatives:
                        details += " (±)"
                    details += f" - {', '.join(top_reasons)}"

                    context_parts.append(f" - {details}")

            if discrete_measures:
                context_parts.append(f"Discrete Measures:")
                for measure in discrete_measures:
                    col_name = measure['column']
                    min_val = measure.get('min_value', 0)
                    max_val = measure.get('max_value', 0)
                    median_val = measure.get('median_value', 0)
                    unique_count = measure.get('unique_count', 0)
                    top_reasons = measure.get('reasons', [])[:2]

                    details = f"{col_name} [min: {int(min_val)}, max: {int(max_val)}, median: {median_val}, unique count: {unique_count}] - {', '.join(top_reasons)}"
                    context_parts.append(f" - {details}")

            context_parts.append("All measures detected by: range analysis, correlation patterns, aggregation effectiveness.")

    # Relationships
    if 'relationships' in enhanced_summary and enhanced_summary['relationships']:
        relationships = enhanced_summary['relationships']
        context_parts.append("\nDATA RELATIONSHIPS:")

        fk_relationships = relationships.get('potential_foreign_keys', [])
        if fk_relationships:
            context_parts.append(f"\nPotential FK Relationships:")
            for fk in fk_relationships[:20]:
                context_parts.append(f" - {fk.get('child_column', '')} → {fk.get('parent_column', '')}")

        functional_dependencies = relationships.get('functional_dependencies', [])
        if functional_dependencies:
            context_parts.append(f"\nFunctional Dependencies:")
            for fd in functional_dependencies[:20]:
                context_parts.append(f" - {fd.get('determinant', '')} determines {fd.get('dependent', '')}")

    context_parts.append("\n")
    context_parts.append("DATA-DRIVEN COLUMN CLASSIFICATION SUMMARY:")
    context_parts.append("- IDENTIFIERS: Detected by uniqueness patterns, consistency, and correlation analysis")
    context_parts.append("- DIMENSIONS: Detected by cardinality analysis and grouping effectiveness testing")
    context_parts.append("- MEASURES: Detected by statistical distribution analysis and aggregation meaningfulness")
    context_parts.append("- LONG TEXT: Detected by word count and character pattern analysis")
    context_parts.append("- SPECIAL NUMERIC: Detected by column names, value patterns, and data ranges")
    context_parts.append("\n")
    context_parts.append("USAGE RECOMMENDATIONS:")
    context_parts.append("- Use IDENTIFIERS for joins/references, exclude from grouping/aggregation")
    context_parts.append("- Use DIMENSIONS for GROUP BY, filtering, and categorical analysis")
    context_parts.append("- Use MEASURES for SUM, AVG, COUNT, and other aggregations")
    context_parts.append("- IGNORE long text columns completely in structured analysis")
    context_parts.append("- Handle SPECIAL NUMERIC columns with appropriate aggregation methods")
    context_parts.append("- Use EXACT column names to access data in python code")

    # Final reminder about special numeric columns
    if special_numeric and special_numeric.get('summary', {}).get('total_special_columns', 0) > 0:
        summary = special_numeric['summary']
        careful_aggregation = summary.get('requires_careful_aggregation', [])
        avoid_arithmetic = summary.get('avoid_simple_arithmetic', [])
        
        context_parts.append("\n" + "=" * 60)
        context_parts.append(" CRITICAL NUMERIC COLUMN HANDLING WARNINGS ")
        context_parts.append("=" * 60)
        
        if careful_aggregation:
            context_parts.append(f"NEVER use simple sum() or mean() on these columns:")
            context_parts.append(f" FORBIDDEN: {', '.join(careful_aggregation)}")
            context_parts.append(" Instead use: weighted averages, harmonic means, or contextual aggregation")
        
        if avoid_arithmetic:
            context_parts.append(f"\nCOMPLETELY AVOID arithmetic operations on:")
            context_parts.append(f" NO MATH: {', '.join(avoid_arithmetic)}")
            context_parts.append(" Instead use: counting, grouping, value_counts")
        
        context_parts.append("\n ALWAYS check column type before applying mathematical operations!")
        context_parts.append("=" * 60)

    context_parts.append(
        "\nUse this enhanced data profile to generate more sophisticated analysis and better error handling!")

    # Final reminder about excluded columns
    if long_text_columns:
        context_parts.append("\n" + "=" * 50)
        context_parts.append("CRITICAL REMINDER")
        context_parts.append(
            f"The following {len(long_text_columns)} columns contain LONG FREE-FORM TEXT and must be COMPLETELY IGNORED:")
        context_parts.append(f"IGNORE THESE: {', '.join(long_text_columns)}")
        context_parts.append("These columns are NOT suitable for:")
        context_parts.append("- Categorical analysis • Grouping operations • Statistical analysis")
        context_parts.append("- Value counting • Pattern analysis • Any data manipulation")
        context_parts.append("=" * 50)

    return "\n".join(context_parts)
