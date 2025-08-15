"""
Result Formatter Module for DataAnalysisAgent - FIXED VERSION
============================================

This module ensures all Python objects and data structures are formatted 
in a user-friendly way before being sent to the frontend. It prevents 
raw dictionaries, tuples, complex objects etc. from being displayed 
to non-technical users.

CRITICAL FIXES:
- Handles pandas DataFrames/Series when they appear as dict values
- Robust type checking to prevent "ambiguous truth value" errors
- Comprehensive fallback mechanisms for unexpected types

Usage in agent.py:
    from result_formatter import format_result_for_user
    
    # In _execute_code method, replace result processing with:
    formatted_result = format_result_for_user(result, result_type=state.get('result_type'))
    state['execution_result'] = formatted_result
"""

import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Union, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

class ResultFormatter:
    """Main class for formatting results for user consumption"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration for result formatting"""
        return {
            'max_table_rows': 10000,
            'max_table_cols': 100,
            'max_list_items': 10000,
            'max_string_length': 10000,
            'decimal_places': 3,
            'large_number_threshold': 1000000000000,
            'dict_display_threshold': 100,  # Max dict items to show as table
            'summary_threshold': 10,  # When to create summaries vs full display
            'date_format': '%Y-%m-%d',
            'datetime_format': '%Y-%m-%d %H:%M:%S',
            # New: DataFrame handling in dicts
            'max_dataframe_rows_in_dict': 1000,  # Max rows to preserve in dict DataFrames
            'preserve_all_dataframes': True,  # Whether to always preserve DataFrame data
        }
    
    def format_result_for_user(self, result: Any, result_type: str = 'auto') -> Union[str, List[Dict], Dict]:
        """
        Main function to format any Python object for user-friendly display
        
        Args:
            result: The Python object to format
            result_type: Hint about expected result type ('table', 'chart', 'value', 'auto')
            
        Returns:
            User-friendly formatted result (string, list of dicts for tables, or simple dict)
        """
        try:
            logger.info(f"Formatting result of type {type(result)} with result_type hint: {result_type}")
            
            # Handle None/empty results
            if result is None:
                return "No result returned from analysis"
            
            # Handle pandas objects first (CRITICAL: Before other type checks)
            if isinstance(result, pd.DataFrame):
                return self._format_dataframe(result, result_type)
            elif isinstance(result, pd.Series):
                return self._format_series(result, result_type)
            
            # Handle numpy arrays
            elif isinstance(result, np.ndarray):
                return self._format_numpy_array(result, result_type)
            
            # Handle basic Python types
            elif isinstance(result, (str, int, float, bool)):
                return self._format_scalar(result)
            
            # Handle sequences (lists, tuples)
            elif isinstance(result, (list, tuple)):
                return self._format_sequence(result, result_type)
            
            # Handle dictionaries
            elif isinstance(result, dict):
                return self._format_dictionary(result, result_type)
            
            # Handle datetime objects
            elif isinstance(result, (datetime, date)):
                return self._format_datetime(result)
            
            # Handle complex objects (try to extract meaningful info)
            else:
                return self._format_complex_object(result, result_type)
                
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            # Fallback to safe string representation
            return f"Analysis completed. Result: {self._safe_str_representation(result)}"
    
    def _format_dataframe(self, df: pd.DataFrame, result_type: str) -> Union[List[Dict], str]:
        """Format pandas DataFrame for user display"""
        if df.empty:
            return "No data found matching your criteria"
        
        rows, cols = df.shape
        
        # Handle very large DataFrames
        if rows > self.config['max_table_rows']:
            sample_df = df.head(self.config['max_table_rows'])
            formatted_data = self._df_to_user_friendly_records(sample_df)
            formatted_data.append({
                "_note": f"📊 Showing first {self.config['max_table_rows']:,} of {rows:,} total rows",
                "_type": "system_message"
            })
            return formatted_data
        
        # Handle too many columns
        if cols > self.config['max_table_cols']:
            # Select most important columns
            important_cols = self._select_important_columns(df)
            df_subset = df[important_cols]
            formatted_data = self._df_to_user_friendly_records(df_subset)
            formatted_data.append({
                "_note": f"📊 Showing {len(important_cols)} most relevant of {cols} total columns",
                "_type": "system_message"
            })
            return formatted_data
        
        return self._df_to_user_friendly_records(df)
    
    def _format_series(self, series: pd.Series, result_type: str) -> Union[List[Dict], str, float]:
        """Format pandas Series for user display"""
        if len(series) == 0:
            return "No data found"
        
        # Single value series
        if len(series) == 1:
            value = series.iloc[0]
            if pd.isna(value):
                return "No data available"
            return self._format_scalar(value)
        
        # Multiple values - convert to user-friendly format
        if len(series) <= self.config['summary_threshold']:
            # Small series - show as key-value pairs
            result = []
            for idx, val in series.items():
                result.append({
                    "Item": str(idx),
                    "Value": self._format_scalar(val)
                })
            return result
        else:
            # Large series - provide summary
            non_null_count = series.count()
            if pd.api.types.is_numeric_dtype(series):
                stats = {
                    "Count": non_null_count,
                    "Mean": round(series.mean(), self.config['decimal_places']),
                    "Min": self._format_scalar(series.min()),
                    "Max": self._format_scalar(series.max())
                }
                if len(series) <= self.config['max_list_items']:
                    stats["All_Values"] = [self._format_scalar(x) for x in series.tolist()]
                return [{"Statistic": k, "Value": v} for k, v in stats.items()]
            else:
                # Non-numeric series
                value_counts = series.value_counts().head(10)
                return [{"Value": str(val), "Count": count} for val, count in value_counts.items()]
    
    def _format_numpy_array(self, arr: np.ndarray, result_type: str) -> Union[List[Dict], str]:
        """Format numpy array for user display"""
        if arr.size == 0:
            return "Empty array result"
        
        # Single value
        if arr.size == 1:
            return self._format_scalar(arr.item())
        
        # Small 1D arrays
        if arr.ndim == 1 and arr.size <= self.config['max_list_items']:
            return [{"Index": i, "Value": self._format_scalar(val)} for i, val in enumerate(arr)]
        
        # Large or multi-dimensional arrays - provide summary
        return f"Array result: {arr.shape} shape, {arr.dtype} type. " + \
               f"Values range from {self._format_scalar(arr.min())} to {self._format_scalar(arr.max())}"
    
    def _format_scalar(self, value: Union[str, int, float, bool, np.number]) -> Union[str, int, float, bool]:
        """Format scalar values for user display - FIXED to handle DataFrames"""
        
        # CRITICAL FIX: Handle DataFrames/Series that shouldn't be here
        if isinstance(value, (pd.DataFrame, pd.Series)):
            logger.warning(f"DataFrame/Series passed to _format_scalar, redirecting to proper handler")
            if isinstance(value, pd.DataFrame):
                return self._format_dataframe(value, 'auto')
            else:
                return self._format_series(value, 'auto')
        
        # CRITICAL FIX: Handle numpy arrays that shouldn't be here
        if isinstance(value, np.ndarray):
            logger.warning(f"NumPy array passed to _format_scalar, redirecting to proper handler")
            return self._format_numpy_array(value, 'auto')
        
        # Now safe to check for NaN on scalar values only
        try:
            if pd.isna(value):
                return "N/A"
        except (TypeError, ValueError):
            # If pd.isna fails, the value is not a scalar pandas can handle
            # Fall back to string representation
            return self._safe_str_representation(value)
        
        if isinstance(value, (np.integer, int)):
            if abs(value) >= self.config['large_number_threshold']:
                return f"{value:,}"  # Add comma separators for large numbers
            return int(value)
        
        elif isinstance(value, (np.floating, float)):
            if abs(value) >= self.config['large_number_threshold']:
                return f"{value:,.{self.config['decimal_places']}f}"
            return round(float(value), self.config['decimal_places'])
        
        elif isinstance(value, bool):
            return value
        
        elif isinstance(value, str):
            if len(value) > self.config['max_string_length']:
                return value[:self.config['max_string_length']] + "..."
            return value
        
        else:
            return self._safe_str_representation(value)
    
    def _format_sequence(self, seq: Union[List, Tuple], result_type: str) -> Union[List[Dict], str]:
        """Format lists and tuples for user display"""
        if len(seq) == 0:
            return "Empty list result"
        
        # Single item
        if len(seq) == 1:
            # CRITICAL FIX: Don't pass complex objects to _format_scalar
            item = seq[0]
            if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                return self.format_result_for_user(item, result_type)
            else:
                return self._format_scalar(item)
        
        # Check if it's a list of similar items that can be formatted as table
        if self._is_homogeneous_sequence(seq):
            # Homogeneous sequence - format as table
            if len(seq) <= self.config['max_list_items']:
                if isinstance(seq[0], dict):
                    # List of dictionaries - already table-ready
                    return [self._clean_dict_for_display(item) for item in seq]
                else:
                    # List of simple values - but check for complex types first
                    formatted_items = []
                    for i, item in enumerate(seq):
                        if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                            # Complex item - summarize it
                            formatted_value = self._summarize_complex_value(item)
                        else:
                            formatted_value = self._format_scalar(item)
                        formatted_items.append({"Index": i+1, "Value": formatted_value})
                    return formatted_items
            else:
                # Too many items - provide summary
                sample_items = []
                for i, item in enumerate(seq[:self.config['summary_threshold']]):
                    if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                        formatted_value = self._summarize_complex_value(item)
                    else:
                        formatted_value = self._format_scalar(item)
                    sample_items.append({"Index": i+1, "Value": formatted_value})
                
                sample_items.append({
                    "_note": f"... and {len(seq) - len(sample_items) + 1} more items", 
                    "_type": "system_message"
                })
                return sample_items
        else:
            # Heterogeneous sequence - format each item
            if len(seq) <= self.config['summary_threshold']:
                formatted_items = []
                for i, item in enumerate(seq):
                    if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                        formatted_value = self._summarize_complex_value(item)
                    else:
                        formatted_value = str(self._format_scalar(item))
                    formatted_items.append({"Index": i+1, "Value": formatted_value})
                return formatted_items
            else:
                # Summarize large heterogeneous sequences
                first_few = []
                for item in seq[:3]:
                    if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                        first_few.append(self._summarize_complex_value(item))
                    else:
                        first_few.append(str(item))
                return f"List with {len(seq)} diverse items. First few: {first_few}"
    
    def _format_dictionary(self, d: Dict, result_type: str) -> Union[List[Dict], Dict]:
        """Format dictionaries for user display - PRESERVES ALL DATA including DataFrames"""
        if len(d) == 0:
            return "Empty result dictionary"
        
        # ENHANCED: Create structured result that preserves all data
        formatted_result = {
            "_type": "multi_section_result",
            "_sections": {}
        }
        
        for key, value in d.items():
            section_key = str(key).replace('_', ' ').title()
            
            # CRITICAL FIX: Preserve all DataFrame data while making it user-friendly
            if isinstance(value, pd.DataFrame):
                if value.empty:
                    formatted_result["_sections"][section_key] = {
                        "_type": "empty_data",
                        "_content": "No data available"
                    }
                else:
                    # Handle large DataFrames appropriately
                    if len(value) > self.config['max_dataframe_rows_in_dict']:
                        # Large DataFrame - provide sample + summary
                        sample_data = self._df_to_user_friendly_records(value.head(self.config['max_dataframe_rows_in_dict']))
                        sample_data.append({
                            "_note": f"📊 Showing first {self.config['max_dataframe_rows_in_dict']} of {len(value)} total rows",
                            "_type": "system_message",
                            "_full_data_available": True
                        })
                        
                        formatted_result["_sections"][section_key] = {
                            "_type": "large_data_table",
                            "_content": sample_data,
                            "_metadata": {
                                "rows": len(value),
                                "columns": len(value.columns),
                                "rows_shown": self.config['max_dataframe_rows_in_dict'],
                                "description": f"Large table with {len(value)} rows and {len(value.columns)} columns (sample shown)"
                            }
                        }
                    else:
                        # Normal size DataFrame - convert to user-friendly table format
                        table_data = self._df_to_user_friendly_records(value)
                        formatted_result["_sections"][section_key] = {
                            "_type": "data_table",
                            "_content": table_data,
                            "_metadata": {
                                "rows": len(value),
                                "columns": len(value.columns),
                                "description": f"Table with {len(value)} rows and {len(value.columns)} columns"
                            }
                        }
            
            elif isinstance(value, pd.Series):
                if len(value) == 0:
                    formatted_result["_sections"][section_key] = {
                        "_type": "empty_data", 
                        "_content": "No data available"
                    }
                else:
                    # Convert Series to appropriate format
                    series_data = self._format_series(value, 'auto')
                    formatted_result["_sections"][section_key] = {
                        "_type": "data_series",
                        "_content": series_data,
                        "_metadata": {
                            "length": len(value),
                            "description": f"Series with {len(value)} values"
                        }
                    }
            
            elif isinstance(value, list):
                # Preserve lists (like insights)
                formatted_result["_sections"][section_key] = {
                    "_type": "list_data",
                    "_content": value,
                    "_metadata": {
                        "count": len(value),
                        "description": f"List with {len(value)} items"
                    }
                }
            
            elif isinstance(value, dict):
                # Recursive formatting for nested dicts
                nested_result = self._format_dictionary(value, result_type)
                formatted_result["_sections"][section_key] = {
                    "_type": "nested_dict",
                    "_content": nested_result,
                    "_metadata": {
                        "keys": len(value),
                        "description": f"Dictionary with {len(value)} items"
                    }
                }
            
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to list format
                if value.size <= self.config['max_list_items']:
                    array_data = value.tolist()
                else:
                    array_data = value.flatten()[:self.config['max_list_items']].tolist()
                    array_data.append(f"... and {value.size - self.config['max_list_items']} more values")
                
                formatted_result["_sections"][section_key] = {
                    "_type": "array_data",
                    "_content": array_data,
                    "_metadata": {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "description": f"Array with shape {value.shape}"
                    }
                }
            
            else:
                # Simple scalar values (strings, numbers, etc.)
                formatted_result["_sections"][section_key] = {
                    "_type": "scalar_value",
                    "_content": self._format_scalar(value),
                    "_metadata": {
                        "type": type(value).__name__,
                        "description": f"{type(value).__name__} value"
                    }
                }
        
        # If this is a simple dictionary with just a few scalar values, use simpler format
        if all(section["_type"] == "scalar_value" for section in formatted_result["_sections"].values()) and len(d) <= 5:
            simple_result = []
            for section_name, section_data in formatted_result["_sections"].items():
                simple_result.append({
                    "Attribute": section_name,
                    "Value": section_data["_content"]
                })
            return simple_result
        
        return formatted_result
    
    def _format_datetime(self, dt: Union[datetime, date]) -> str:
        """Format datetime objects for user display"""
        if isinstance(dt, datetime):
            return dt.strftime(self.config['datetime_format'])
        else:
            return dt.strftime(self.config['date_format'])
    
    def _format_complex_object(self, obj: Any, result_type: str) -> str:
        """Format complex/unknown objects for user display"""
        obj_type = type(obj).__name__
        
        # Try to extract meaningful information
        if hasattr(obj, '__dict__'):
            # Object with attributes
            attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            if attrs:
                return f"{obj_type} object with attributes: {list(attrs.keys())[:5]}"
        
        # Try to get string representation
        str_repr = self._safe_str_representation(obj)
        if len(str_repr) > 200:
            return f"{obj_type} object: {str_repr[:200]}..."
        
        return f"{obj_type} object: {str_repr}"
    
    def _df_to_user_friendly_records(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to user-friendly list of dictionaries"""
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, value in row.items():
                # Clean column names
                clean_col = str(col).replace('_', ' ').title()
                # Format values
                record[clean_col] = self._format_scalar(value)
            records.append(record)
        return records
    
    def _select_important_columns(self, df: pd.DataFrame) -> List[str]:
        """Select most important columns when there are too many"""
        columns = df.columns.tolist()
        
        # Prioritize columns with certain characteristics
        priority_score = {}
        
        for col in columns:
            score = 0
            col_lower = str(col).lower()
            
            # Higher priority for key-like columns
            if any(word in col_lower for word in ['id', 'name', 'key', 'code']):
                score += 10
            
            # Higher priority for non-null columns
            non_null_ratio = df[col].count() / len(df)
            score += non_null_ratio * 5
            
            # Higher priority for columns with good data variety
            if df[col].dtype in ['object', 'string']:
                unique_ratio = df[col].nunique() / df[col].count() if df[col].count() > 0 else 0
                score += min(unique_ratio * 3, 3)  # Cap at 3
            elif pd.api.types.is_numeric_dtype(df[col]):
                score += 3  # Numeric columns are generally important
            
            priority_score[col] = score
        
        # Sort by priority and take top columns
        sorted_cols = sorted(priority_score.items(), key=lambda x: x[1], reverse=True)
        return [col for col, _ in sorted_cols[:self.config['max_table_cols']]]
    
    def _is_homogeneous_sequence(self, seq: Union[List, Tuple]) -> bool:
        """Check if sequence contains similar types of items"""
        if len(seq) <= 1:
            return True
        
        first_type = type(seq[0])
        return all(isinstance(item, first_type) for item in seq[:10])  # Check first 10 items
    
    def _clean_dict_for_display(self, d: Dict) -> Dict:
        """Clean dictionary for user display - FIXED to handle complex values"""
        cleaned = {}
        for key, value in d.items():
            clean_key = str(key).replace('_', ' ').title()
            
            # Handle complex values properly
            if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                cleaned[clean_key] = self._summarize_complex_value(value)
            else:
                cleaned[clean_key] = self._format_scalar(value)
        return cleaned
    
    def _summarize_complex_value(self, value: Any) -> str:
        """Create summary for complex nested values - ENHANCED"""
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return "Empty DataFrame"
            return f"DataFrame: {value.shape[0]} rows × {value.shape[1]} columns"
        elif isinstance(value, pd.Series):
            if len(value) == 0:
                return "Empty Series"
            return f"Series with {len(value)} values"
        elif isinstance(value, np.ndarray):
            return f"Array: {value.shape} shape, {value.dtype} type"
        elif isinstance(value, dict):
            return f"Dictionary with {len(value)} items"
        elif isinstance(value, (list, tuple)):
            return f"List with {len(value)} items"
        else:
            return str(type(value).__name__)
    
    def _safe_str_representation(self, obj: Any) -> str:
        """Get safe string representation of any object"""
        try:
            # Special handling for pandas objects
            if isinstance(obj, pd.DataFrame):
                if obj.empty:
                    return "Empty DataFrame"
                return f"DataFrame with {obj.shape[0]} rows and {obj.shape[1]} columns"
            elif isinstance(obj, pd.Series):
                if len(obj) == 0:
                    return "Empty Series"
                return f"Series with {len(obj)} values"
            elif isinstance(obj, np.ndarray):
                return f"NumPy array with shape {obj.shape}"
            else:
                return str(obj)
        except Exception:
            return f"<{type(obj).__name__} object>"


# Global formatter instance
_formatter = ResultFormatter()

def format_result_for_user(result: Any, result_type: str = 'auto', config=None) -> Union[str, List[Dict], Dict]:
    """
    Main function to format any Python object for user-friendly display
    
    Args:
        result: The Python object to format
        result_type: Hint about expected result type ('table', 'chart', 'value', 'auto')
        config: Optional configuration dict
        
    Returns:
        User-friendly formatted result
    """
    global _formatter
    if config:
        _formatter = ResultFormatter(config)
    
    formatted = _formatter.format_result_for_user(result, result_type)
    
    # Handle structured multi-section results
    if isinstance(formatted, dict) and formatted.get("_type") == "multi_section_result":
        return format_multi_section_result(formatted)
    
    return formatted


def format_multi_section_result(structured_result: Dict) -> Dict:
    """
    Format multi-section results for optimal frontend display
    
    Args:
        structured_result: Result with _type="multi_section_result"
        
    Returns:
        Optimized result for frontend consumption
    """
    sections = structured_result.get("_sections", {})
    
    if len(sections) == 1:
        # Single section - return the content directly
        section_data = list(sections.values())[0]
        return section_data["_content"]
    
    # Multiple sections - create organized output
    result = {
        "_type": "analysis_result",
        "_summary": f"Analysis result with {len(sections)} sections",
        "sections": {}
    }
    
    for section_name, section_data in sections.items():
        section_type = section_data["_type"]
        section_content = section_data["_content"]
        section_metadata = section_data.get("_metadata", {})
        
        if section_type == "data_table":
            # For data tables, include both data and metadata
            result["sections"][section_name] = {
                "type": "table",
                "data": section_content,
                "summary": section_metadata.get("description", "Data table")
            }
        
        elif section_type == "large_data_table":
            # For large data tables, include sample data with metadata
            result["sections"][section_name] = {
                "type": "large_table", 
                "data": section_content,
                "summary": section_metadata.get("description", "Large data table"),
                "total_rows": section_metadata.get("rows", "unknown"),
                "rows_shown": section_metadata.get("rows_shown", "unknown")
            }
        
        elif section_type == "list_data":
            # For lists (like insights), format appropriately
            result["sections"][section_name] = {
                "type": "list",
                "data": section_content,
                "summary": section_metadata.get("description", "List of items")
            }
        
        elif section_type == "scalar_value":
            # For simple values
            result["sections"][section_name] = {
                "type": "value",
                "data": section_content,
                "summary": f"Single {section_metadata.get('type', 'value')}"
            }
        
        elif section_type == "data_series":
            # For series data
            result["sections"][section_name] = {
                "type": "series",
                "data": section_content,
                "summary": section_metadata.get("description", "Data series")
            }
        
        else:
            # Fallback for other types
            result["sections"][section_name] = {
                "type": "other",
                "data": section_content,
                "summary": section_metadata.get("description", "Data")
            }
    
    return result


def format_plot_result(result: Any, plot_data: Dict) -> Union[str, List[Dict]]:
    """
    Special formatter for results that come with plots
    
    Args:
        result: The result object
        plot_data: The plot/chart data
        
    Returns:
        Formatted result optimized for display alongside charts
    """
    if result is None or (isinstance(result, str) and not result.strip()):
        return "Chart generated successfully"
    
    # For plot results, be more concise
    if isinstance(result, pd.DataFrame) and len(result) > 20:
        # Large DataFrames with plots - show summary
        return f"📊 Chart shows data from table with {len(result):,} rows and {len(result.columns)} columns"
    
    return format_result_for_user(result, 'chart')


def validate_result_user_friendliness(result: Any) -> Tuple[bool, str]:
    """
    Validate if a result is user-friendly enough for frontend display
    
    Args:
        result: The result to validate
        
    Returns:
        (is_user_friendly, reason_if_not)
    """
    # Check for problematic types that should not reach frontend
    problematic_types = (
        type, type(lambda: None), type(iter([])),  # functions, iterators, etc.
        type(Exception()), 
    )
    
    if isinstance(result, problematic_types):
        return False, f"Result contains non-user-friendly type: {type(result)}"
    
    # Check for very large raw dictionaries
    if isinstance(result, dict) and len(result) > 50:
        return False, "Large raw dictionary should be formatted as table"
    
    # Check for nested complex structures
    if isinstance(result, (list, tuple)):
        if len(result) > 0 and isinstance(result[0], dict):
            # List of dicts is okay (table format)
            return True, ""
        elif any(isinstance(item, (dict, list, tuple)) for item in result):
            return False, "Nested complex structures need formatting"
    
    return True, ""


def flatten_multi_section_result(structured_result: Dict, section_name: str = None) -> Union[List[Dict], str, Dict]:
    """
    Extract a specific section from multi-section result or flatten to simple format
    
    Args:
        structured_result: Result with _type="multi_section_result"
        section_name: Optional - name of specific section to extract
        
    Returns:
        Flattened result suitable for simple frontend display
    """
    if not isinstance(structured_result, dict) or structured_result.get("_type") != "multi_section_result":
        return structured_result
    
    sections = structured_result.get("_sections", {})
    
    if section_name:
        # Extract specific section
        if section_name in sections:
            return sections[section_name]["_content"]
        else:
            # Try case-insensitive match
            for key, value in sections.items():
                if key.lower() == section_name.lower():
                    return value["_content"]
            return f"Section '{section_name}' not found"
    
    # Flatten all sections to simple format
    if len(sections) == 1:
        # Single section - return content directly
        return list(sections.values())[0]["_content"]
    
    # Multiple sections - create simple combined result
    flattened = {}
    for section_name, section_data in sections.items():
        flattened[section_name] = section_data["_content"]
    
    return flattened


def get_result_summary(result: Any) -> str:
    """
    Get a brief text summary of any result type
    
    Args:
        result: Any formatted result
        
    Returns:
        Brief text summary of the result
    """
    if isinstance(result, dict):
        if result.get("_type") == "multi_section_result":
            sections = result.get("_sections", {})
            summaries = []
            for name, data in sections.items():
                metadata = data.get("_metadata", {})
                desc = metadata.get("description", f"{name} section")
                summaries.append(desc)
            return f"Analysis with {len(sections)} sections: " + ", ".join(summaries)
        
        elif result.get("_type") == "analysis_result":
            return result.get("_summary", "Multi-section analysis result")
        
        elif len(result) <= 5:
            return f"Result with {len(result)} attributes"
        else:
            return f"Complex result with {len(result)} sections"
    
    elif isinstance(result, list):
        if len(result) == 0:
            return "Empty result"
        elif len(result) == 1:
            return "Single item result"
        elif all(isinstance(item, dict) for item in result):
            return f"Table with {len(result)} rows"
        else:
            return f"List with {len(result)} items"
    
    elif isinstance(result, str):
        if len(result) > 100:
            return f"Text result ({len(result)} characters)"
        else:
            return "Text result"
    
    else:
        return f"Result of type {type(result).__name__}"