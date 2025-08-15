"""
Text Standardization Module - Conservative Version
==================================================

A conservative text standardization module that only fixes obvious typos
and case inconsistencies while preserving all semantic differences.

Rules:
1. Skip values containing any numbers/digits
2. Skip values that are ALL UPPERCASE or all lowercase
3. Only standardize mixed-case pure text values
4. Use very high similarity thresholds (95%+)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter
import logging
from fuzzywuzzy import fuzz
import re

logger = logging.getLogger(__name__)

class ColumnStandardizationResult:
    """Container for column standardization results"""
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.original_unique_count = 0
        self.final_unique_count = 0
        self.variants_found = 0
        self.mappings = {}  # original -> standardized
        self.standardization_applied = False
        self.skip_reason = None
        
class DatasetStandardizationResult:
    """Container for complete dataset standardization results"""
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.columns_analyzed = 0
        self.columns_standardized = 0
        self.columns_skipped = 0
        self.total_variants_resolved = 0
        self.column_results = {}  # column_name -> ColumnStandardizationResult
        self.processing_time = 0
        self.success = True
        self.errors = []

class ConservativeTextStandardizer:
    """Conservative text standardization that only fixes obvious typos"""
    
    def __init__(self, 
                 similarity_threshold: int = 95,
                 min_column_size: int = 5,
                 max_unique_ratio: float = 0.5):
        """
        Initialize the conservative standardizer.
        
        Args:
            similarity_threshold: Very high threshold for safety (default 95%)
            min_column_size: Minimum non-null values in a column
            max_unique_ratio: Maximum ratio of unique values to total rows
        """
        self.similarity_threshold = similarity_threshold
        self.min_column_size = min_column_size
        self.max_unique_ratio = max_unique_ratio
    
    def should_skip_value(self, value: Any) -> Tuple[bool, str]:
        """
        Check if a value should be skipped from standardization.
        
        Returns:
            (should_skip, reason)
        """
        if pd.isna(value):
            return True, "null_value"
        
        value_str = str(value)
        
        # Rule 1 & 4: Skip if contains any digits/numbers
        if any(char.isdigit() for char in value_str):
            return True, "contains_numbers"
        
        # Rule 3: Skip if ALL UPPERCASE or all lowercase
        # We want mixed case for standardization
        if value_str.isupper():
            return True, "all_uppercase"
        if value_str.islower():
            return True, "all_lowercase"
        
        # Skip if too short (less likely to be a meaningful entity)
        if len(value_str.strip()) < 3:
            return True, "too_short"
        
        # Skip if it looks like a code/identifier (mix of uppercase and special chars)
        if re.match(r'^[A-Z_\-]+$', value_str):
            return True, "looks_like_code"
        
        return False, "ok"
    
    def should_standardize_column(self, series: pd.Series, column_name: str) -> Tuple[bool, str]:
        """
        Determine if a column should be standardized based on conservative rules.
        """
        # Skip numeric columns entirely
        if pd.api.types.is_numeric_dtype(series):
            return False, "numeric_column"
        
        # Skip datetime columns
        if pd.api.types.is_datetime64_any_dtype(series):
            return False, "datetime_column"
        
        # Check if column has enough data
        non_null_count = series.notna().sum()
        if non_null_count < self.min_column_size:
            return False, f"too_few_values ({non_null_count})"
        
        # Check unique value ratio
        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)
        
        if unique_ratio > self.max_unique_ratio:
            return False, f"too_many_unique ({unique_ratio:.1%})"
        
        # Sample the values to check if they're suitable for standardization
        sample_values = series.dropna().head(20)
        skippable_values = 0
        
        for value in sample_values:
            should_skip, _ = self.should_skip_value(value)
            if should_skip:
                skippable_values += 1
        
        # If most sample values should be skipped, skip the whole column
        if skippable_values > len(sample_values) * 0.7:
            return False, "mostly_non_standardizable_values"
        
        # Check if column name suggests it shouldn't be standardized
        col_lower = column_name.lower()
        skip_patterns = ['id', 'code', 'number', 'date', 'time', 'year', 'month', 'price', 'amount', 'quantity']
        if any(pattern in col_lower for pattern in skip_patterns):
            return False, f"column_name_suggests_skip ({column_name})"
        
        return True, "suitable_for_standardization"
    
    def is_safe_to_merge(self, value1: str, value2: str) -> bool:
        """
        Very conservative check if two values can be merged.
        Only merge if they're nearly identical.
        """
        # Both must be mixed case text without numbers
        skip1, _ = self.should_skip_value(value1)
        skip2, _ = self.should_skip_value(value2)
        
        if skip1 or skip2:
            return False
        
        val1_lower = value1.lower()
        val2_lower = value2.lower()
        
        # Must have very high similarity
        similarity = fuzz.ratio(val1_lower, val2_lower)
        if similarity < self.similarity_threshold:
            return False
        
        # Additional safety checks
        
        # Don't merge if word count is different
        words1 = val1_lower.split()
        words2 = val2_lower.split()
        if len(words1) != len(words2):
            return False
        
        # Don't merge if any complete word is different
        if set(words1) != set(words2):
            return False
        
        # At this point, they have the same words, just potentially different:
        # - Case (L'oreal vs L'Oreal)
        # - Punctuation/apostrophes (L'oreal vs L'oreal)
        # - Minor spacing issues
        # These are safe to standardize
        
        return True
    
    def find_variants_conservative(self, values: List[Any], column_name: str) -> Dict[Any, Any]:
        """
        Find variants using very conservative matching.
        Only fixes obvious case/punctuation issues in mixed-case text.
        """
        # Count frequencies
        value_counts = Counter(values)
        
        # Filter to only mixed-case text values without numbers
        valid_values = []
        for v in value_counts.keys():
            if pd.notna(v):
                should_skip, _ = self.should_skip_value(v)
                if not should_skip:
                    valid_values.append(v)
        
        if not valid_values:
            return {}
        
        # Sort by frequency (most common = likely correct)
        valid_values = sorted(valid_values, key=lambda x: value_counts[x], reverse=True)
        
        # Build mapping
        mapping = {}
        already_mapped = set()
        
        for standard_value in valid_values:
            if standard_value in already_mapped:
                continue
            
            mapping[standard_value] = standard_value
            already_mapped.add(standard_value)
            
            # Only look for nearly identical variants
            for candidate_value in valid_values:
                if candidate_value in already_mapped or candidate_value == standard_value:
                    continue
                
                # Check if safe to merge (very conservative)
                if self.is_safe_to_merge(str(standard_value), str(candidate_value)):
                    mapping[candidate_value] = standard_value
                    already_mapped.add(candidate_value)
                    logger.debug(f"Column '{column_name}': '{candidate_value}' -> '{standard_value}'")
        
        return mapping
    
    def standardize_column(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, ColumnStandardizationResult]:
        """
        Standardize a single column conservatively.
        """
        result = ColumnStandardizationResult(column_name)
        result.original_unique_count = series.nunique()
        
        # Check if column should be standardized
        should_standardize, reason = self.should_standardize_column(series, column_name)
        
        if not should_standardize:
            logger.info(f"Skipping column '{column_name}': {reason}")
            result.standardization_applied = False
            result.skip_reason = reason
            return series.copy(), result
        
        logger.info(f"Analyzing column '{column_name}' for standardization")
        
        # Get all values (including duplicates for frequency)
        all_values = series.dropna().tolist()
        
        # Find variants conservatively
        mapping = self.find_variants_conservative(all_values, column_name)
        
        # Count actual changes
        changes = {k: v for k, v in mapping.items() if k != v}
        
        if not changes:
            logger.info(f"No safe standardizations found in column '{column_name}'")
            result.standardization_applied = False
            result.skip_reason = "no_safe_variants"
            return series.copy(), result
        
        # Apply mapping
        standardized_series = series.map(lambda x: mapping.get(x, x))
        
        # Update result
        result.final_unique_count = standardized_series.nunique()
        result.variants_found = len(changes)
        result.mappings = changes
        result.standardization_applied = True
        
        logger.info(f"Column '{column_name}': Safely standardized {result.variants_found} variants")
        
        return standardized_series, result
    
    def standardize_dataframe(self, df: pd.DataFrame, 
                             dataset_name: str = "dataset",
                             columns_to_standardize: Optional[List[str]] = None) -> Tuple[pd.DataFrame, DatasetStandardizationResult]:
        """
        Standardize a DataFrame conservatively.
        """
        import time
        start_time = time.time()
        
        result = DatasetStandardizationResult(dataset_name)
        standardized_df = df.copy()
        
        # Determine columns to process
        if columns_to_standardize:
            columns = [col for col in columns_to_standardize if col in df.columns]
        else:
            columns = df.columns.tolist()
        
        result.columns_analyzed = len(columns)
        
        # Process each column
        for column in columns:
            try:
                standardized_series, col_result = self.standardize_column(df[column], column)
                
                if col_result.standardization_applied:
                    standardized_df[column] = standardized_series
                    result.columns_standardized += 1
                    result.total_variants_resolved += col_result.variants_found
                else:
                    result.columns_skipped += 1
                
                result.column_results[column] = col_result
                
            except Exception as e:
                logger.error(f"Error standardizing column '{column}': {e}")
                result.errors.append(f"Column '{column}': {str(e)}")
                result.columns_skipped += 1
        
        result.processing_time = time.time() - start_time
        result.success = len(result.errors) == 0
        
        logger.info(f"Dataset '{dataset_name}': Standardized {result.columns_standardized}/{result.columns_analyzed} columns, "
                   f"skipped {result.columns_skipped}, resolved {result.total_variants_resolved} variants")
        
        return standardized_df, result
    
    def standardize_worksheet_dict(self, worksheets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DatasetStandardizationResult]]:
        """
        Standardize multiple worksheets conservatively.
        """
        standardized_worksheets = {}
        results = {}
        
        for worksheet_name, df in worksheets.items():
            logger.info(f"Processing worksheet: {worksheet_name}")
            standardized_df, result = self.standardize_dataframe(df, worksheet_name)
            standardized_worksheets[worksheet_name] = standardized_df
            results[worksheet_name] = result
        
        return standardized_worksheets, results
    
    def generate_standardization_report(self, results: Union[DatasetStandardizationResult, Dict[str, DatasetStandardizationResult]]) -> str:
        """
        Generate a standardization report.
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TEXT STANDARDIZATION REPORT (Conservative Mode)")
        report_lines.append("=" * 60)
        
        if isinstance(results, DatasetStandardizationResult):
            results = {"Dataset": results}
        
        total_variants = 0
        total_columns_standardized = 0
        total_columns_skipped = 0
        
        for dataset_name, result in results.items():
            report_lines.append(f"\nDataset: {dataset_name}")
            report_lines.append("-" * 40)
            
            if not result.success:
                report_lines.append(f"⚠️ Errors occurred: {', '.join(result.errors)}")
            
            report_lines.append(f"Columns analyzed: {result.columns_analyzed}")
            report_lines.append(f"Columns standardized: {result.columns_standardized}")
            report_lines.append(f"Columns skipped: {result.columns_skipped}")
            report_lines.append(f"Total variants resolved: {result.total_variants_resolved}")
            
            # Show why columns were skipped
            skip_reasons = {}
            for col_name, col_result in result.column_results.items():
                if not col_result.standardization_applied and col_result.skip_reason:
                    reason = col_result.skip_reason
                    if reason not in skip_reasons:
                        skip_reasons[reason] = []
                    skip_reasons[reason].append(col_name)
            
            if skip_reasons:
                report_lines.append("\nSkipped columns by reason:")
                for reason, cols in skip_reasons.items():
                    if len(cols) <= 3:
                        report_lines.append(f"  • {reason}: {', '.join(cols)}")
                    else:
                        report_lines.append(f"  • {reason}: {', '.join(cols[:3])} + {len(cols)-3} more")
            
            if result.columns_standardized > 0:
                report_lines.append("\nStandardized columns:")
                
                for col_name, col_result in result.column_results.items():
                    if col_result.standardization_applied:
                        report_lines.append(f"\n  📊 {col_name}:")
                        report_lines.append(f"     Original unique values: {col_result.original_unique_count}")
                        report_lines.append(f"     Final unique values: {col_result.final_unique_count}")
                        report_lines.append(f"     Variants resolved: {col_result.variants_found}")
                        
                        if col_result.mappings:
                            report_lines.append("     Standardizations:")
                            for original, standard in list(col_result.mappings.items())[:5]:
                                report_lines.append(f"       '{original}' → '{standard}'")
                            
                            if len(col_result.mappings) > 5:
                                report_lines.append(f"       ... and {len(col_result.mappings) - 5} more")
            
            total_variants += result.total_variants_resolved
            total_columns_standardized += result.columns_standardized
            total_columns_skipped += result.columns_skipped
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"Total datasets processed: {len(results)}")
        report_lines.append(f"Total columns standardized: {total_columns_standardized}")
        report_lines.append(f"Total columns skipped: {total_columns_skipped}")
        report_lines.append(f"Total variants resolved: {total_variants}")
        report_lines.append("\nNote: Only mixed-case text without numbers was standardized.")
        
        return "\n".join(report_lines)
    
    def get_standardization_summary(self, results: Union[DatasetStandardizationResult, Dict[str, DatasetStandardizationResult]]) -> Dict[str, Any]:
        """
        Get a summary of standardization results.
        """
        if isinstance(results, DatasetStandardizationResult):
            results = {"Dataset": results}
        
        summary = {
            'datasets_processed': len(results),
            'total_columns_analyzed': sum(r.columns_analyzed for r in results.values()),
            'total_columns_standardized': sum(r.columns_standardized for r in results.values()),
            'total_columns_skipped': sum(r.columns_skipped for r in results.values()),
            'total_variants_resolved': sum(r.total_variants_resolved for r in results.values()),
            'total_processing_time': sum(r.processing_time for r in results.values()),
            'success': all(r.success for r in results.values()),
            'datasets': {}
        }
        
        for dataset_name, result in results.items():
            summary['datasets'][dataset_name] = {
                'columns_standardized': result.columns_standardized,
                'columns_skipped': result.columns_skipped,
                'variants_resolved': result.total_variants_resolved,
                'standardized_columns': [col for col, res in result.column_results.items() 
                                        if res.standardization_applied]
            }
        
        return summary

# Convenience function using the new conservative standardizer
def standardize_data(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     similarity_threshold: int = 95,
                     columns_to_standardize: Optional[List[str]] = None,
                     verbose: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function for conservative text standardization.
    
    Only standardizes mixed-case text without numbers.
    Skips ALL CAPS, all lowercase, and any values with digits.
    
    Args:
        data: DataFrame or dictionary of DataFrames
        similarity_threshold: Very high threshold for safety (95%+)
        columns_to_standardize: Specific columns to process
        verbose: Whether to print report
        
    Returns:
        Conservatively standardized data
    """
    standardizer = ConservativeTextStandardizer(similarity_threshold=similarity_threshold)
    
    if isinstance(data, pd.DataFrame):
        standardized_df, result = standardizer.standardize_dataframe(
            data, "data", columns_to_standardize
        )
        
        if verbose:
            print(standardizer.generate_standardization_report(result))
        
        return standardized_df
    else:
        standardized_worksheets, results = standardizer.standardize_worksheet_dict(data)
        
        if verbose:
            print(standardizer.generate_standardization_report(results))
        
        return standardized_worksheets

# Alias for backward compatibility
TextStandardizer = ConservativeTextStandardizer