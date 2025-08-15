"""
Simplified Data Merger Module - No Key Column Selection
======================================================

Simplified principles:
- Always use ALL columns for duplicate detection
- Always use ALL common columns for merging (no key selection needed)
- Pairwise merging between datasets
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class DataMergeResult:
    """Container for merge operation results"""
    def __init__(self):
        self.success = False
        self.merged_df = None
        self.merge_strategy_used = None
        self.datasets_included = []
        self.datasets_excluded = []
        self.duplicate_stats = {}
        self.merge_summary = ""
        self.warnings = []
        self.errors = []

class DatasetDuplicateInfo:
    """Container for duplicate information"""
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.original_rows = 0
        self.unique_rows = 0
        self.duplicates_removed = 0
        self.strategy_used = ""

def remove_duplicates_from_dataset(df: pd.DataFrame, 
                                  dataset_name: str = "dataset",
                                  strategy: str = 'first') -> Tuple[pd.DataFrame, DatasetDuplicateInfo]:
    """Remove duplicates using ALL columns only."""
    duplicate_info = DatasetDuplicateInfo(dataset_name)
    duplicate_info.original_rows = len(df)
    duplicate_info.strategy_used = strategy
    
    if df.empty:
        duplicate_info.unique_rows = 0
        duplicate_info.duplicates_removed = 0
        return df, duplicate_info
    
    try:
        if strategy == 'strict':
            duplicated_mask = df.duplicated(keep=False)  # All columns
            cleaned_df = df[~duplicated_mask].copy()
        else:
            keep_strategy = 'first' if strategy == 'first' else 'last'
            cleaned_df = df.drop_duplicates(keep=keep_strategy)  # All columns
        
        duplicate_info.unique_rows = len(cleaned_df)
        duplicate_info.duplicates_removed = duplicate_info.original_rows - duplicate_info.unique_rows
        
        if duplicate_info.duplicates_removed > 0:
            logger.info(f"Removed {duplicate_info.duplicates_removed} duplicate rows from {dataset_name}")
        
        return cleaned_df.reset_index(drop=True), duplicate_info
        
    except Exception as e:
        logger.error(f"Error removing duplicates from {dataset_name}: {e}")
        duplicate_info.unique_rows = duplicate_info.original_rows
        duplicate_info.duplicates_removed = 0
        return df, duplicate_info

def find_common_columns_between_two(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    """Find exact column matches between two specific datasets."""
    if df1.empty or df2.empty:
        return []
    
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common = cols1.intersection(cols2)
    
    return sorted(list(common))

def order_datasets_by_size(datasets_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """Order datasets by size (largest first)."""
    dataset_sizes = {}
    for name, df in datasets_dict.items():
        if df is not None and not df.empty:
            dataset_sizes[name] = (len(df), len(df.columns))
    
    ordered_names = sorted(dataset_sizes.keys(), 
                          key=lambda x: dataset_sizes[x], 
                          reverse=True)
    
    logger.info("Dataset merge order (largest first):")
    for i, name in enumerate(ordered_names):
        rows, cols = dataset_sizes[name]
        logger.info(f"  {i+1}. '{name}': {rows:,} rows × {cols} columns")
    
    return ordered_names

def determine_merge_strategy(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    """
    INTELLIGENT: Determine merge strategy based on column overlap.
    
    Logic:
    - If ALL columns are common: concatenate (same structure, different records)
    - If SOME columns are common: join (complementary data)
    
    Returns:
        'concatenate' or 'join'
    """
    common_columns = find_common_columns_between_two(df1, df2)
    
    if not common_columns:
        return 'no_merge_possible'
    
    # Check if ALL columns are common in both datasets
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    common_cols_set = set(common_columns)
    
    # If both datasets have exactly the same columns, concatenate
    if df1_cols == df2_cols == common_cols_set:
        return 'concatenate'
    
    # If datasets have some overlapping columns but also unique columns, join
    else:
        return 'join'

def merge_two_datasets(df1: pd.DataFrame, df2: pd.DataFrame, 
                      name1: str, name2: str, 
                      strategy: str = 'auto') -> Tuple[bool, pd.DataFrame, str]:
    """
    INTELLIGENT: Merge exactly two datasets using data-driven strategy selection.
    
    Returns:
        (success, merged_df, merge_method)
    """
    try:
        # Find common columns between these two datasets only
        common_columns = find_common_columns_between_two(df1, df2)
        
        if not common_columns:
            return False, pd.DataFrame(), "no_common_columns"
        
        # Determine strategy based on data structure if auto
        if strategy == 'auto':
            strategy = determine_merge_strategy(df1, df2)
            if strategy == 'no_merge_possible':
                return False, pd.DataFrame(), "no_common_columns"
        
        logger.info(f"Merging '{name1}' + '{name2}' using {len(common_columns)} common columns: {common_columns}")
        logger.info(f"Strategy selected: {strategy}")
        
        if strategy == 'concatenate':
            # Concatenate strategy - datasets have same structure
            # Use only common columns (should be all columns in this case)
            df1_subset = df1[common_columns].copy()
            df2_subset = df2[common_columns].copy()
            
            df1_subset['_data_source'] = name1
            df2_subset['_data_source'] = name2
            
            merged_df = pd.concat([df1_subset, df2_subset], ignore_index=True, sort=False)
            merge_method = f"concatenate_all_{len(common_columns)}_cols"
            
        else:  # strategy == 'join'
            # Join strategy - datasets have complementary data
            # Use ALL common columns as join keys
            merged_df = pd.merge(
                df1,
                df2,
                on=common_columns,  # Use ALL common columns for join
                how='left',         # 'outer'?
                suffixes=('', f'_{name2}')
            )
            merge_method = f"join_on_{len(common_columns)}_cols"
        
        logger.info(f"Merge result: {df1.shape} + {df2.shape} → {merged_df.shape} using {merge_method}")
        return True, merged_df, merge_method
        
    except Exception as e:
        logger.error(f"Error merging '{name1}' and '{name2}': {e}")
        return False, pd.DataFrame(), f"error: {str(e)}"

def merge_datasets_on_common_columns(datasets_dict: Dict[str, pd.DataFrame],
                                   merge_strategy: str = 'auto',
                                   remove_duplicates_first: bool = True) -> DataMergeResult:
    """
    SIMPLIFIED: Merge datasets using pairwise merging with all common columns.
    """
    result = DataMergeResult()
    
    try:
        # Filter out empty datasets
        valid_datasets = {name: df for name, df in datasets_dict.items() 
                         if df is not None and not df.empty}
        
        if len(valid_datasets) == 0:
            result.errors.append("No valid datasets to merge")
            return result
        
        if len(valid_datasets) == 1:
            # Only one dataset - just clean it
            name, df = list(valid_datasets.items())[0]
            if remove_duplicates_first:
                cleaned_df, dup_info = remove_duplicates_from_dataset(df, name, 'first')
                result.duplicate_stats[name] = dup_info
            else:
                cleaned_df = df.copy()
            
            result.success = True
            result.merged_df = cleaned_df
            result.merge_strategy_used = 'single_dataset'
            result.datasets_included = [name]
            result.merge_summary = f"Single dataset '{name}' processed"
            return result
        
        # Remove duplicates from individual datasets if requested
        cleaned_datasets = {}
        if remove_duplicates_first:
            for name, df in valid_datasets.items():
                cleaned_df, dup_info = remove_duplicates_from_dataset(df, name, 'first')
                cleaned_datasets[name] = cleaned_df
                result.duplicate_stats[name] = dup_info
        else:
            cleaned_datasets = valid_datasets.copy()
        
        # Order datasets by size (largest first)
        ordered_names = order_datasets_by_size(cleaned_datasets)
        
        # Start with the largest dataset
        current_result = cleaned_datasets[ordered_names[0]]
        current_name = ordered_names[0]
        result.datasets_included.append(current_name)
        
        merge_methods_used = []
        
        # Pairwise merging: merge current result with each subsequent dataset
        for next_name in ordered_names[1:]:
            next_dataset = cleaned_datasets[next_name]
            
            # Always use 'auto' to let the function decide based on data structure
            # Override only if user explicitly specified concatenate or join
            if merge_strategy in ['concatenate', 'join']:
                pair_strategy = merge_strategy
            else:
                pair_strategy = 'auto'  # Let determine_merge_strategy decide
            
            # Merge current result with next dataset
            success, merged_df, merge_method = merge_two_datasets(
                current_result, next_dataset, current_name, next_name, pair_strategy
            )
            
            if success:
                current_result = merged_df
                current_name = f"{current_name}+{next_name}"
                result.datasets_included.append(next_name)
                merge_methods_used.append(merge_method)
                
                logger.info(f"Successfully merged with '{next_name}': {merged_df.shape[0]} rows")
            else:
                result.datasets_excluded.append(next_name)
                result.warnings.append(f"Failed to merge '{next_name}': {merge_method}")
                logger.warning(f"Skipping '{next_name}': {merge_method}")
        
        # Final result
        if len(result.datasets_included) >= 1:
            result.success = True
            result.merged_df = current_result
            result.merge_strategy_used = " → ".join(merge_methods_used) if merge_methods_used else "single_dataset"
            result.merge_summary = f"Pairwise merged {len(result.datasets_included)} datasets using: {result.merge_strategy_used}"
            
            # Final duplicate removal from merged data
            final_cleaned_df, final_dup_info = remove_duplicates_from_dataset(
                result.merged_df, "final_merged_data", 'first'
            )
            result.merged_df = final_cleaned_df
            result.duplicate_stats['final_merge'] = final_dup_info
            
            if final_dup_info.duplicates_removed > 0:
                result.warnings.append(f"Removed {final_dup_info.duplicates_removed} duplicate rows from final merged data")
        
        else:
            result.errors.append("No datasets could be successfully merged")
        
        return result
        
    except Exception as e:
        result.errors.append(f"Merge operation failed: {str(e)}")
        logger.error(f"Error in merge_datasets_on_common_columns: {e}")
        return result

def create_merged_excel_file(merged_df: pd.DataFrame,
                           output_path: str,
                           original_datasets_info: Dict[str, Any],
                           merge_result: DataMergeResult) -> Tuple[bool, str]:
    """Create an Excel file with the merged data and metadata."""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            merged_df.to_excel(writer, sheet_name='Merged_Data', index=False)
            
            # Create metadata sheet
            metadata_rows = []
            metadata_rows.append(['Merge Operation Summary', ''])
            metadata_rows.append(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            metadata_rows.append(['Merge Strategy', merge_result.merge_strategy_used])
            metadata_rows.append(['Total Datasets Processed', len(original_datasets_info)])
            metadata_rows.append(['Datasets Included', len(merge_result.datasets_included)])
            metadata_rows.append(['Datasets Excluded', len(merge_result.datasets_excluded)])
            metadata_rows.append(['Final Data Shape', f"{merged_df.shape[0]} rows × {merged_df.shape[1]} columns"])
            metadata_rows.append(['', ''])
            
            # Datasets included (in merge order)
            metadata_rows.append(['Datasets Included (merge order):', ''])
            for i, dataset in enumerate(merge_result.datasets_included):
                metadata_rows.append([f"{i+1}. {dataset}", 'Included'])
            
            # Duplicate removal stats
            if merge_result.duplicate_stats:
                metadata_rows.append(['', ''])
                metadata_rows.append(['Duplicate Removal Stats:', ''])
                for dataset_name, dup_info in merge_result.duplicate_stats.items():
                    metadata_rows.append([f"{dataset_name} - Original Rows", dup_info.original_rows])
                    metadata_rows.append([f"{dataset_name} - Final Rows", dup_info.unique_rows])
                    metadata_rows.append([f"{dataset_name} - Duplicates Removed", dup_info.duplicates_removed])
            
            metadata_df = pd.DataFrame(metadata_rows, columns=['Field', 'Value'])
            metadata_df.to_excel(writer, sheet_name='Merge_Metadata', index=False)
        
        return True, f"Successfully created merged Excel file: {output_path}"
        
    except Exception as e:
        error_msg = f"Failed to create merged Excel file: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def analyze_merge_feasibility(datasets_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze merge feasibility using intelligent strategy selection."""
    analysis = {
        'feasible': False,
        'recommended_strategy': 'auto',
        'total_datasets': len(datasets_dict),
        'valid_datasets': 0,
        'recommendations': [],
        'warnings': [],
        'strategy_analysis': {}
    }
    
    try:
        # Filter valid datasets
        valid_datasets = {name: df for name, df in datasets_dict.items() 
                         if df is not None and not df.empty}
        analysis['valid_datasets'] = len(valid_datasets)
        
        if len(valid_datasets) < 2:
            analysis['recommendations'].append("Need at least 2 non-empty datasets to merge")
            return analysis
        
        # Pairwise analysis with intelligent strategy selection
        dataset_names = list(valid_datasets.keys())
        mergeable_pairs = 0
        concatenate_pairs = 0
        join_pairs = 0
        
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                df1, df2 = valid_datasets[name1], valid_datasets[name2]
                
                common_cols = find_common_columns_between_two(df1, df2)
                strategy = determine_merge_strategy(df1, df2)
                
                pair_key = f"{name1} + {name2}"
                analysis['strategy_analysis'][pair_key] = {
                    'common_columns': common_cols,
                    'common_count': len(common_cols),
                    'recommended_strategy': strategy,
                    'mergeable': strategy != 'no_merge_possible'
                }
                
                if strategy != 'no_merge_possible':
                    mergeable_pairs += 1
                    if strategy == 'concatenate':
                        concatenate_pairs += 1
                    elif strategy == 'join':
                        join_pairs += 1
        
        # Overall feasibility and recommendations
        if mergeable_pairs > 0:
            analysis['feasible'] = True
            analysis['recommended_strategy'] = 'auto'  # Let each pair decide intelligently
            
            strategy_summary = []
            if concatenate_pairs > 0:
                strategy_summary.append(f"{concatenate_pairs} pairs for concatenation")
            if join_pairs > 0:
                strategy_summary.append(f"{join_pairs} pairs for joining")
            
            analysis['recommendations'].append(f"Intelligent merging possible: {', '.join(strategy_summary)}")
            analysis['recommendations'].append("Using 'auto' strategy for data-driven merge decisions")
        else:
            analysis['recommendations'].append("No common columns found between any dataset pairs - datasets cannot be merged")
        
        return analysis
        
    except Exception as e:
        analysis['warnings'].append(f"Analysis failed: {str(e)}")
        return analysis

def should_merge_datasets(datasets_dict: Dict[str, pd.DataFrame]) -> bool:
    """Quick check if datasets should be merged using pairwise analysis."""
    if len(datasets_dict) < 2:
        return False
    
    valid_datasets = {name: df for name, df in datasets_dict.items() 
                     if df is not None and not df.empty}
    
    if len(valid_datasets) < 2:
        return False
    
    # Check if any pair of datasets can be merged
    dataset_names = list(valid_datasets.keys())
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            df1, df2 = valid_datasets[dataset_names[i]], valid_datasets[dataset_names[j]]
            if len(find_common_columns_between_two(df1, df2)) > 0:
                return True
    
    return False