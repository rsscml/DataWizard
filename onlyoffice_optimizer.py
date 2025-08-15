"""
OnlyOffice Dataset Optimizer
============================

Optimizes datasets for OnlyOffice display to prevent browser memory issues.
Creates display-friendly versions while preserving full datasets for analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OnlyOfficeOptimizer:
    """Optimizes datasets for OnlyOffice browser display"""
    
    # Configuration constants
    MAX_CELLS_FOR_ONLYOFFICE = 10_000_000   # Maximum cells to send to OnlyOffice
    MAX_ROWS_FOR_ONLYOFFICE = 200_000    # Maximum rows to send to OnlyOffice
    MAX_COLS_FOR_ONLYOFFICE = 150        # Maximum columns to send to OnlyOffice
    FLOAT_PRECISION_DIGITS = 4           # Precision for float rounding
    
    @staticmethod
    def calculate_dataset_complexity(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate complexity metrics for a dataset"""
        if df.empty:
            return {
                'total_cells': 0,
                'estimated_memory_mb': 0,
                'needs_optimization': False,
                'complexity_score': 0
            }
        
        rows, cols = df.shape
        total_cells = rows * cols
        
        # Estimate memory usage based on data types
        memory_estimate = 0
        float64_cols = len(df.select_dtypes(include=['float64']).columns)
        int64_cols = len(df.select_dtypes(include=['int64']).columns)
        object_cols = len(df.select_dtypes(include=['object']).columns)
        
        # Rough memory estimates (bytes per cell)
        memory_estimate += float64_cols * rows * 8  # 8 bytes per float64
        memory_estimate += int64_cols * rows * 8    # 8 bytes per int64
        memory_estimate += object_cols * rows * 32  # ~32 bytes per string (average)
        
        estimated_memory_mb = memory_estimate / (1024 * 1024)
        
        # Calculate complexity score
        complexity_score = (
            (total_cells / 1_000_000) * 0.4 +          # 40% cell count factor
            (float64_cols / cols) * 0.3 +               # 30% float64 ratio factor
            (estimated_memory_mb / 100) * 0.3           # 30% memory factor
        )
        
        needs_optimization = (
            total_cells > OnlyOfficeOptimizer.MAX_CELLS_FOR_ONLYOFFICE or
            rows > OnlyOfficeOptimizer.MAX_ROWS_FOR_ONLYOFFICE or
            cols > OnlyOfficeOptimizer.MAX_COLS_FOR_ONLYOFFICE or
            complexity_score > 1.0
        )
        
        return {
            'total_cells': total_cells,
            'rows': rows,
            'cols': cols,
            'float64_cols': float64_cols,
            'int64_cols': int64_cols,
            'object_cols': object_cols,
            'estimated_memory_mb': estimated_memory_mb,
            'complexity_score': complexity_score,
            'needs_optimization': needs_optimization
        }
    
    @staticmethod
    def optimize_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Optimize data types to reduce memory footprint"""
        df_optimized = df.copy()
        optimizations = []
        
        # Convert float64 to float32 where possible
        float64_cols = df_optimized.select_dtypes(include=['float64']).columns
        for col in float64_cols:
            try:
                # Check if values fit in float32 range
                col_values = df_optimized[col].dropna()
                if not col_values.empty:
                    min_val, max_val = col_values.min(), col_values.max()
                    if (min_val >= np.finfo(np.float32).min and 
                        max_val <= np.finfo(np.float32).max):
                        df_optimized[col] = df_optimized[col].astype('float32')
                        optimizations.append(f"'{col}': float64 → float32")
            except Exception as e:
                logger.warning(f"Could not optimize column {col}: {e}")
        
        # Convert int64 to int32 where possible
        int64_cols = df_optimized.select_dtypes(include=['int64']).columns
        for col in int64_cols:
            try:
                col_values = df_optimized[col].dropna()
                if not col_values.empty:
                    min_val, max_val = col_values.min(), col_values.max()
                    if (min_val >= np.iinfo(np.int32).min and 
                        max_val <= np.iinfo(np.int32).max):
                        df_optimized[col] = df_optimized[col].astype('int32')
                        optimizations.append(f"'{col}': int64 → int32")
            except Exception as e:
                logger.warning(f"Could not optimize column {col}: {e}")
        
        return df_optimized, optimizations
    
    @staticmethod
    def reduce_float_precision(df: pd.DataFrame, precision: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """Reduce float precision to save memory and improve rendering"""
        if precision is None:
            precision = OnlyOfficeOptimizer.FLOAT_PRECISION_DIGITS
            
        df_reduced = df.copy()
        reductions = []
        
        float_cols = df_reduced.select_dtypes(include=['float32', 'float64']).columns
        for col in float_cols:
            try:
                # Round to specified precision
                df_reduced[col] = df_reduced[col].round(precision)
                reductions.append(f"'{col}': rounded to {precision} decimal places")
            except Exception as e:
                logger.warning(f"Could not reduce precision for column {col}: {e}")
        
        return df_reduced, reductions
    
    @staticmethod
    def sample_rows_intelligently(df: pd.DataFrame, max_rows: int) -> Tuple[pd.DataFrame, str]:
        """Intelligently sample rows to stay within limits"""
        if len(df) <= max_rows:
            return df, "No row sampling needed"
        
        # Strategy: Take first N/2 rows + last N/2 rows + some random middle rows
        head_rows = max_rows // 3
        tail_rows = max_rows // 3  
        middle_rows = max_rows - head_rows - tail_rows
        
        # Get head and tail
        head_data = df.head(head_rows)
        tail_data = df.tail(tail_rows)
        
        # Get random middle sample
        middle_start = head_rows
        middle_end = len(df) - tail_rows
        if middle_end > middle_start:
            middle_indices = np.random.choice(
                range(middle_start, middle_end), 
                size=min(middle_rows, middle_end - middle_start), 
                replace=False
            )
            middle_data = df.iloc[sorted(middle_indices)]
        else:
            middle_data = pd.DataFrame()
        
        # Combine samples
        sampled_df = pd.concat([head_data, middle_data, tail_data], ignore_index=True)
        
        sampling_note = f"Sampled {len(sampled_df)} rows from {len(df)} (head: {len(head_data)}, middle: {len(middle_data)}, tail: {len(tail_data)})"
        
        return sampled_df, sampling_note
    
    @staticmethod
    def select_important_columns(df: pd.DataFrame, max_cols: int) -> Tuple[pd.DataFrame, str]:
        """Select most important columns if there are too many"""
        if len(df.columns) <= max_cols:
            return df, "No column selection needed"
        
        # Strategy: Prioritize columns with more data and variety
        column_scores = {}
        
        for col in df.columns:
            score = 0
            col_data = df[col]
            
            # Factor 1: Data completeness (50% weight)
            completeness = col_data.count() / len(df)
            score += completeness * 0.5
            
            # Factor 2: Data variety (30% weight)
            if col_data.dtype in ['object', 'string']:
                unique_ratio = col_data.nunique() / col_data.count() if col_data.count() > 0 else 0
                score += min(unique_ratio * 2, 1.0) * 0.3
            elif col_data.dtype in ['float32', 'float64', 'int32', 'int64']:
                # For numeric: reward columns with reasonable variance
                try:
                    if col_data.std() > 0:
                        score += 0.3
                except:
                    pass
            
            # Factor 3: Column name importance (20% weight)
            col_lower = col.lower()
            important_keywords = ['id', 'name', 'date', 'time', 'amount', 'value', 'price', 'total', 'count']
            if any(keyword in col_lower for keyword in important_keywords):
                score += 0.2
            
            column_scores[col] = score
        
        # Select top columns
        top_columns = sorted(column_scores.keys(), key=lambda x: column_scores[x], reverse=True)[:max_cols]
        
        selected_df = df[top_columns]
        selection_note = f"Selected {len(top_columns)} most important columns from {len(df.columns)}"
        
        return selected_df, selection_note
    
    @classmethod
    def optimize_for_onlyoffice(cls, df: pd.DataFrame, worksheet_name: str = "Sheet") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main optimization function - creates OnlyOffice-friendly version of dataset
        
        Returns:
            (optimized_df, optimization_report)
        """
        if df.empty:
            return df, {'optimizations_applied': [], 'original_complexity': {}, 'final_complexity': {}}
        
        optimization_report = {
            'worksheet_name': worksheet_name,
            'optimizations_applied': [],
            'original_shape': df.shape,
            'warnings': []
        }
        
        # Calculate original complexity
        original_complexity = cls.calculate_dataset_complexity(df)
        optimization_report['original_complexity'] = original_complexity
        
        logger.info(f"Optimizing worksheet '{worksheet_name}' for OnlyOffice")
        logger.info(f"Original: {df.shape[0]:,} rows × {df.shape[1]} cols = {original_complexity['total_cells']:,} cells")
        logger.info(f"Complexity score: {original_complexity['complexity_score']:.2f}, Estimated memory: {original_complexity['estimated_memory_mb']:.1f} MB")
        
        if not original_complexity['needs_optimization']:
            logger.info("Dataset is already OnlyOffice-friendly, no optimization needed")
            optimization_report['final_complexity'] = original_complexity
            optimization_report['final_shape'] = df.shape
            return df, optimization_report
        
        # Start optimization process
        optimized_df = df.copy()
        
        # Step 1: Optimize data types
        optimized_df, type_optimizations = cls.optimize_data_types(optimized_df)
        if type_optimizations:
            optimization_report['optimizations_applied'].extend([f"Data type optimization: {opt}" for opt in type_optimizations])
        
        # Step 2: Reduce float precision
        optimized_df, precision_reductions = cls.reduce_float_precision(optimized_df)
        if precision_reductions:
            optimization_report['optimizations_applied'].extend([f"Precision reduction: {red}" for red in precision_reductions])
        
        # Step 3: Column selection if needed
        if optimized_df.shape[1] > cls.MAX_COLS_FOR_ONLYOFFICE:
            optimized_df, col_selection_note = cls.select_important_columns(optimized_df, cls.MAX_COLS_FOR_ONLYOFFICE)
            optimization_report['optimizations_applied'].append(f"Column selection: {col_selection_note}")
            optimization_report['warnings'].append(f"Only showing {len(optimized_df.columns)} most important columns out of {df.shape[1]} total")
        
        # Step 4: Row sampling if needed
        if optimized_df.shape[0] > cls.MAX_ROWS_FOR_ONLYOFFICE:
            optimized_df, row_sampling_note = cls.sample_rows_intelligently(optimized_df, cls.MAX_ROWS_FOR_ONLYOFFICE)
            optimization_report['optimizations_applied'].append(f"Row sampling: {row_sampling_note}")
            optimization_report['warnings'].append(f"Only showing {len(optimized_df)} sampled rows out of {df.shape[0]} total")
        
        # Calculate final complexity
        final_complexity = cls.calculate_dataset_complexity(optimized_df)
        optimization_report['final_complexity'] = final_complexity
        optimization_report['final_shape'] = optimized_df.shape
        
        # Log results
        memory_reduction = original_complexity['estimated_memory_mb'] - final_complexity['estimated_memory_mb']
        logger.info(f"Optimization complete for '{worksheet_name}':")
        logger.info(f"  {df.shape[0]:,} → {optimized_df.shape[0]:,} rows ({optimized_df.shape[0]/df.shape[0]*100:.1f}%)")
        logger.info(f"  {df.shape[1]} → {optimized_df.shape[1]} columns ({optimized_df.shape[1]/df.shape[1]*100:.1f}%)")
        logger.info(f"  {original_complexity['total_cells']:,} → {final_complexity['total_cells']:,} cells ({final_complexity['total_cells']/original_complexity['total_cells']*100:.1f}%)")
        logger.info(f"  Memory: {original_complexity['estimated_memory_mb']:.1f} → {final_complexity['estimated_memory_mb']:.1f} MB (saved {memory_reduction:.1f} MB)")
        
        return optimized_df, optimization_report

def optimize_worksheet_data_for_onlyoffice(worksheet_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Optimize all worksheets in a workbook for OnlyOffice display
    
    Args:
        worksheet_data: Dictionary of worksheet_name -> DataFrame
        
    Returns:
        (optimized_worksheet_data, overall_optimization_report)
    """
    optimized_worksheets = {}
    overall_report = {
        'total_worksheets': len(worksheet_data),
        'worksheets_optimized': 0,
        'worksheets_unchanged': 0,
        'total_memory_saved_mb': 0,
        'optimization_details': {},
        'summary_message': ""
    }
    
    logger.info(f"Starting OnlyOffice optimization for {len(worksheet_data)} worksheet(s)")
    
    for worksheet_name, df in worksheet_data.items():
        optimized_df, optimization_report = OnlyOfficeOptimizer.optimize_for_onlyoffice(df, worksheet_name)
        optimized_worksheets[worksheet_name] = optimized_df
        overall_report['optimization_details'][worksheet_name] = optimization_report
        
        if optimization_report['optimizations_applied']:
            overall_report['worksheets_optimized'] += 1
            memory_saved = (optimization_report['original_complexity']['estimated_memory_mb'] - 
                          optimization_report['final_complexity']['estimated_memory_mb'])
            overall_report['total_memory_saved_mb'] += memory_saved
        else:
            overall_report['worksheets_unchanged'] += 1
    
    # Create summary message
    if overall_report['worksheets_optimized'] > 0:
        overall_report['summary_message'] = (
            f"Optimized {overall_report['worksheets_optimized']} worksheet(s) for OnlyOffice display. "
            f"Estimated memory savings: {overall_report['total_memory_saved_mb']:.1f} MB. "
            f"Your full dataset remains available for analysis."
        )
    else:
        overall_report['summary_message'] = "All worksheets are already OnlyOffice-friendly, no optimization needed."
    
    logger.info(f"OnlyOffice optimization complete: {overall_report['summary_message']}")
    
    return optimized_worksheets, overall_report
