"""
Flask Excel Analytics Web Application with OnlyOffice Document Server Integration
================================================================================

Updated version with session-aware user separation for multi-user support.

Features:
- Unified file upload for Excel/CSV files (single or multiple)
- Multiple CSV files can be uploaded and combined into a single Excel workbook
- OnlyOffice Document Server integration for professional spreadsheet editing
- Multi-worksheet Excel file support
- AI agent for data analysis with interactive charts
- Pandas-based data analysis with intelligent worksheet merging
- Download functionality for analyzed spreadsheets
- Real-time collaborative editing via OnlyOffice
- Enhanced output panel for better result display
- Persistent output history with copy functionality
- Session-wide context maintenance for follow-up queries
- SESSION-AWARE: Each user gets their own isolated data and history
"""

from shared_state import shared_state
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime, timedelta
import traceback
import tempfile
import shutil
import atexit
import signal
import logging
import threading
import time

# entity resolution
from text_standardizer import (
    ConservativeTextStandardizer,
    standardize_data,
    DatasetStandardizationResult
)

# Multi dataset merge functionality
from data_merger import (
    merge_datasets_on_common_columns,
    create_merged_excel_file,
    analyze_merge_feasibility,
    should_merge_datasets,
    remove_duplicates_from_dataset
)

# Import session management
from session_manager import (
    session_manager, setup_session_cleanup,
    get_current_data, set_current_data,
    get_current_filename, set_current_filename,
    get_data_summary, set_data_summary,
    get_worksheet_data, set_worksheet_data,
    get_worksheet_summaries, set_worksheet_summaries,
    get_analyzable_worksheets, set_analyzable_worksheets,
    get_excluded_worksheets, set_excluded_worksheets,
    get_active_worksheet, set_active_worksheet,
    get_current_document_key, set_current_document_key,
    get_current_document_url, set_current_document_url,
    get_session_history, set_session_history, add_to_session_history, clear_session_history,
    reset_current_session
)

# for enhanced follow-ups
from session_context_enhancer import (
    enhance_session_context,
    get_follow_up_suggestions,
    is_follow_up_query,
    create_result_summary
)

# Import modularized components
from agent import DataAnalysisAgent
from onlyoffice_optimizer import OnlyOfficeOptimizer, optimize_worksheet_data_for_onlyoffice
from excel_handlers import filter_and_read_clean_worksheets, create_worksheet_report, create_worksheet_report_with_exclusions
from enhanced_data_profiler import format_enhanced_context_for_llm

# Import database specific components - new in v10
from database_routes import register_database_routes
# ENHANCED: Import the new supplemental context routes
from supplemental_context_routes import register_supplemental_context_routes
from database_config import get_config_manager
from template_creation_routes import register_template_management_routes
from template_manager import template_manager, TemplateDefinition
from enhanced_analysis import (
    analyze_data_with_template_support,
    create_enhanced_context_for_query,
    get_template_suggestions_for_data,
    apply_template_manually,
    validate_data_against_template,
    analyze_data_standard
)

# file cleanup
from file_cleanup_manager import setup_file_cleanup

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
    
# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['DOCUMENTS_FOLDER'] = 'documents'  # For OnlyOffice documents
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.secret_key = os.getenv('SECRET_KEY', 'mysecretkey')

# Session configuration for user separation
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# OnlyOffice Configuration
ONLYOFFICE_SERVER_URL = os.getenv('ONLYOFFICE_SERVER_URL', 'https://ai-ods-demo.azurewebsites.net')
ONLYOFFICE_CALLBACK_URL = os.getenv('ONLYOFFICE_CALLBACK_URL', 'http://localhost:5000/onlyoffice/callback')

# Ensure directories exist
for folder in ['UPLOAD_FOLDER', 'DOWNLOAD_FOLDER', 'DOCUMENTS_FOLDER']:
    os.makedirs(app.config[folder], exist_ok=True)

# ENHANCED: Create db_context directory for supplemental context storage
os.makedirs('db_context', exist_ok=True)
    
# Configure Flask to avoid restart issues
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Setup session cleanup for automatic maintenance
setup_session_cleanup(app)

# Text standardization configuration
# Text standardization configuration (Conservative mode)
TEXT_STANDARDIZATION_ENABLED = os.getenv('TEXT_STANDARDIZATION_ENABLED', 'True').lower() == 'true'
TEXT_STANDARDIZATION_THRESHOLD = int(os.getenv('TEXT_STANDARDIZATION_THRESHOLD', '95'))

# Initialize conservative text standardizer
text_standardizer = ConservativeTextStandardizer(
    similarity_threshold=TEXT_STANDARDIZATION_THRESHOLD,
    min_column_size=5,
    max_unique_ratio=0.5
)

# Initialize the agent (remains global as it's shared infrastructure)
try:
    agent = DataAnalysisAgent()
    logger.info("DataAnalysisAgent initialized successfully with auto-refresh tokens")
except Exception as e:
    logger.error(f"Failed to initialize DataAnalysisAgent: {e}")
    raise

# Initialize file cleanup system
try:
    file_cleanup_manager = setup_file_cleanup(app)
    logger.info("File cleanup system initialized")
except Exception as e:
    logger.error(f"Failed to initialize file cleanup system: {e}")
    # Don't raise - app can work without cleanup but log warning

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_csv_file(filename):
    """Check if file is a CSV file"""
    return filename.lower().endswith('.csv')

def generate_document_key():
    """Generate a unique document key for OnlyOffice"""
    return str(uuid.uuid4())


def combine_csv_files_to_excel_with_merge(csv_files):
    """
    Enhanced: Combine multiple CSV files, with intelligent merging if common columns exist
    """
    print(f"Processing {len(csv_files)} CSV files with intelligent merging...")

    # First, read all CSV files into a dictionary
    csv_datasets = {}

    for csv_file_info in csv_files:
        file_path = csv_file_info['path']
        original_filename = csv_file_info['filename']

        # Generate dataset name from filename (remove .csv extension)
        dataset_name = os.path.splitext(original_filename)[0]

        print(f"Reading CSV: {original_filename}")

        try:
            # Read CSV with different encoding attempts
            csv_data = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    csv_data = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if csv_data is None:
                print(f"Warning: Could not read {original_filename} with any encoding, skipping...")
                continue

            # Clean the data for better JSON serialization (existing code)
            for col in csv_data.columns:
                dtype_name = str(csv_data[col].dtype)
                if any(problematic in dtype_name for problematic in
                       ['Int64', 'Float64', 'boolean', 'string']):
                    print(f"Converting column {col} from {dtype_name} to standard type in {dataset_name}")
                    if 'Int' in dtype_name:
                        csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce').astype('float64')
                    elif 'Float' in dtype_name:
                        csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce').astype('float64')
                    elif 'boolean' in dtype_name:
                        csv_data[col] = csv_data[col].astype('object')
                    else:  # string type
                        csv_data[col] = csv_data[col].astype('object')

            csv_datasets[dataset_name] = csv_data
            print(f"CSV {original_filename} loaded successfully with shape {csv_data.shape}")

        except Exception as e:
            print(f"Error processing CSV {original_filename}: {e}")
            continue

    if not csv_datasets:
        raise Exception("No CSV files could be successfully processed")

    # Apply text standardization to CSV datasets BEFORE merging
    if TEXT_STANDARDIZATION_ENABLED and csv_datasets:
        print("\n=== APPLYING TEXT STANDARDIZATION TO CSV FILES ===")
        standardized_csv_datasets, standardization_results = text_standardizer.standardize_worksheet_dict(
            csv_datasets
        )

        # Use standardized datasets for merging
        csv_datasets = standardized_csv_datasets

        # Generate summary
        standardization_summary = text_standardizer.get_standardization_summary(standardization_results)

        # Log standardization summary
        if standardization_summary and standardization_summary['total_variants_resolved'] > 0:
            print(f"Standardized {standardization_summary['total_variants_resolved']} text variants across CSV files")

    # NEW: Analyze merge feasibility
    merge_analysis = analyze_merge_feasibility(csv_datasets)
    print(f"Merge analysis: {merge_analysis['feasible']} (common columns: {merge_analysis['strategy_analysis']})")

    if merge_analysis['feasible'] and should_merge_datasets(csv_datasets):
        print("MERGING STRATEGY: Datasets have common columns - performing intelligent merge")

        # Perform the merge
        merge_result = merge_datasets_on_common_columns(
            csv_datasets,
            merge_strategy='auto',
            remove_duplicates_first=True)

        if merge_result.success:
            # Create temporary Excel file with merged data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_excel_filename = f"merged_csvs_{len(csv_files)}files_{timestamp}.xlsx"
            temp_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_excel_filename)

            # Save merged data to Excel
            success, message = create_merged_excel_file(
                merge_result.merged_df,
                temp_excel_path,
                {name: {'shape': df.shape} for name, df in csv_datasets.items()},
                merge_result
            )

            if success:
                # Return merged result as single worksheet
                merged_worksheet_data = {'Merged_Data': merge_result.merged_df}
                print(f"Successfully merged {len(csv_files)} CSV files into single dataset")
                print(f"Merge summary: {merge_result.merge_summary}")
                print(f"Final shape: {merge_result.merged_df.shape}")

                return temp_excel_path, temp_excel_filename, merged_worksheet_data
            else:
                print(f"Failed to create merged Excel file: {message}")
                # Fallback to original method
        else:
            print(f"Merge failed: {'; '.join(merge_result.errors)}")
            print("Falling back to separate worksheets approach")

    # FALLBACK: Use original method (separate worksheets)
    print("FALLBACK STRATEGY: Creating separate worksheets (no merge)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_excel_filename = f"combined_csvs_{len(csv_files)}_files_{timestamp}.xlsx"
    temp_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_excel_filename)

    combined_worksheet_data = {}

    with pd.ExcelWriter(temp_excel_path, engine='openpyxl') as writer:
        for dataset_name, csv_data in csv_datasets.items():
            # Ensure worksheet name is valid for Excel
            worksheet_name = dataset_name[:31]
            worksheet_name = ''.join(c for c in worksheet_name if c.isalnum() or c in (' ', '_', '-'))
            if not worksheet_name:
                worksheet_name = f"Sheet{len(combined_worksheet_data) + 1}"

            # Make sure worksheet name is unique
            original_name = worksheet_name
            counter = 1
            while worksheet_name in combined_worksheet_data:
                worksheet_name = f"{original_name}_{counter}"
                counter += 1

            # Save to Excel workbook
            csv_data.to_excel(writer, sheet_name=worksheet_name, index=False)
            combined_worksheet_data[worksheet_name] = csv_data

    return temp_excel_path, temp_excel_filename, combined_worksheet_data


def process_multi_worksheet_excel_with_merge(clean_worksheets, original_filename, base_path):
    """
    Enhanced: Process Excel with multiple worksheets, with intelligent merging if common columns exist
    """
    print(f"Processing Excel file with {len(clean_worksheets)} worksheets - checking merge feasibility...")

    # Analyze merge feasibility
    merge_analysis = analyze_merge_feasibility(clean_worksheets)
    print(f"Merge analysis: {merge_analysis['feasible']} (common columns: {merge_analysis['strategy_analysis']})")

    if merge_analysis['feasible'] and should_merge_datasets(clean_worksheets):
        print("MERGING STRATEGY: Worksheets have common columns - performing intelligent merge")

        # Perform the merge
        merge_result = merge_datasets_on_common_columns(
            clean_worksheets,
            merge_strategy='auto',
            remove_duplicates_first=True)

        if merge_result.success:
            # Create new Excel file with merged data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(original_filename)[0]
            merged_filename = f"{base_name}_merged_{timestamp}.xlsx"
            merged_file_path = os.path.join(base_path, merged_filename)

            # Save merged data to Excel
            success, message = create_merged_excel_file(
                merge_result.merged_df,
                merged_file_path,
                {name: {'shape': df.shape} for name, df in clean_worksheets.items()},
                merge_result
            )

            if success:
                # Return merged result as single worksheet
                merged_worksheet_data = {'Merged_Data': merge_result.merged_df}
                print(f"Successfully merged {len(clean_worksheets)} worksheets into single dataset")
                print(f"Merge summary: {merge_result.merge_summary}")
                print(f"Final shape: {merge_result.merged_df.shape}")

                return True, merged_file_path, merged_filename, merged_worksheet_data, merge_result
            else:
                print(f"Failed to create merged Excel file: {message}")
        else:
            print(f"Merge failed: {'; '.join(merge_result.errors)}")
            print("Falling back to original multi-worksheet approach")

    # No merge performed or merge failed
    return False, None, None, clean_worksheets, None

def create_onlyoffice_document(file_path, original_filename):
    """Create an OnlyOffice-compatible document and return the document URL and key"""
    
    try:
        # Generate unique document key
        document_key = generate_document_key()
        
        # Create documents directory if it doesn't exist
        documents_dir = app.config['DOCUMENTS_FOLDER']
        os.makedirs(documents_dir, exist_ok=True, mode=0o755)
        
        # Determine file extension - ensure it's valid for OnlyOffice
        file_ext = 'xlsx'  # default to xlsx
        if original_filename and '.' in original_filename:
            extracted_ext = original_filename.split('.')[-1].lower()
            # Validate against supported OnlyOffice spreadsheet formats
            valid_extensions = ['xlsx', 'xls', 'csv', 'ods', 'xlsm', 'xlsb']
            if extracted_ext in valid_extensions:
                file_ext = extracted_ext
            else:
                print(f"Warning: Unsupported file extension '{extracted_ext}', defaulting to 'xlsx'")
        
        print(f"Creating OnlyOffice document with extension: {file_ext}")
        
        # Create the document filename with the document key
        document_filename = f"{document_key}.{file_ext}"
        document_path = os.path.join(documents_dir, document_filename)
        
        if file_ext == 'csv':
            # Convert CSV to Excel for better OnlyOffice compatibility
            df = pd.read_csv(file_path)
            document_filename = f"{document_key}.xlsx"
            document_path = os.path.join(documents_dir, document_filename)
            df.to_excel(document_path, index=False, sheet_name='Sheet1')
        else:
            # Copy Excel file to documents folder
            shutil.copy2(file_path, document_path)

        # Associate document with session
        session_id = session_manager.get_current_session_id()
        file_cleanup_manager.associate_file_with_session(document_path, session_id, 'document')

        # Set proper file permissions
        try:
            os.chmod(document_path, 0o644)
        except Exception as perm_error:
            print(f"Warning: Could not set file permissions: {perm_error}")
        
        # Create document URL (served by Flask)
        document_url = f"{request.host_url}documents/{document_filename}"
        
        # Store globally for current session
        set_current_document_key(document_key)
        set_current_document_url(document_url)
        
        print(f"OnlyOffice document created: {document_path}")
        print(f"Document URL: {document_url}")
        print(f"Document Key: {document_key}")
        print(f"Document filename: {document_filename}")
        
        # Test if OnlyOffice server is accessible
        try:
            import requests
            health_check = requests.get(f"{ONLYOFFICE_SERVER_URL}/healthcheck", timeout=5)
            if health_check.status_code != 200:
                print(f"Warning: OnlyOffice server health check failed: {health_check.status_code}")
        except Exception as health_error:
            print(f"Warning: Could not reach OnlyOffice server: {health_error}")
        
        return document_url, document_key
        
    except Exception as e:
        print(f"Error creating OnlyOffice document: {e}")
        raise e

def create_cleaned_excel_for_onlyoffice(clean_worksheets, original_filename, base_path):
    """
    Create a cleaned AND OPTIMIZED Excel file for OnlyOffice display.

    NEW: Now includes intelligent optimization to prevent browser memory issues.
    Creates display-optimized version while preserving full dataset for analysis.

    Args:
        clean_worksheets: Dictionary of cleaned DataFrames
        original_filename: Original file name for reference
        base_path: Base path for saving the cleaned file

    Returns:
        (cleaned_file_path, cleaned_filename, optimization_report): Paths to the cleaned Excel file and optimization details
    """
    try:
        # Generate cleaned filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(original_filename)[0]
        cleaned_filename = f"{base_name}_onlyoffice_optimized_{timestamp}.xlsx"
        cleaned_file_path = os.path.join(base_path, cleaned_filename)

        print(f"  Creating optimized Excel file for OnlyOffice: {cleaned_filename}")

        # STEP 1: Optimize worksheets for OnlyOffice display
        optimized_worksheets, optimization_report = optimize_worksheet_data_for_onlyoffice(clean_worksheets)

        # Log optimization results
        if optimization_report['worksheets_optimized'] > 0:
            print(f"  OnlyOffice optimization applied to {optimization_report['worksheets_optimized']} worksheet(s)")
            print(f"  Estimated memory savings: {optimization_report['total_memory_saved_mb']:.1f} MB")

            # Log specific optimizations
            for ws_name, ws_report in optimization_report['optimization_details'].items():
                if ws_report['optimizations_applied']:
                    original_shape = ws_report['original_shape']
                    final_shape = ws_report['final_shape']
                    print(
                        f"    '{ws_name}': {original_shape[0]:,}×{original_shape[1]} → {final_shape[0]:,}×{final_shape[1]} cells")

                    # Show warnings to user about what was optimized
                    for warning in ws_report.get('warnings', []):
                        print(f"        {warning}")
        else:
            print(f"  No OnlyOffice optimization needed - datasets are already browser-friendly")

        # STEP 2: Create Excel file with optimized data
        with pd.ExcelWriter(cleaned_file_path, engine='openpyxl') as writer:
            for sheet_name, df in optimized_worksheets.items():
                # Ensure the data starts at A1 and is properly formatted
                clean_df = df.copy()

                # Reset index to ensure clean start
                clean_df = clean_df.reset_index(drop=True)

                # Ensure column names are clean strings
                clean_df.columns = [str(col).strip() for col in clean_df.columns]

                # Remove any completely empty rows/columns that might have been missed
                clean_df = clean_df.dropna(how='all').dropna(axis=1, how='all')

                # Write to Excel starting at A1
                clean_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)

                print(f"    Sheet '{sheet_name}': {clean_df.shape[0]} rows × {clean_df.shape[1]} columns")

        print(f"  Optimized Excel file created successfully: {cleaned_file_path}")
        return cleaned_file_path, cleaned_filename, optimization_report

    except Exception as e:
        print(f"  Error creating optimized Excel file: {e}")
        raise e


def apply_text_standardization(worksheets_dict, file_type="Excel"):
    """
    Apply text standardization to worksheet data

    Args:
        worksheets_dict: Dictionary of worksheet_name -> DataFrame
        file_type: Type of file being processed (for logging)

    Returns:
        (standardized_worksheets, standardization_results, summary)
    """
    if not TEXT_STANDARDIZATION_ENABLED:
        logger.info("Text standardization is disabled")
        return worksheets_dict, {}, None

    try:
        logger.info(f"Applying text standardization to {len(worksheets_dict)} worksheet(s) from {file_type} file")

        # Standardize the worksheets
        standardized_worksheets, results = text_standardizer.standardize_worksheet_dict(
            worksheets_dict,
            method=TEXT_STANDARDIZATION_METHOD
        )

        # Generate summary
        summary = text_standardizer.get_standardization_summary(results)

        # Log results
        if summary['total_variants_resolved'] > 0:
            logger.info(
                f"Text standardization complete: {summary['total_variants_resolved']} variants resolved across {summary['total_columns_standardized']} columns")

            # Print detailed report
            report = text_standardizer.generate_standardization_report(results)
            print(report)
        else:
            logger.info("No text variants found that needed standardization")

        return standardized_worksheets, results, summary

    except Exception as e:
        logger.error(f"Error during text standardization: {e}")
        import traceback
        traceback.print_exc()
        # Return original data if standardization fails
        return worksheets_dict, {}, None


def cleanup_old_cleaned_files(directory, max_age_hours=24):
    """
    Clean up old cleaned Excel files to save disk space.

    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep (default 24 hours)
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for filename in os.listdir(directory):
            if filename.endswith('_cleaned_') and filename.endswith('.xlsx'):
                file_path = os.path.join(directory, filename)
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > max_age_seconds:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")

    except Exception as e:
        print(f"Warning: Could not clean up old files: {e}")

"""
def convert_pandas_types(obj):
    # Convert pandas types to JSON serializable types
    # Handle strings first - they should NOT be converted to character lists
    if isinstance(obj, str):
        return obj

    # Handle DataFrames and Series first (before pd.isna check)
    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame to records format
        try:
            return obj.to_dict('records')
        except:
            return str(obj)
    elif isinstance(obj, pd.Series):
        # Convert Series to list
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
        # pd.isna() failed, obj is likely not a scalar
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
        # Handle nullable integer types
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

    # Handle pandas dtypes directly
    elif hasattr(obj, '__class__') and 'DType' in str(type(obj)):
        return str(obj)

    # Handle remaining dictionaries and lists
    elif isinstance(obj, dict):
        return {k: convert_pandas_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_pandas_types(v) for v in obj]

    # Default case
    else:
        return obj
"""

def _replace_non_finite_in_df(df: pd.DataFrame) -> pd.DataFrame:
    # Convert non-finite floats and NaNs to None (which becomes null in JSON)
    return df.replace({np.nan: None, np.inf: None, -np.inf: None})

def _sanitize_float(x):
    try:
        # Handles both Python float and np.floating
        if isinstance(x, (float, np.floating)):
            return x if np.isfinite(x) else None
        return x
    except Exception:
        return x

def convert_pandas_types(obj):
    """Convert pandas/NumPy/odd types to JSON-serializable, RFC 8259-valid values."""
    # Strings unchanged
    if isinstance(obj, str):
        return obj

    # ---- pandas containers ----
    if isinstance(obj, pd.DataFrame):
        try:
            df = _replace_non_finite_in_df(obj)
            # Convert to records then RECURSIVELY sanitize
            records = df.to_dict(orient='records')
            return [convert_pandas_types(rec) for rec in records]
        except Exception:
            return str(obj)

    if isinstance(obj, pd.Series):
        try:
            ser = obj.replace({np.nan: None, np.inf: None, -np.inf: None})
            return [convert_pandas_types(v) for v in ser.tolist()]
        except Exception:
            return str(obj)

    if isinstance(obj, (pd.Index, pd.MultiIndex)):
        try:
            vals = list(obj)
            return [convert_pandas_types(v) for v in vals]
        except Exception:
            return str(obj)

    # pd.NA / NaN / None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # ---- numpy scalars/arrays ----
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _sanitize_float(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [convert_pandas_types(v) for v in obj.tolist()]

    # ---- datetime / timedelta ----
    from datetime import date, datetime, time as dtime, timedelta
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        # ISO 8601 for interop
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, np.timedelta64, timedelta)):
        return str(obj)

    # ---- other common oddballs ----
    try:
        from decimal import Decimal
        from uuid import UUID
        from pathlib import Path
    except Exception:
        Decimal = UUID = Path = None

    if Decimal is not None and isinstance(obj, Decimal):
        # pick string to avoid precision surprises
        return str(obj)
    if UUID is not None and isinstance(obj, UUID):
        return str(obj)
    if Path is not None and isinstance(obj, Path):
        return str(obj)

    # ---- mappings and sequences ----
    if isinstance(obj, dict):
        return {str(k): convert_pandas_types(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [convert_pandas_types(v) for v in obj]

    # Objects with .tolist()
    if hasattr(obj, 'tolist'):
        try:
            return [convert_pandas_types(v) for v in obj.tolist()]
        except Exception:
            pass

    # Final float safety
    if isinstance(obj, float):
        return _sanitize_float(obj)

    # bool/int fine as-is; anything else: string fallback
    if isinstance(obj, (bool, int)):
        return obj

    return str(obj)

def remove_plot_objects_from_result(data):
    """
    Recursively remove plot objects (Plotly figures, matplotlib figures, etc.)
    from the result dictionary to prevent JSON serialization errors.
    """
    import plotly.graph_objects as go
    import plotly.graph_objs as graph_objs

    def is_plot_object(obj):
        """Check if an object is a plotting object that should be excluded"""
        # Check for Plotly Figure objects
        if hasattr(go, 'Figure') and isinstance(obj, go.Figure):
            return True
        if hasattr(graph_objs, 'Figure') and isinstance(obj, graph_objs.Figure):
            return True

        # Check for matplotlib figures
        try:
            import matplotlib.figure
            if isinstance(obj, matplotlib.figure.Figure):
                return True
        except ImportError:
            pass

        # Check for other common plotting objects by class name
        obj_type_name = type(obj).__name__
        plot_object_names = [
            'Figure', 'Axes', 'Artist', 'Plot', 'Chart',
            'Visualization', 'Graph', 'PlotlyFigure'
        ]

        if any(plot_name in obj_type_name for plot_name in plot_object_names):
            return True

        # Check for objects with plotting-specific attributes
        plot_attributes = ['data', 'layout', 'to_html', 'show', 'to_json']
        if hasattr(obj, 'data') and hasattr(obj, 'layout') and hasattr(obj, 'show'):
            return True

        return False

    def clean_dict(d):
        """Recursively clean a dictionary"""
        if not isinstance(d, dict):
            return d

        cleaned = {}
        for key, value in d.items():
            if is_plot_object(value):
                print(f"Removed plot object from result: key='{key}', type={type(value).__name__}")
                continue  # Skip this key-value pair
            elif isinstance(value, dict):
                cleaned[key] = clean_dict(value)
            elif isinstance(value, list):
                cleaned[key] = clean_list(value)
            else:
                cleaned[key] = value
        return cleaned

    def clean_list(lst):
        """Recursively clean a list"""
        cleaned = []
        for item in lst:
            if is_plot_object(item):
                print(f"Removed plot object from result list: type={type(item).__name__}")
                continue  # Skip this item
            elif isinstance(item, dict):
                cleaned.append(clean_dict(item))
            elif isinstance(item, list):
                cleaned.append(clean_list(item))
            else:
                cleaned.append(item)
        return cleaned

    # Handle different data types
    if isinstance(data, dict):
        return clean_dict(data)
    elif isinstance(data, list):
        return clean_list(data)
    else:
        # For non-dict/list data, check if it's a plot object
        if is_plot_object(data):
            print(f"Removed plot object from result: type={type(data).__name__}")
            return None
        return data

# added in v4
def merge_worksheets(worksheet_dict):
    """Intelligently merge multiple worksheets with data type harmonization"""
    if not worksheet_dict or len(worksheet_dict) == 0:
        return pd.DataFrame()

    if len(worksheet_dict) == 1:
        return list(worksheet_dict.values())[0].copy()

    print(f"Merging {len(worksheet_dict)} worksheets with data type harmonization...")

    # Step 1: Analyze column data types across all worksheets
    def harmonize_column_types(worksheet_dict, common_columns):
        """Harmonize data types for common columns across worksheets"""
        harmonized_worksheets = {}
        type_conversion_log = []

        # Analyze each common column to determine best target type
        target_types = {}
        for col in common_columns:
            print(f"Analyzing column '{col}' across worksheets...")

            # Collect all non-null values from this column across all worksheets
            all_values = []
            dtypes_found = []

            for sheet_name, df in worksheet_dict.items():
                if col in df.columns:
                    col_data = df[col].dropna()
                    if not col_data.empty:
                        all_values.extend(col_data.tolist())
                        dtypes_found.append(str(df[col].dtype))

            if not all_values:
                target_types[col] = 'object'
                continue

            # Determine best target type
            unique_dtypes = list(set(dtypes_found))
            print(f"  Column '{col}' has types: {unique_dtypes}")

            target_type = determine_best_type(all_values, unique_dtypes, col)
            target_types[col] = target_type
            print(f"  Target type for '{col}': {target_type}")

        # Convert all worksheets to use harmonized types
        for sheet_name, df in worksheet_dict.items():
            df_harmonized = df.copy()

            for col in common_columns:
                if col in df_harmonized.columns:
                    original_dtype = str(df_harmonized[col].dtype)
                    target_dtype = target_types[col]

                    if original_dtype != target_dtype:
                        try:
                            df_harmonized[col] = convert_column_type(
                                df_harmonized[col], target_dtype, col, sheet_name
                            )
                            type_conversion_log.append(
                                f"Sheet '{sheet_name}', Column '{col}': {original_dtype} → {target_dtype}"
                            )
                        except Exception as conv_error:
                            print(f"  Warning: Could not convert {col} in {sheet_name}: {conv_error}")
                            # Fall back to string conversion
                            df_harmonized[col] = df_harmonized[col].astype('str')
                            type_conversion_log.append(
                                f"Sheet '{sheet_name}', Column '{col}': {original_dtype} → str (fallback)"
                            )

            harmonized_worksheets[sheet_name] = df_harmonized

        if type_conversion_log:
            print("Data type conversions performed:")
            for log_entry in type_conversion_log:
                print(f"  {log_entry}")

        return harmonized_worksheets

    def determine_best_type(all_values, dtypes_found, col_name):
        """Determine the best target data type for a column"""
        # If all current types are the same, keep it
        if len(set(dtypes_found)) == 1:
            return dtypes_found[0]

        # Sample some values to test conversion possibilities
        sample_values = all_values[:100] if len(all_values) > 100 else all_values

        # Check if all values can be converted to numeric
        numeric_compatible = True
        datetime_compatible = True

        try:
            # Test numeric conversion
            for val in sample_values:
                if val is not None and str(val).strip() != '':
                    pd.to_numeric(str(val))
        except (ValueError, TypeError):
            numeric_compatible = False

        try:
            # Test datetime conversion
            for val in sample_values:
                if val is not None and str(val).strip() != '':
                    pd.to_datetime(str(val))
        except (ValueError, TypeError):
            datetime_compatible = False

        # Decision logic
        if 'float64' in dtypes_found or 'float32' in dtypes_found:
            return 'float64'  # Prefer float if any worksheet has float
        elif numeric_compatible and ('int64' in dtypes_found or 'int32' in dtypes_found):
            return 'float64'  # Use float to handle potential decimals
        elif numeric_compatible:
            return 'float64'
        elif datetime_compatible and any('datetime' in dtype for dtype in dtypes_found):
            return 'datetime64[ns]'
        else:
            return 'object'  # Default to object/string for mixed types

    def convert_column_type(series, target_type, col_name, sheet_name):
        """Convert a pandas series to target data type with error handling"""
        if target_type == 'float64':
            # Handle numeric conversion
            return pd.to_numeric(series, errors='coerce')
        elif target_type == 'int64':
            # Convert to numeric first, then to int
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series.astype('Int64')  # Nullable integer
        elif target_type == 'datetime64[ns]':
            return pd.to_datetime(series, errors='coerce')
        elif target_type == 'bool':
            # Handle boolean conversion
            return series.astype('bool')
        else:
            # Default to object/string
            return series.astype('str')

    # Get all columns across worksheets
    all_columns = [set(df.columns) for df in worksheet_dict.values()]
    common_columns = set(all_columns[0])
    for cols in all_columns[1:]:
        common_columns = common_columns.intersection(cols)

    if not common_columns:
        print("No common columns found - creating summary")
        summary_data = []
        for sheet_name, df in worksheet_dict.items():
            summary_data.append({
                'worksheet': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': ', '.join(df.columns[:5]) + ('...' if len(df.columns) > 5 else '')
            })
        return pd.DataFrame(summary_data)

    print(f"Common columns found: {list(common_columns)}")

    # Step 2: Harmonize data types
    harmonized_worksheets = harmonize_column_types(worksheet_dict, common_columns)

    # Step 3: Determine merge strategy
    # Check if all worksheets have identical structure after harmonization
    harmonized_columns = [set(df.columns) for df in harmonized_worksheets.values()]
    if all(cols == harmonized_columns[0] for cols in harmonized_columns):
        print("All worksheets have identical columns after harmonization - concatenating...")
        dataframes = []
        for sheet_name, df in harmonized_worksheets.items():
            df_copy = df.copy()
            df_copy['_worksheet_source'] = sheet_name
            dataframes.append(df_copy)
        return pd.concat(dataframes, ignore_index=True)

    # Step 4: Try intelligent joining
    # Look for key columns
    key_candidates = []
    for col in common_columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number', 'ref']):
            key_candidates.append(col)

    if key_candidates and len(common_columns) > 1:
        print(f"Attempting merge using key column: {key_candidates[0]}")
        try:
            key_col = key_candidates[0]
            merged_data = None

            for i, (sheet_name, df) in enumerate(harmonized_worksheets.items()):
                if i == 0:
                    merged_data = df.copy()
                else:
                    # Perform outer join to preserve all data
                    merged_data = pd.merge(
                        merged_data,
                        df,
                        on=key_col,
                        how='outer',
                        suffixes=('', f'_{sheet_name}')
                    )

            if merged_data is not None and not merged_data.empty:
                print(f"Successfully merged {len(harmonized_worksheets)} worksheets on '{key_col}'")
                return merged_data

        except Exception as merge_error:
            print(f"Key-based merge failed even after type harmonization: {merge_error}")

    # Step 5: Fallback to common columns concatenation
    print(f"Using common columns concatenation with {len(common_columns)} columns")
    dataframes = []
    for sheet_name, df in harmonized_worksheets.items():
        df_subset = df[list(common_columns)].copy()
        df_subset['_worksheet_source'] = sheet_name
        dataframes.append(df_subset)

    return pd.concat(dataframes, ignore_index=True)

# added in v4
def analyze_worksheet_schema(worksheet_data):
    """Analyze and report data type mismatches across worksheets"""
    if len(worksheet_data) <= 1:
        return "Single worksheet - no schema conflicts possible"

    schema_report = []

    # Get all unique column names
    all_columns = set()
    for df in worksheet_data.values():
        all_columns.update(df.columns)

    # Analyze each column
    for col in sorted(all_columns):
        col_info = {'column': col, 'worksheets': []}

        for sheet_name, df in worksheet_data.items():
            if col in df.columns:
                dtype = str(df[col].dtype)
                sample_values = df[col].dropna().head(3).tolist()
                col_info['worksheets'].append({
                    'sheet': sheet_name,
                    'dtype': dtype,
                    'samples': [str(v) for v in sample_values]
                })

        # Check for type conflicts
        if len(col_info['worksheets']) > 1:
            dtypes = [w['dtype'] for w in col_info['worksheets']]
            if len(set(dtypes)) > 1:
                schema_report.append(col_info)

    if schema_report:
        print("\n  Data type conflicts detected:")
        for conflict in schema_report:
            print(f"\nColumn '{conflict['column']}':")
            for ws_info in conflict['worksheets']:
                print(f"  {ws_info['sheet']}: {ws_info['dtype']} (samples: {ws_info['samples']})")
    else:
        print(" No data type conflicts detected across worksheets")

    return schema_report

@app.route('/config')
def get_config():
    """Get client configuration"""
    return jsonify({
        'onlyoffice_server_url': ONLYOFFICE_SERVER_URL
    })

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/documents/<filename>')
def serve_document(filename):
    """Serve OnlyOffice documents"""
    try:
        return send_from_directory(app.config['DOCUMENTS_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving document {filename}: {e}")
        return jsonify({'error': 'Document not found'}), 404

@app.route('/onlyoffice/callback', methods=['POST'])
def onlyoffice_callback():
    """Handle OnlyOffice document server callbacks"""
    try:
        data = request.get_json()
        print(f"OnlyOffice callback received: {data}")
        
        # Handle different callback statuses
        status = data.get('status', 0)
        
        if status == 1:  # Document is being edited
            print("Document is being edited")
        elif status == 2:  # Document is ready for saving
            print("Document is ready for saving")
        elif status == 3:  # Document saving error
            print("Document saving error")
        elif status == 4:  # Document closed without changes
            print("Document closed without changes")
        elif status == 6:  # Document is being edited, but the current document state is saved
            print("Document is being edited and saved")
        elif status == 7:  # Error has occurred while force saving the document
            print("Error occurred while force saving")
        
        return jsonify({'error': 0})  # Success response
        
    except Exception as e:
        print(f"OnlyOffice callback error: {e}")
        return jsonify({'error': 1}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with unified support for Excel and CSV files"""

    # Reinitialize agent when switching back to file upload mode from database mode
    global agent
    try:
        # Test if the agent's LLM client is still functional
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'client'):
            # Check if the HTTP client is closed
            client = getattr(agent.llm, 'client', None)
            if client and hasattr(client, '_client'):
                if getattr(client._client, 'is_closed', False):
                    logger.info("Detected closed HTTP client, refreshing agent connection...")
                    agent.refresh_llm_connection()
                    logger.info("Agent connection refreshed successfully")
            # Additional check for the http_client from agent.py
            from agent import http_client
            if http_client and hasattr(http_client, '_client'):
                if getattr(http_client._client, 'is_closed', False):
                    logger.info("Detected closed global HTTP client, refreshing agent...")
                    agent.refresh_llm_connection()
                    logger.info("Agent refreshed due to closed global HTTP client")
    except Exception as check_error:
        logger.warning(f"Error checking agent connection state: {check_error}")
        try:
            # If we can't check the state, try to refresh anyway
            logger.info("Refreshing agent connection as precaution...")
            agent.refresh_llm_connection()
            logger.info("Agent connection refreshed successfully")
        except Exception as refresh_error:
            logger.error(f"Failed to refresh agent connection: {refresh_error}")
            # Try to create a completely new agent as last resort
            try:
                from agent import DataAnalysisAgent
                agent = DataAnalysisAgent()
                logger.info("Created new agent instance as fallback")
            except Exception as new_agent_error:
                logger.error(f"Failed to create new agent: {new_agent_error}")
                # Continue with existing agent and hope for the best

    cleanup_old_cleaned_files(app.config['UPLOAD_FOLDER'])

    # Initialize variables for all code paths
    clean_worksheets = {}
    skipped_worksheets = {}
    excluded_worksheets = {}
    worksheet_scores = {}
    optimization_report = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        print(f"Received {len(files)} file(s) for upload")
        
        # Clear session history when new file is uploaded
        clear_session_history()
        print("Session history cleared due to new file upload")
        
        # Check if all files are valid
        for file in files:
            if file.filename == '':
                return jsonify({'error': 'One or more files have no name'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type: {file.filename}. Please upload CSV, XLSX, or XLS files.'}), 400
        
        # Determine file types and handling strategy
        csv_files = [f for f in files if is_csv_file(f.filename)]
        excel_files = [f for f in files if not is_csv_file(f.filename)]
        
        # Validation logic
        if len(files) > 1:
            if excel_files:
                return jsonify({'error': 'Multiple file upload is only supported for CSV files. For Excel files, please upload one file at a time.'}), 400
            if len(csv_files) != len(files):
                return jsonify({'error': 'When uploading multiple files, all files must be CSV format.'}), 400

        # Handle multiple CSV files
        if len(csv_files) > 1:
            print(f"Processing {len(csv_files)} CSV files for intelligent merging...")

            # Save all CSV files temporarily
            csv_file_paths = []
            session_id = session_manager.get_current_session_id()

            for csv_file in csv_files:
                filename = secure_filename(csv_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
                csv_file.save(filepath)
                # Associate temp file with session
                file_cleanup_manager.associate_file_with_session(filepath, session_id, 'temp')
                csv_file_paths.append({
                    'path': filepath,
                    'filename': filename
                })

            try:
                # Use enhanced CSV combination with merge
                combined_excel_path, combined_filename, combined_worksheet_data = combine_csv_files_to_excel_with_merge(
                    csv_file_paths)

                # Associate combined file with session
                file_cleanup_manager.associate_file_with_session(combined_excel_path, session_id, 'upload')

                # NEW: Apply OnlyOffice optimization to combined CSV data
                print(f"Applying OnlyOffice optimization to combined CSV data...")
                optimized_combined_data, optimization_report = optimize_worksheet_data_for_onlyoffice(combined_worksheet_data)

                # Create optimized Excel file for OnlyOffice
                if optimization_report['worksheets_optimized'] > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    optimized_filename = f"combined_csvs_optimized_{len(csv_files)}_files_{timestamp}.xlsx"
                    optimized_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], optimized_filename)

                    with pd.ExcelWriter(optimized_excel_path, engine='openpyxl') as writer:
                        for sheet_name, df in optimized_combined_data.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # Use optimized file for OnlyOffice
                    final_file_path = optimized_excel_path
                    file_cleanup_manager.associate_file_with_session(optimized_excel_path, session_id, 'upload')

                    print(f"OnlyOffice optimization applied: {optimization_report['summary_message']}")
                else:
                    # No optimization needed
                    final_file_path = combined_excel_path
                    optimization_report = None

                # Clean up temporary CSV files
                for csv_info in csv_file_paths:
                    try:
                        os.remove(csv_info['path'])
                    except:
                        pass

                # Determine if data was merged or kept as separate worksheets
                if len(combined_worksheet_data) == 1 and 'Merged_Data' in combined_worksheet_data:
                    # Data was merged
                    print("CSV files were successfully merged into single dataset")
                    set_current_filename(f"Merged_CSVs_{len(csv_files)}_files.xlsx")
                    onlyoffice_display_filename = f"Merged_CSVs_{len(csv_files)}_files.xlsx"
                    has_multiple_worksheets = False
                else:
                    # Data kept as separate worksheets
                    print("CSV files kept as separate worksheets")
                    set_current_filename(f"Combined_CSVs_{len(csv_files)}_files.xlsx")
                    onlyoffice_display_filename = f"Combined_CSVs_{len(csv_files)}_files.xlsx"
                    has_multiple_worksheets = len(combined_worksheet_data) > 1

                # Set up session variables
                set_worksheet_data(combined_worksheet_data)
                set_active_worksheet(list(combined_worksheet_data.keys())[0])
                set_current_data(combined_worksheet_data[list(combined_worksheet_data.keys())[0]])

                # Set variables for later processing
                clean_worksheets = combined_worksheet_data

                print(f"Combined workbook created successfully with {len(combined_worksheet_data)} worksheet(s)")

                # Use the combined Excel file for OnlyOffice
                final_file_path = combined_excel_path

            except Exception as combine_error:
                # Clean up temporary files on error
                for csv_info in csv_file_paths:
                    try:
                        os.remove(csv_info['path'])
                    except:
                        pass
                raise combine_error

        # Handle single file (CSV or Excel)
        else:
            file = files[0]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Associate file with session
            session_id = session_manager.get_current_session_id()
            file_cleanup_manager.associate_file_with_session(filepath, session_id, 'upload')
            
            print(f"Processing single file: {filename}")
            set_current_filename(filename)
            final_file_path = filepath
            
            try:
                # Read the file with proper error handling and multi-worksheet support
                if filename.endswith('.csv'):
                    # CSV files - single worksheet only
                    try:
                        current_data = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            current_data = pd.read_csv(filepath, encoding='latin-1')
                        except UnicodeDecodeError:
                            current_data = pd.read_csv(filepath, encoding='cp1252')
                    
                    # For CSV, create single worksheet entry
                    set_worksheet_data({'Sheet1': current_data})
                    set_active_worksheet('Sheet1')
                    set_current_data(current_data)
                    has_multiple_worksheets = False
                    onlyoffice_display_filename = filename

                    # Set variables for later processing
                    clean_worksheets = {'Sheet1': current_data}
                else:
                    # Excel files - filter and read only clean worksheets
                    print("  Reading Excel file with enhanced worksheet filtering and recovery...")

                    try:
                        clean_worksheets, skipped_worksheets, worksheet_scores = filter_and_read_clean_worksheets(filepath)

                        if not clean_worksheets:
                            error_msg = "No properly formatted worksheets found in Excel file."
                            if skipped_worksheets:
                                error_msg += f" Skipped {len(skipped_worksheets)} worksheets due to formatting issues."

                                # Provide more helpful error message about what was tried
                                recovery_attempts = sum(1 for issues in skipped_worksheets.values() if len(issues) > 0)
                                error_msg += f" Recovery was attempted for worksheets with formatting issues."

                            raise Exception(error_msg)

                        # Apply text standardization BEFORE setting worksheet data
                        standardization_summary = None
                        if clean_worksheets:
                            print("\n=== APPLYING TEXT STANDARDIZATION ===")
                            standardized_worksheets, standardization_results = text_standardizer.standardize_worksheet_dict(
                                clean_worksheets
                            )

                            # Generate summary
                            standardization_summary = text_standardizer.get_standardization_summary(
                                standardization_results)

                            # Use standardized worksheets instead of original
                            clean_worksheets = standardized_worksheets

                            # Log results
                            if standardization_summary['total_variants_resolved'] > 0:
                                report = text_standardizer.generate_standardization_report(standardization_results)
                                print(report)

                            # Store standardization results in session for later reference
                            try:
                                session_obj = session_manager.get_current_session()
                                session_obj.text_standardization_results = standardization_results
                                session_obj.text_standardization_summary = standardization_summary
                            except Exception as e:
                                logger.warning(f"Could not store standardization results in session: {e}")

                        set_worksheet_data(clean_worksheets)

                        # NEW: Check if we should merge multiple worksheets FIRST
                        if len(get_worksheet_data()) > 1:
                            print("Checking if Excel worksheets should be merged...")

                            merge_attempted, merged_file_path, merged_filename, merged_or_original_data, merge_result = process_multi_worksheet_excel_with_merge(
                                get_worksheet_data(), filename, app.config['UPLOAD_FOLDER']
                            )

                            if merge_attempted and merged_file_path:
                                # Merge was successful - update everything to use merged data
                                print("Excel worksheets successfully merged into single dataset")

                                # Update session data to reflect merged result
                                set_worksheet_data(merged_or_original_data)  # Now contains only 'Merged_Data'
                                set_active_worksheet('Merged_Data')  # Direct assignment - no need to find "best"
                                set_current_data(merged_or_original_data['Merged_Data'])  # set current data
                                set_current_filename(f"{os.path.splitext(filename)[0]}_merged.xlsx")

                                # Use merged file for later processing
                                final_file_path = merged_file_path
                                onlyoffice_display_filename = f"{os.path.splitext(filename)[0]}_merged.xlsx"

                                # Associate merged file with session
                                session_id = session_manager.get_current_session_id()
                                file_cleanup_manager.associate_file_with_session(merged_file_path, session_id, 'upload')

                                print(f"Merge summary: {merge_result.merge_summary}")
                                print(f"Final merged data shape: {merged_or_original_data['Merged_Data'].shape}")

                                # Reset template data since structure changed
                                try:
                                    session_obj = session_manager.get_current_session()
                                    if session_obj.detected_template:
                                        print("Resetting template data due to worksheet merge")
                                        session_obj.detected_template = None
                                        session_obj.template_validation_results = None
                                        # Keep manual_template_override in case user wants to reapply manually
                                except Exception as e:
                                    print(f"Warning: Error resetting template data after merge: {e}")

                                # Skip the best sheet selection since we now have only one sheet
                                has_multiple_worksheets = False
                            else:
                                print("Excel worksheets kept as separate worksheets (no merge performed)")
                                # Continue to best sheet selection below

                        # Select the worksheet with the most data as active
                        if len(get_worksheet_data()) == 1:
                            set_active_worksheet(list(get_worksheet_data().keys())[0])
                            print(f"  Single worksheet - Active: '{get_active_worksheet()}'")
                        else:
                            # Find worksheet with highest data score
                            best_sheet = max(worksheet_scores.keys(), key=lambda sheet: worksheet_scores[sheet]['score'])
                            set_active_worksheet(best_sheet)

                            print(f"  Selected worksheet with most data as active: '{get_active_worksheet()}'")
                            print(f"  Data comparison across worksheets:")

                            # Sort worksheets by data score for display
                            sorted_sheets = sorted(worksheet_scores.items(),
                                                   key=lambda x: x[1]['score'], reverse=True)

                            for i, (sheet_name, scores) in enumerate(sorted_sheets):
                                active_marker = " (ACTIVE)" if sheet_name == get_active_worksheet() else ""
                                print(f"       '{sheet_name}': {scores['non_null_cells']:,} data cells, "
                                      f"{scores['rows']:,} rows × {scores['columns']} cols, "
                                      f"{scores['data_density']}% density{active_marker}")

                        set_current_data(get_worksheet_data()[get_active_worksheet()])
                        has_multiple_worksheets = len(get_worksheet_data()) > 1

                        print(f"  Total clean worksheets loaded: {len(get_worksheet_data())}")

                        # CREATE CLEANED EXCEL FILE FOR ONLYOFFICE
                        optimization_report = None
                        try:
                            cleaned_file_path, cleaned_filename, optimization_report = create_cleaned_excel_for_onlyoffice(
                                get_worksheet_data(), filename, app.config['UPLOAD_FOLDER']
                                # Note: using get_worksheet_data() (filtered) not clean_worksheets
                            )

                            # Use cleaned file for OnlyOffice instead of original
                            final_file_path = cleaned_file_path
                            onlyoffice_display_filename = f"{os.path.splitext(filename)[0]}_optimized.xlsx"

                            # Create comprehensive summary of what was cleaned and excluded
                            total_original = len(clean_worksheets) + len(skipped_worksheets)
                            total_excluded = len(skipped_worksheets) + len(excluded_worksheets)

                            if total_excluded > 0:
                                exclusion_details = []
                                if len(skipped_worksheets) > 0:
                                    exclusion_details.append(f"{len(skipped_worksheets)} had formatting issues")
                                if len(excluded_worksheets) > 0:
                                    exclusion_details.append(f"{len(excluded_worksheets)} couldn't be analyzed")

                                exclusion_summary = " and ".join(exclusion_details)
                                print(f" OnlyOffice will show optimized file ({len(get_worksheet_data())} worksheets kept, {total_excluded} removed: {exclusion_summary})")
                            else:
                                print(f" OnlyOffice will show optimized file (all {len(get_worksheet_data())} worksheets kept, data optimized for browser display)")

                            # Add optimization information to display message
                            if optimization_report and optimization_report['worksheets_optimized'] > 0:
                                print(f" OnlyOffice optimization: {optimization_report['summary_message']}")

                        except Exception as cleanup_error:
                            print(f"  Warning: Could not create optimized file: {cleanup_error}")
                            print(f"  OnlyOffice will show original file (may have formatting and memory issues for larger datasets)")
                            # Fallback to original file
                            final_file_path = filepath
                            onlyoffice_display_filename = filename
                            optimization_report = None

                        # Create enhanced user-friendly report
                        worksheet_report = create_worksheet_report_with_exclusions(
                            get_worksheet_data(),      # Final analyzable worksheets
                            skipped_worksheets,  # Worksheets skipped during recovery
                            excluded_worksheets,  # Worksheets excluded due to analysis issues
                            worksheet_scores,
                            get_active_worksheet()
                        )
                        print(worksheet_report)
                        
                        # Additional logging for recovery information
                        recovered_sheets = []
                        for sheet_name, df in clean_worksheets.items():
                            # Check if this sheet was recovered (you might want to track this in the enhanced functions)
                            if hasattr(df, '_recovery_info'):  # If we add this metadata
                                recovered_sheets.append(sheet_name)

                        if recovered_sheets:
                            print(f" Successfully recovered data from {len(recovered_sheets)} worksheet(s) with formatting issues: {recovered_sheets}")

                    except Exception as excel_error:
                        error_message = f"Error reading Excel file: {str(excel_error)}"

                        # ADD specific guidance for common Excel issues
                        error_str = str(excel_error).lower()
                        if "no properly formatted worksheets found" in error_str:
                            error_message += "\n\n Tips for Excel files:"
                            error_message += "\n. Ensure data starts near cell A1"
                            error_message += "\n. Avoid complex merged headers or formatting"
                            error_message += "\n. Make sure worksheets contain actual tabular data"
                            error_message += "\n. Consider saving as a simpler .xlsx format"

                        print(f"  Excel processing error: {error_message}")
                        raise Exception(error_message)

            except Exception as read_error:
                print(f"Error reading file: {str(read_error)}")
                return jsonify({'error': f'Error reading file: {str(read_error)}'}), 500
        
        print(f"File(s) loaded successfully. Active worksheet: {get_active_worksheet()}, Shape: {get_current_data().shape}")
        
        # Verify data is properly loaded
        if get_current_data() is None or get_current_data().empty:
            raise Exception("Data was loaded but appears to be empty or None")
        
        print(f"Data verification passed. Columns: {list(get_current_data().columns)[:5]}...")  # Show first 5 columns
        
        # Create OnlyOffice document
        print("Creating OnlyOffice document...")
        document_url, document_key = create_onlyoffice_document(final_file_path, onlyoffice_display_filename)

        print("Generating data summaries and filtering analyzable worksheets...")

        # Enhanced Template detection - supports multi-worksheet templates
        detected_template = None
        template_validation_results = None

        try:
            # Prepare data for template detection
            original_filename = get_current_filename()
            worksheet_data_for_template = get_worksheet_data()  # All worksheets

            logger.info(
                f"Starting template detection for file '{original_filename}' with {len(worksheet_data_for_template)} worksheet(s)")

            # Use multi-worksheet template detection
            detected_template = template_manager.detect_template(worksheet_data_for_template, original_filename)

            if detected_template:
                logger.info(f"Template '{detected_template.name}' detected (Type: {detected_template.template_type})")

                # Validate all worksheets against the template
                template_validation_results = validate_data_against_template(worksheet_data_for_template, detected_template)

                # Store in session
                session_obj = session_manager.get_current_session()
                session_obj.detected_template = detected_template
                session_obj.template_validation_results = template_validation_results

                # Log template match details
                if template_validation_results and 'summary' in template_validation_results:
                    summary = template_validation_results['summary']
                    logger.info(
                        f"Template validation: {summary['found_worksheets']}/{summary['total_template_worksheets']} worksheets matched ({summary['match_percentage']:.1f}%)")

            else:
                # Get template suggestions if no auto-detection
                template_suggestions = get_template_suggestions_for_data(worksheet_data_for_template, original_filename)
                if template_suggestions:
                    logger.info(f"Template suggestions available: {[s['name'] for s in template_suggestions]}")
                    # We'll add these to the response later

        except Exception as e:
            logger.error(f"Error in template detection: {e}")
            import traceback
            traceback.print_exc()
            detected_template = None

        # Test each worksheet individually for data summary generation
        analyzable_worksheets = {}
        worksheet_summaries = {}
        
        for sheet_name, sheet_data in get_worksheet_data().items():
            print(f"  Testing data analysis capability for '{sheet_name}'...")

            try:
                try:
                    # Use enhanced analysis that supports template context
                    if detected_template:
                        # For template-detected worksheets, use enhanced analysis
                        summary, _ = analyze_data_with_template_support(sheet_data, original_filename)

                        # Add specific template information for this worksheet
                        summary['template_info'].update({
                            'template_id': detected_template.id,
                            'template_name': detected_template.name,
                            'template_domain': detected_template.domain,
                            'template_version': detected_template.version,
                            'template_type': detected_template.template_type,
                            'enhanced_context_available': True
                        })

                        # If this is the active worksheet, include validation results
                        if sheet_name == get_active_worksheet() and template_validation_results:
                            summary['template_validation'] = template_validation_results

                            # Add worksheet-specific validation if available
                            if 'worksheet_validations' in template_validation_results:
                                for ws_validation in template_validation_results['worksheet_validations']:
                                    if ws_validation['matched_worksheet'] == sheet_name:
                                        summary['worksheet_validation'] = ws_validation
                                        break

                    else:
                        # Standard analysis for non-template files
                        summary, _ = analyze_data_with_template_support(sheet_data, original_filename)
                        if 'template_info' not in summary:
                            summary['template_info'] = {'enhanced_context_available': False}

                except Exception as analysis_error:
                    # Fallback to original analyze_data if enhanced analysis fails
                    logger.warning(f"Enhanced analysis failed for {sheet_name}, using fallback: {analysis_error}")
                    summary = analyze_data_standard(sheet_data)
                    summary['template_info'] = {'enhanced_context_available': False}

                # Additional validation checks for summary quality
                if not summary or not isinstance(summary, dict):
                    raise Exception("Summary generation returned invalid result")

                # Check if essential summary components exist
                required_fields = ['shape', 'columns', 'dtypes']
                missing_fields = [field for field in required_fields if field not in summary]
                if missing_fields:
                    raise Exception(f"Summary missing required fields: {missing_fields}")

                # Check if we have actual columns and data
                if not summary.get('columns') or len(summary.get('columns', [])) == 0:
                    raise Exception("No valid columns found in summary")

                if not summary.get('shape') or summary.get('shape', [0, 0])[0] == 0:
                    raise Exception("No data rows found in summary")

                # If we reach here, the worksheet is analyzable
                analyzable_worksheets[sheet_name] = sheet_data
                worksheet_summaries[sheet_name] = summary
                print(
                    f"    '{sheet_name}': Analyzable ({summary['shape'][0]} rows, {len(summary['columns'])} columns)")

            except Exception as summary_error:
                # This worksheet cannot be properly analyzed - exclude it
                excluded_worksheets[sheet_name] = str(summary_error)
                print(f"   '{sheet_name}': Excluded - {summary_error}")

        # Add template suggestions to the main data summary if no template was detected
        if not detected_template and 'template_suggestions' in locals():
            # Add suggestions to the active worksheet's summary
            active_worksheet = get_active_worksheet()
            if active_worksheet in worksheet_summaries:
                worksheet_summaries[active_worksheet]['template_suggestions'] = template_suggestions
        
        # Check if we have any analyzable worksheets left
        if not analyzable_worksheets:
            total_worksheets = len(get_worksheet_data())
            error_details = []
            for sheet_name, error in excluded_worksheets.items():
                error_details.append(f"'{sheet_name}': {error}")

            error_msg = f"No analyzable worksheets found in Excel file. All {total_worksheets} worksheet(s) were excluded due to analysis issues:"
            for detail in error_details[:5]:  # Show first 5 errors
                error_msg += f"\n• {detail}"
            if len(error_details) > 5:
                error_msg += f"\n• ... and {len(error_details) - 5} more issues"

            error_msg += "\n\n Tips to fix these issues:"
            error_msg += "\n. Ensure worksheets contain actual tabular data"
            error_msg += "\n. Remove complex formatting, charts, and pivot tables"
            error_msg += "\n. Make sure data starts near cell A1"
            error_msg += "\n. Verify columns have proper headers"

            raise Exception(error_msg)

        # Update worksheet_data to only include analyzable worksheets
        original_count = len(get_worksheet_data())
        set_worksheet_data(analyzable_worksheets)
        set_analyzable_worksheets(analyzable_worksheets)
        set_excluded_worksheets(excluded_worksheets)
        final_count = len(get_worksheet_data())

        # Log the filtering results
        if excluded_worksheets:
            print(f"  Worksheet filtering complete: {final_count}/{original_count} worksheets are analyzable")
            print(f"  Excluded {len(excluded_worksheets)} worksheet(s): {list(excluded_worksheets.keys())}")
        else:
            print(f"  All {final_count} worksheets are analyzable")

        # Re-select active worksheet (original selection might have been excluded)
        if get_active_worksheet() not in get_worksheet_data():
            print(f"  Originally selected worksheet '{get_active_worksheet()}' was excluded")

            # Find the worksheet with the most data from remaining analyzable worksheets
            if len(get_worksheet_data()) == 1:
                set_active_worksheet(list(get_worksheet_data().keys())[0])
                print(f"  New active worksheet: '{get_active_worksheet()}' (only remaining option)")
            else:
                # Recalculate scores for remaining worksheets
                remaining_scores = {}
                for sheet_name in get_worksheet_data().keys():
                    if sheet_name in worksheet_scores:
                        remaining_scores[sheet_name] = worksheet_scores[sheet_name]

                if remaining_scores:
                    set_active_worksheet(max(remaining_scores.keys(),
                                           key=lambda sheet: remaining_scores[sheet]['score']))
                    print(f"  New active worksheet: '{get_active_worksheet()}' (most data among remaining)")
                else:
                    # Fallback: just pick the first one
                    set_active_worksheet(list(get_worksheet_data().keys())[0])
                    print(f"  New active worksheet: '{get_active_worksheet()}' (fallback selection)")

        # Update current_data to the final active worksheet
        set_current_data(get_worksheet_data()[get_active_worksheet()])
        set_data_summary(worksheet_summaries[get_active_worksheet()])

        # Update shared state
        shared_state.update_data(get_current_data(), get_current_filename(), get_data_summary(), get_current_document_key(), get_current_document_url())

        # Update has_multiple_worksheets flag
        has_multiple_worksheets = len(get_worksheet_data()) > 1

        # Analyze schema conflicts if multiple worksheets remain
        if len(get_worksheet_data()) > 1:
            print("  Analyzing schema compatibility among analyzable worksheets...")
            try:
                schema_conflicts = analyze_worksheet_schema(get_worksheet_data())
            except Exception as schema_error:
                print(f"  Schema analysis warning: {schema_error}")

        print(f" Data summary generation complete. Active worksheet: '{get_active_worksheet()}'")
        
        # Create response
        file_description = f"{len(csv_files)} CSV files combined" if len(csv_files) > 1 else get_current_filename()
        display_filename = file_description if len(csv_files) <= 1 else f"Combined_CSVs_{len(csv_files)}_files.xlsx"

        response_data = {
            'success': True,
            'filename': file_description,
            'display_filename': display_filename,
            'shape': list(get_current_data().shape),
            'columns': list(get_current_data().columns),
            'summary': get_data_summary(),
            'document_url': document_url,
            'document_key': document_key,
            'has_multiple_worksheets': has_multiple_worksheets,
            'active_worksheet': get_active_worksheet(),
            'is_combined_csv': len(csv_files) > 1,
            'csv_count': len(csv_files) if len(csv_files) > 1 else 0,
        }

        # Add OnlyOffice optimization information
        if optimization_report:
            response_data['onlyoffice_optimization'] = {
                'was_optimized': optimization_report['worksheets_optimized'] > 0,
                'worksheets_optimized': optimization_report['worksheets_optimized'],
                'memory_saved_mb': optimization_report['total_memory_saved_mb'],
                'optimization_summary': optimization_report['summary_message'],
                'optimization_warnings': []
            }

            # Collect all warnings from worksheet optimizations
            for ws_name, ws_report in optimization_report['optimization_details'].items():
                for warning in ws_report.get('warnings', []):
                    response_data['onlyoffice_optimization']['optimization_warnings'].append(f"{ws_name}: {warning}")
        else:
            response_data['onlyoffice_optimization'] = {
                'was_optimized': False,
                'optimization_summary': 'No optimization applied'
            }

        # Add merge information to response if applicable
        merge_info = {}
        current_filename = get_current_filename()
        if current_filename and ('merged' in current_filename.lower() or 'Merged_' in get_active_worksheet()):
            merge_info = {
                'data_was_merged': True,
                'merge_type': 'csv_collection' if len(csv_files) > 1 else 'excel_worksheets',
                'original_count': len(csv_files) if len(csv_files) > 1 else len(
                    clean_worksheets) if 'clean_worksheets' in locals() else 0,
                'active_worksheet': get_active_worksheet()
            }
        else:
            merge_info = {
                'data_was_merged': False,
                'merge_type': None
            }

        response_data['merge_info'] = merge_info

        # Only add excel_processing_info for Excel files
        current_filename = get_current_filename()
        if not (current_filename and current_filename.endswith('.csv')) and len(csv_files) <= 1:
            response_data['excel_processing_info'] = {
                'total_worksheets_found': len(clean_worksheets) + len(skipped_worksheets),
                'clean_worksheets': len(clean_worksheets),
                'skipped_worksheets': len(skipped_worksheets),
                'worksheets_recovered': len([name for name in clean_worksheets.keys()
                                             if hasattr(clean_worksheets[name], '_recovery_info')])
            }
        else:
            response_data['excel_processing_info'] = None

        # Enhanced success message with merge information
        if len(csv_files) > 1:
            if merge_info['data_was_merged']:
                csv_info = f" Intelligently merged {len(csv_files)} CSV files into a single unified dataset based on common columns."
                worksheet_info = " All your data is now in one worksheet for easier analysis."
                multi_sheet_tip = " Ask questions about your merged data!"
            else:
                csv_info = f" Combined {len(csv_files)} CSV files into separate worksheets (no common columns for merging)."
                worksheet_info = f" The file contains {len(get_worksheet_data())} worksheets - you can switch between them using OnlyOffice."
                multi_sheet_tip = " You can ask questions about specific worksheets or analyze data across all worksheets!"
            cleaning_info = ""
        else:
            csv_info = ""
            if merge_info['data_was_merged']:
                worksheet_info = " Multiple worksheets were intelligently merged into a single dataset for easier analysis."
                multi_sheet_tip = " Ask questions about your merged data!"
            elif has_multiple_worksheets:
                worksheet_info = f" The file contains {len(get_worksheet_data())} worksheets - you can switch between them using OnlyOffice."
                multi_sheet_tip = " You can ask questions about specific worksheets or analyze data across all worksheets!"
            else:
                worksheet_info = ""
                multi_sheet_tip = ""

            # Add cleaning information
            cleaning_info = ""
            if 'cleaned_file_path' in locals() and cleaned_file_path != filepath:
                total_original = len(clean_worksheets) + len(skipped_worksheets)
                total_excluded = len(skipped_worksheets) + len(excluded_worksheets)

                if total_excluded > 0:
                    exclusion_types = []
                    if len(skipped_worksheets) > 0:
                        exclusion_types.append(f"{len(skipped_worksheets)} with formatting issues")
                    if len(excluded_worksheets) > 0:
                        exclusion_types.append(f"{len(excluded_worksheets)} that couldn't be analyzed")

                    exclusion_desc = " and ".join(exclusion_types)
                    cleaning_info = f" Cleaned up your Excel file - kept {len(get_worksheet_data())} analyzable worksheet(s) and removed {total_excluded} ({exclusion_desc}). All data now starts properly from cell A1."
                else:
                    cleaning_info = " Cleaned up your Excel file - all data now starts properly from cell A1."

                cleaning_info += " The OnlyOffice editor shows only your analyzable, cleaned data."

        # Add OnlyOffice optimization information to the message
        optimization_info = ""
        if optimization_report and optimization_report['worksheets_optimized'] > 0:
            optimization_info = f"  OnlyOffice display optimized for better browser performance - saved ~{optimization_report['total_memory_saved_mb']:.0f}MB memory."

            # Add specific optimization warnings if any
            total_warnings = sum(len(ws_report.get('warnings', [])) for ws_report in optimization_report['optimization_details'].values())
            if total_warnings > 0:
                optimization_info += f" Note: Large dataset was intelligently sampled for display while your full data remains available for analysis."

        # Final success message
        display_filename_for_message = get_current_filename() or "your file"
        success_message = (f'File "{display_filename_for_message}" loaded successfully!{csv_info} '
                           f'Your data is now in the OnlyOffice spreadsheet with {get_current_data().shape[0]} rows and {get_current_data().shape[1]} columns.'
                           f'{worksheet_info}{cleaning_info}{optimization_info}{multi_sheet_tip} '
                           f'Ask me questions about your data and I\'ll provide analysis and visualizations!')

        print(f"Upload completed successfully \n {success_message}")

        # Add enhanced template information to the response
        template_info = {}
        session_obj = session_manager.get_current_session()

        if session_obj.detected_template:
            template_info = {
                'detected': True,
                'template_id': session_obj.detected_template.id,
                'template_name': session_obj.detected_template.name,
                'template_domain': session_obj.detected_template.domain,
                'template_type': session_obj.detected_template.template_type,
                'multi_worksheet_template': session_obj.detected_template.template_type in ['multi_worksheet',
                                                                                            'csv_collection'],
                'validation_results': session_obj.template_validation_results
            }

            # Add worksheet mapping information for multi-worksheet templates
            if (session_obj.detected_template.template_type in ['multi_worksheet', 'csv_collection'] and
                    session_obj.template_validation_results and
                    'worksheet_validations' in session_obj.template_validation_results):
                template_info['worksheet_mapping'] = session_obj.template_validation_results['worksheet_validations']

        else:
            # Get template suggestions for the response
            template_suggestions = get_template_suggestions_for_data(get_worksheet_data(), original_filename)
            template_info = {
                'detected': False,
                'suggestions': template_suggestions,
                'suggestions_count': len(template_suggestions)
            }

        response_data['template_info'] = template_info

        return jsonify(response_data)
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file(s): {str(e)}'}), 500


def _generate_temporal_context(ws_summary):
    """Generate temporal context section"""
    temporal_analysis = ws_summary.get('temporal_columns_analysis', {})
    total_temporal = temporal_analysis.get('summary', {}).get('total_temporal_columns', 0)

    if total_temporal == 0:
        return ""

    context_parts = [f"### TEMPORAL COLUMNS DETECTED ({total_temporal} total):"]

    # Temporal value columns
    temporal_value_cols = temporal_analysis.get('temporal_value_columns', [])
    if temporal_value_cols:
        context_parts.append("**Temporal Value Columns (Business Date Formats):**")
        for col in temporal_value_cols:
            sample_matches = ', '.join(col.get('sample_matches', [])[:2])
            sorting_note = " ⚠️NEEDS SORTING" if not col.get('is_properly_sorted', True) else ""
            context_parts.append(
                f"- {col['column']}: {col['pattern_type']} ({sample_matches}){sorting_note}")

    # Temporal name columns
    temporal_name_cols = temporal_analysis.get('temporal_name_columns', [])
    if temporal_name_cols:
        col_names = ', '.join([col['column'] for col in temporal_name_cols])
        context_parts.append(f"**Temporal Column Names:** {col_names}")

    # Critical sorting reminder
    requires_sorting = temporal_analysis.get('summary', {}).get('requires_sorting_attention', [])
    if requires_sorting:
        context_parts.append(
            "**CRITICAL:** Always sort temporal columns chronologically before time series analysis!")
        context_parts.append(f"**URGENT SORTING NEEDED:** {', '.join(requires_sorting)}")
    else:
        context_parts.append(
            "**CRITICAL:** Always sort temporal columns chronologically before time series analysis!")

    return '\n'.join(context_parts)

@app.route('/query', methods=['POST'])
def process_query():
    """Process AI query with multi-worksheet support and session context"""
    global agent

    # Get data from shared state (now session-aware)
    #current_data = shared_state.get_current_data()
    #worksheet_data = shared_state.get_worksheet_data()
    #data_summary = shared_state.get_data_summary()
    #worksheet_summaries = shared_state.get_worksheet_summaries()
    #active_worksheet = shared_state.get_active_worksheet()

    current_data = get_current_data()
    worksheet_data = get_worksheet_data()
    data_summary = get_data_summary()
    worksheet_summaries = get_worksheet_summaries()
    active_worksheet = get_active_worksheet()

    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"Processing query: {user_query}")
        print(f"Active worksheet: {active_worksheet}")
        print(f"Available worksheets: {list(worksheet_data.keys()) if worksheet_data else ['None']}")
        print(f"Current data shape: {current_data.shape if current_data is not None else 'None'}")
        print(f"Current data type: {type(current_data)}")
        print(f"Session history length: {len(get_session_history())}")
        
        # Verify current_data is valid
        if current_data is None:
            print("ERROR: current_data is None!")
            return jsonify({
                'success': False,
                'error': 'No data available for analysis. Please upload a file first.',
                'result': '',
                'code': '',
                'messages': [],
                'query': user_query,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Create comprehensive context string for the AI including all worksheets
        context = ""
        
        if len(worksheet_data) > 1:
            # Multi-worksheet context
            context += f"MULTI-WORKSHEET EXCEL FILE:\n"
            context += f"Total worksheets: {len(worksheet_data)}\n"
            context += f"Active worksheet: {active_worksheet}\n\n"
            
            for ws_name, ws_summary in worksheet_summaries.items():
                context += f"=== Worksheet: {ws_name} ===\n"
                context += f"Shape: {ws_summary['shape'][0]} rows × {ws_summary['shape'][1]} columns\n"
                # Ensure all column names are strings before joining
                column_names = [str(col) for col in ws_summary['columns']]
                context += f"Columns: {', '.join(column_names)}\n"

                # Same for numeric and categorical columns
                numeric_cols = [str(col) for col in ws_summary['numeric_columns']]
                categorical_cols = [str(col) for col in ws_summary['categorical_columns']]

                context += f"Numeric columns: {', '.join(numeric_cols)}\n"
                context += f"Categorical columns: {', '.join(categorical_cols)}\n"
                
                # Sample values for this worksheet
                if ws_name in worksheet_data:
                    sample_values = {}
                    for col in worksheet_data[ws_name].columns:  # Show sample values for first 3 columns
                        sample_vals = worksheet_data[ws_name][col].dropna().head().tolist()
                        sample_values[col] = [str(val) for val in sample_vals]
                    context += f"Sample Values: {sample_values}\n"
                
                missing_info = [f"{col}: {count} missing" for col, count in ws_summary['missing_values'].items() if count > 0]
                if missing_info:
                    context += f"Missing Values: {', '.join(missing_info)}\n"  # Show first 5
                context += "\n"

                context += f"""
                Numeric Columns Statistics:
                {chr(10).join([f"- {col}: {stat}" for col, stat in ws_summary['numeric_stats'].items()])}
                """
                context += "\n"
                context += f"""
                Categorical Columns: Common values and their frequency
                {chr(10).join([f"- {col}: {stat}" for col, stat in ws_summary['categorical_stats'].items()])}
                """
                context += "\n"
                context += f"""
                {_generate_temporal_context(ws_summary)}
                """
                
            context += f"CURRENT ANALYSIS SCOPE:\n"
            context += f"- Use 'current_data' for analysis specific to the active worksheet ({active_worksheet})\n"
            context += f"- Use 'worksheet_data' dictionary to access any specific worksheet\n"
            context += f"- Use 'merge_worksheets(worksheet_data)' to combine all worksheets for cross-sheet analysis\n"
            
        else:
            # Single worksheet context (backward compatibility)
            ws_summary = data_summary
            sample_values = {}
            for col in current_data.columns:  # Show sample values for first 5 columns
                sample_vals = current_data[col].dropna().head().tolist()
                sample_values[col] = [str(val) for val in sample_vals]
            
            context = f"""
Dataset Information:
Shape: {ws_summary['shape'][0]} rows × {ws_summary['shape'][1]} columns

Columns: {', '.join(ws_summary['columns'])}

Data Types:
{chr(10).join([f"- {col}: {dtype}" for col, dtype in ws_summary['dtypes'].items()])}

Numeric columns: {', '.join(ws_summary['numeric_columns'])}
Categorical columns: {', '.join(ws_summary['categorical_columns'])}

Sample Values:
{chr(10).join([f"- {col}: {sample_values.get(col, [])}" for col in list(sample_values.keys())])}

Missing Values:
{chr(10).join([f"- {col}: {count} missing" for col, count in ws_summary['missing_values'].items() if count > 0])}

Numeric Columns Statistics:
{chr(10).join([f"- {col}: {stat}" for col, stat in ws_summary['numeric_stats'].items()])}

Categorical Columns: Common values and their frequency
{chr(10).join([f"- {col}: {stat}" for col, stat in ws_summary['categorical_stats'].items()])}

Temporal Columns Analysis:
{chr(10).join([f"- {k}: {v}" for k, v in ws_summary['temporal_columns_analysis'].items()])}

{_generate_temporal_context(ws_summary)}
"""
        context += f"""
DataFrame variable names:
- current_data: Active worksheet data ({active_worksheet})
- worksheet_data: Dictionary of all worksheets (keys: {list(worksheet_data.keys())})
- merge_worksheets(): Function to intelligently combine worksheets

Use these exact variable names to access the data in your Python code.
        """

        # ENHANCED TEMPLATE SUPPORT (Multi-Worksheet)
        # ============================================================================
        # Create enhanced context if template is available
        session_obj = session_manager.get_current_session()
        detected_template = session_obj.detected_template

        if detected_template:
            # Pass worksheet data for multi-worksheet template context
            enhanced_context = create_enhanced_context_for_query(
                context,
                user_query,
                detected_template,
                worksheet_data  # Pass all worksheet data for multi-worksheet templates
            )
            context = enhanced_context

            # NEW: Add enhanced context if available
            if data_summary.get('enhanced_profiling_available', False):
                context = format_enhanced_context_for_llm(data_summary, context)

            # Log template usage
            template_type_msg = f" (Type: {detected_template.template_type})"
            if detected_template.template_type in ['multi_worksheet', 'csv_collection']:
                template_type_msg += f" with {len(worksheet_data)} worksheets"

            logger.info(f"Using enhanced context from template: {detected_template.name}{template_type_msg}")
            print(f"Enhanced context applied using template: {detected_template.name}{template_type_msg}")
        else:
            print("No template detected, using standard context")
            # NEW: Add enhanced context if available
            if data_summary.get('enhanced_profiling_available', False):
                context = format_enhanced_context_for_llm(data_summary, context)

        # Enhanced session history context for better follow-up support
        current_session_history = get_session_history()
        if current_session_history:
            try:
                # Use enhanced context creation
                context = enhance_session_context(context, current_session_history, user_query)
                logger.info(
                    f"Enhanced session context applied with {len(current_session_history)} history entries")
            except Exception as context_error:
                logger.warning(f"Failed to enhance session context: {context_error}")
                # Fallback to basic session history (existing code)
                context += f"\n\nSESSION HISTORY (Previous {len(current_session_history)} interactions):\n"
                for i, entry in enumerate(current_session_history[-10:], 1):
                    context += f"\n{i}. USER: {entry['query']}\n"
                    if entry.get('result_summary'):
                        context += f"   RESULT: {entry['result_summary']}\n"

        print(f"Context created. Active worksheet: {active_worksheet}")
        if len(worksheet_data) > 1:
            print(f"Multi-worksheet context includes {len(worksheet_data)} worksheets")
        print(f"Session history entries: {len(get_session_history())}")

        if len(current_session_history) > 0:
            print("DEBUG: follow-up Context: \n", context)
        else:
            print("DEBUG: init Context: \n", context)

        # Process query with the agent
        try:
            result = agent.process_query(
                user_query,
                context,
                current_data=current_data,
                worksheet_data=worksheet_data,
                merge_worksheets_func=merge_worksheets
            )
        except Exception as agent_error:
            # Handle token refresh or other agent errors
            error_str = str(agent_error).lower()
            if any(keyword in error_str for keyword in ['access denied', 'permission', '401', 'unauthorized', 'token']):
                logger.warning(f"Authentication/permission error: {agent_error}")
                try:
                    # Force token refresh and reinitialize agent
                    from agent import token_manager
                    if token_manager:
                        token_manager.force_refresh()

                    agent = DataAnalysisAgent()
                    logger.info("Agent reinitialized after auth error")

                    # Retry the query once
                    result = agent.process_query(user_query, 
                                                 context, 
                                                 current_data=current_data,
                                                 worksheet_data=worksheet_data, 
                                                 merge_worksheets_func=merge_worksheets)
                except Exception as retry_error:
                    return jsonify({
                        'success': False,
                        'error': f'Authentication failed: Please check your Azure permissions. Details: {str(retry_error)}',
                        'result': '', 
                        'code': '', 
                        'messages': [],
                        'query': user_query, 
                        'timestamp': datetime.now().isoformat(),
                        'output_id': str(uuid.uuid4())
                    }), 500
                
            elif 'token' in error_str or 'authentication' in error_str or '401' in error_str:
                logger.warning("Agent query failed due to authentication, attempting to reinitialize agent...")
                try:
                    # Reinitialize agent with fresh token
                    agent = DataAnalysisAgent()
                    logger.info("Agent reinitialized successfully")

                    # Retry the query
                    result = agent.process_query(
                        user_query,
                        context,
                        current_data=current_data,
                        worksheet_data=worksheet_data,
                        merge_worksheets_func=merge_worksheets
                    )
                except Exception as retry_error:
                    logger.error(f"Failed to reinitialize agent and retry: {retry_error}")
                    return jsonify({
                        'success': False,
                        'error': f'Authentication error: {str(retry_error)}',
                        'result': '',
                        'code': '',
                        'messages': [],
                        'query': user_query,
                        'timestamp': datetime.now().isoformat(),
                        'output_id': str(uuid.uuid4())
                    }), 500    
            else:
                # Re-raise non-authentication errors
                raise agent_error
                    
        print(f"Query processed. Success: {result['success']}")
        if not result['success']:
            print(f"Error details: {result['error']}")

        # Update session history with successful queries
        if result['success']:
            # since plot_data is a separate key, ensure result.result doesn't contain any plot objects
            result['result'] = remove_plot_objects_from_result(result['result'])

            # Use enhanced result summary creation
            result_summary = create_result_summary(result.get('result'))

            # Check if this is a follow-up query
            current_history = get_session_history()
            is_follow_up = is_follow_up_query(user_query, current_history)

            # Enforce 10 query limit BEFORE adding new entry
            if len(current_history) >= 10:
                logger.info(f"Follow-up limit reached ({len(current_history)} queries), removing oldest")
                # Keep only last 9 entries to make room for new one
                current_history = current_history[-9:]
                set_session_history(current_history)

            # Add to session history with enhanced context
            session_entry = {
                'query': user_query,
                'code': result.get('code', ''),
                'result_summary': result_summary,  # Now properly comprehensive
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'is_follow_up': is_follow_up,
                'result_type': result.get('result_type', 'unknown'),
                'has_plot': bool(result.get('plot_data', {})),
                'worksheet_used': active_worksheet if worksheet_data and len(worksheet_data) > 1 else None
            }

            add_to_session_history(session_entry)

            # Log session status
            updated_history = get_session_history()
            print(f"Session history updated: {len(updated_history)}/10 entries")
            print(f"Query type: {'Follow-up' if is_follow_up else 'New'} query")
            if result_summary:
                print(f"Result summary length: {len(result_summary)} chars")

            # Generate smart follow-up suggestions
            try:
                followup_suggestions = get_follow_up_suggestions(updated_history)
                if followup_suggestions:
                    result['followup_suggestions'] = followup_suggestions
                    logger.info(f"Added {len(followup_suggestions)} follow-up suggestions")
            except Exception as suggestion_error:
                logger.warning(f"Failed to generate follow-up suggestions: {suggestion_error}")

        # Add session status info to response
        result['session_info'] = {
            'history_count': len(get_session_history()),
            'max_follow_ups': 10,
            'remaining_follow_ups': max(0, 10 - len(get_session_history())),
            'is_follow_up': is_follow_up if 'is_follow_up' in locals() else False
        }

        # Add metadata to result for frontend tracking
        result['query'] = user_query
        result['timestamp'] = datetime.now().isoformat()
        result['output_id'] = str(uuid.uuid4())

        # Convert all pandas types to JSON serializable types
        cleaned_result = convert_pandas_types(result)
        
        return jsonify(cleaned_result)
    
    except Exception as e:
        error_msg = f'Error processing query: {str(e)}'
        print(f"Query processing error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg,
            'result': '',
            'code': '',
            'messages': [],
            'query': user_query,
            'timestamp': datetime.now().isoformat(),
            'output_id': str(uuid.uuid4())
        }), 500

###### Helper function to get session status
@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Get current session status including follow-up count"""
    try:
        history = get_session_history()

        # Get last few queries for context
        recent_queries = []
        for entry in history[-3:]:  # Last 3 queries
            recent_queries.append({
                'query': entry.get('query', ''),
                'timestamp': entry.get('timestamp', ''),
                'was_follow_up': entry.get('is_follow_up', False)
            })

        return jsonify({
            'success': True,
            'current_count': len(history),
            'max_follow_ups': 10,
            'remaining': max(0, 10 - len(history)),
            'at_limit': len(history) >= 10,
            'recent_queries': recent_queries,
            'has_context': len(history) > 0
        })
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


###### template specific routes
@app.route('/api/templates', methods=['GET'])
def get_available_templates():
    """Get all available templates"""
    try:
        templates = template_manager.get_all_templates()
        template_list = []

        for template in templates:
            template_list.append({
                'id': template.id,
                'name': template.name,
                'description': template.description,
                'domain': template.domain,
                'version': template.version
            })

        return jsonify({
            'success': True,
            'templates': template_list,
            'count': len(template_list)
        })

    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/templates/suggestions', methods=['POST'])
def get_template_suggestions():
    """Get template suggestions for current data"""
    try:
        current_data = get_current_data()
        if current_data is None:
            return jsonify({
                'success': False,
                'error': 'No data loaded'
            }), 400

        filename = get_current_filename()
        worksheet_data_for_suggestions = get_worksheet_data()
        suggestions = get_template_suggestions_for_data(worksheet_data_for_suggestions, filename)

        return jsonify({
            'success': True,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"Error getting template suggestions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/templates/apply', methods=['POST'])
def apply_template():
    """Manually apply a template to current data"""
    try:
        data = request.get_json()
        template_id = data.get('template_id')

        if not template_id:
            return jsonify({
                'success': False,
                'error': 'Template ID required'
            }), 400

        current_data = get_current_data()
        if current_data is None:
            return jsonify({
                'success': False,
                'error': 'No data loaded'
            }), 400

        # Apply template (multi-worksheet aware)
        worksheet_data_for_template = get_worksheet_data()
        enhanced_summary, template = apply_template_manually(worksheet_data_for_template, template_id)

        # Update session
        session_obj = session_manager.get_current_session()
        session_obj.detected_template = template
        session_obj.manual_template_override = template_id

        # Validate data against template (multi-worksheet aware)
        worksheet_data_for_validation = get_worksheet_data()
        validation_results = validate_data_against_template(worksheet_data_for_validation, template)
        session_obj.template_validation_results = validation_results

        # Update data summary using enhanced analysis
        enhanced_summary, _ = analyze_data_with_template_support(worksheet_data_for_validation, get_current_filename())

        # Update data summary
        set_data_summary(enhanced_summary)

        # Update shared state
        shared_state.update_data(
            current_data,
            get_current_filename(),
            enhanced_summary,
            get_current_document_key(),
            get_current_document_url()
        )

        return jsonify({
            'success': True,
            'template': {
                'id': template.id,
                'name': template.name,
                'domain': template.domain
            },
            'validation': validation_results,
            'message': f'Template "{template.name}" applied successfully'
        })

    except Exception as e:
        logger.error(f"Error applying template: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/templates/worksheet_mapping', methods=['GET'])
def get_worksheet_mapping():
    """Get worksheet mapping information for current template"""
    try:
        session_obj = session_manager.get_current_session()

        if not session_obj.detected_template:
            return jsonify({
                'success': False,
                'error': 'No template currently applied'
            }), 400

        if session_obj.detected_template.template_type not in ['multi_worksheet', 'csv_collection']:
            return jsonify({
                'success': True,
                'is_multi_worksheet': False,
                'message': 'Current template is single worksheet'
            })

        validation_results = session_obj.template_validation_results or {}
        worksheet_mapping = validation_results.get('worksheet_validations', [])

        return jsonify({
            'success': True,
            'is_multi_worksheet': True,
            'template_type': session_obj.detected_template.template_type,
            'worksheet_mapping': worksheet_mapping,
            'summary': validation_results.get('summary', {}),
            'template_info': {
                'name': session_obj.detected_template.name,
                'total_worksheets': len(session_obj.detected_template.worksheets),
                'required_worksheets': sum(1 for ws in session_obj.detected_template.worksheets if ws.is_required)
            }
        })

    except Exception as e:
        logger.error(f"Error getting worksheet mapping: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/templates/current', methods=['GET'])
def get_current_template():
    """Get currently applied template information"""
    try:
        session_obj = session_manager.get_current_session()

        if not session_obj.detected_template:
            return jsonify({
                'success': True,
                'template': None,
                'message': 'No template currently applied'
            })

        template = session_obj.detected_template
        validation = session_obj.template_validation_results or {}

        return jsonify({
            'success': True,
            'template': {
                'id': template.id,
                'name': template.name,
                'description': template.description,
                'domain': template.domain,
                'version': template.version,
                'manually_applied': session_obj.manual_template_override is not None
            },
            'validation': validation
        })

    except Exception as e:
        logger.error(f"Error getting current template: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/templates/remove', methods=['POST'])
def remove_template():
    """Remove currently applied template"""
    try:
        session_obj = session_manager.get_current_session()

        if session_obj.detected_template:
            template_name = session_obj.detected_template.name
            session_obj.detected_template = None
            session_obj.manual_template_override = None
            session_obj.template_validation_results = None

            # Regenerate standard data summary
            current_data = get_current_data()
            if current_data is not None:
                standard_summary = analyze_data_standard(current_data)
                set_data_summary(standard_summary)

            return jsonify({
                'success': True,
                'message': f'Template "{template_name}" removed. Using standard analysis.'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No template was applied'
            })

    except Exception as e:
        logger.error(f"Error removing template: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear_chat', methods=['POST'])
def clear_chat_history():
    """Clear chat and session history while keeping data loaded"""
    try:
        clear_session_history()
        print("Session history cleared")
        return jsonify({'success': True, 'message': 'Chat and session history cleared successfully'})
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': f'Error clearing chat history: {str(e)}'}), 500

@app.route('/download', methods=['GET'])
def download_spreadsheet():
    """Download the current spreadsheet with analysis results"""
    current_data = get_current_data()
    current_filename = get_current_filename()
    worksheet_data = get_worksheet_data()
    
    if current_data is None:
        return jsonify({'error': 'No data to download'}), 400
    
    try:
        # Create a new Excel file with current data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(current_filename)[0] if current_filename else "analysis"
        download_filename = f"{base_name}_analyzed_{timestamp}.xlsx"
        download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], download_filename)
        
        # Create Excel writer
        with pd.ExcelWriter(download_path, engine='openpyxl') as writer:
            if worksheet_data and len(worksheet_data) > 1:
                # Multi-worksheet file
                for sheet_name, sheet_data in worksheet_data.items():
                    sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Single worksheet
                current_data.to_excel(writer, sheet_name='Sheet1', index=False)

        # Associate download file with session
        session_id = session_manager.get_current_session_id()
        file_cleanup_manager.associate_file_with_session(download_path, session_id, 'download')
        
        if os.path.exists(download_path):
            return send_file(
                download_path,
                as_attachment=True,
                download_name=download_filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            return jsonify({'error': 'Error creating download file'}), 500
            
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': f'Error preparing download: {str(e)}'}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset all session data for current user"""
    shared_state.reset()  # This is now session-aware
    reset_current_session()  # Reset all session data including history

    # Clear template data from session
    try:
        session_obj = session_manager.get_current_session()
        session_obj.detected_template = None
        session_obj.template_validation_results = None
        session_obj.manual_template_override = None
        logger.info("Template data cleared during session reset")
    except Exception as e:
        logger.warning(f"Error clearing template data during reset: {e}")
    
    print("Full session reset including session history")
    return jsonify({'success': True, 'message': 'Session reset successfully'})

@app.route('/files/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Session monitoring endpoint
@app.route('/admin/sessions')
def session_status():
    """Admin endpoint to monitor active sessions"""
    return jsonify({
        'active_sessions': session_manager.get_session_count(),
        'current_session_id': session_manager.get_current_session_id()
    })

# cleanup connection
def cleanup_on_shutdown():
    """Cleanup function for graceful shutdown"""
    logger.info("Application shutting down, cleaning up resources...")
    try:
        # Import cleanup function from agent module
        from agent import cleanup_connections
        cleanup_connections()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Graceful shutdown handler
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_on_shutdown()
    exit(0)

# Register signal handlers and cleanup function
# Only register signal handlers in the main thread
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_shutdown)
    logger.info("Signal handlers registered successfully")
else:
    logger.info("Skipping signal handler registration (not in main thread)")

# endpoint for health check of token
@app.route('/health/token', methods=['GET'])
def check_token_health():
    """Health check endpoint for token status"""
    try:
        from agent import token_manager
        if token_manager:
            token = token_manager.get_valid_token()
            return jsonify({
                'token_valid': bool(token),
                'expires_at': token_manager.token_expires_at.isoformat() if token_manager.token_expires_at else None,
                'status': 'healthy'
            })
        else:
            return jsonify({'status': 'token_manager_not_initialized'}), 500
    except Exception as e:
        logger.error(f"Token health check failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Register database routes & additional routes for db config management & healthcheck
try:
    register_database_routes(app)
    # ENHANCED: Register supplemental context routes
    register_supplemental_context_routes(app)
    logger.info("Database routes registered successfully")
except Exception as e:
    logger.error(f"Failed to register database routes: {e}")
    print(f"Warning: Database functionality may not be available: {e}")

# route for database configuration management (optional - for future use)
@app.route('/api/admin/database_configs', methods=['GET', 'POST'])
def manage_database_configs():
    """Admin endpoint for managing database configurations"""
    try:
        config_manager = get_config_manager()

        if request.method == 'GET':
            # Return all configurations (without sensitive data)
            configs = config_manager.get_safe_configs(enabled_only=False)
            return jsonify({
                'success': True,
                'configs': configs,
                'count': len(configs)
            })

        elif request.method == 'POST':
            # Add or update configuration
            data = request.get_json()
            action = data.get('action', 'add')

            if action == 'add':
                from database_config import DatabaseConfig
                config = DatabaseConfig.from_dict(data.get('config', {}))

                # Validate configuration
                is_valid, validation_msg = config_manager.validate_config(config)
                if not is_valid:
                    return jsonify({
                        'success': False,
                        'error': validation_msg
                    }), 400

                config_manager.add_config(config)
                return jsonify({
                    'success': True,
                    'message': f'Configuration {config.name} added successfully'
                })

            elif action == 'test':
                config_id = data.get('config_id')
                config = config_manager.get_config(config_id)
                if not config:
                    return jsonify({
                        'success': False,
                        'error': f'Configuration {config_id} not found'
                    }), 404

                success, message = config_manager.test_config(config)
                return jsonify({
                    'success': success,
                    'message': message
                })

            else:
                return jsonify({
                    'success': False,
                    'error': f'Unknown action: {action}'
                }), 400

    except Exception as e:
        logger.error(f"Error managing database configs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# health check endpoint for database functionality
@app.route('/health/database', methods=['GET'])
def check_database_health():
    """Health check endpoint for database functionality"""
    try:
        config_manager = get_config_manager()
        configs = config_manager.get_all_configs(enabled_only=True)

        # Test one configuration if available
        test_results = []
        if configs:
            for config in configs[:1]:  # Test only the first config
                success, message = config_manager.test_config(config)
                test_results.append({
                    'config_id': config.id,
                    'config_name': config.name,
                    'success': success,
                    'message': message
                })

        return jsonify({
            'status': 'healthy',
            'database_mode_available': True,
            'configurations_count': len(configs),
            'test_results': test_results
        })

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return jsonify({
            'status': 'error',
            'database_mode_available': False,
            'error': str(e)
        }), 500

# Register template creation and management routes
try:
    register_template_management_routes(app)
    logger.info("Template creation routes registered successfully")
except Exception as e:
    logger.error(f"Failed to register template creation routes: {e}")
    print(f"Warning: Template creation functionality may not be available: {e}")

# File cleanup admin routes
@app.route('/admin/cleanup/status', methods=['GET'])
def cleanup_status():
    """Admin endpoint to check cleanup status"""
    try:
        stats = file_cleanup_manager.get_cleanup_stats()
        disk_info = file_cleanup_manager.check_disk_space()

        return jsonify({
            'success': True,
            'stats': stats,
            'disk_info': disk_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/admin/cleanup/force', methods=['POST'])
def force_cleanup():
    """Admin endpoint to force cleanup"""
    try:
        data = request.get_json() or {}
        max_age_hours = data.get('max_age_hours', 24)

        deleted_count, size_freed = file_cleanup_manager.cleanup_old_files(max_age_hours)

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'size_freed_mb': size_freed / (1024 * 1024),
            'message': f'Cleaned up {deleted_count} files'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/admin/cleanup/emergency', methods=['POST'])
def emergency_cleanup():
    """Admin endpoint for emergency cleanup"""
    try:
        file_cleanup_manager.emergency_cleanup()
        return jsonify({
            'success': True,
            'message': 'Emergency cleanup completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get debug mode from environment variable, default to False for production-like behavior
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Print startup information
    print("Flask Excel Analytics App with OnlyOffice Integration Starting...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Download folder: {app.config['DOWNLOAD_FOLDER']}")
    print(f"Documents folder: {app.config['DOCUMENTS_FOLDER']}")
    print(f"OnlyOffice server URL: {ONLYOFFICE_SERVER_URL}")
    print(f"Debug mode: {debug_mode}")
    print("Session-aware interface with enhanced output display and per-user context!")
    print(f"Session management: Active sessions will be tracked and cleaned up automatically")
    
    if debug_mode:
        print("Debug mode is ON - auto-reload enabled")
        print("To disable auto-restart, set FLASK_DEBUG=False in your .env file")
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
    else:
        print("Production mode - auto-reload disabled")
        print("To enable debug mode, set FLASK_DEBUG=True in your .env file")
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
