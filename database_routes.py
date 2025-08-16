"""
Database-specific Flask routes for the Excel Analytics Platform
ENHANCED with supplemental context integration and improved Text-to-SQL error correction
"""

from flask import request, render_template, jsonify, send_file, current_app
from shared_state import shared_state
import os
import pandas as pd
import numpy as np
import tempfile
import uuid
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

# ENHANCED: Import database-related modules with supplemental context support
from database_config import get_config_manager, DatabaseConfig
from databricks_handler_dbapi import EnhancedDatabricksHandler  # ENHANCED: Use enhanced handler
from enhanced_text2sql_agent import create_enhanced_text2sql_agent, SQLGenerationResult
from supplemental_context_manager import get_context_manager  # ENHANCED: Import context manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_onlyoffice_document(file_path, original_filename):
    """Create OnlyOffice document - copied from app.py to avoid circular import"""
    try:
        # Generate unique document key
        document_key = str(uuid.uuid4())

        # Get app config from current Flask app
        documents_dir = current_app.config['DOCUMENTS_FOLDER']
        os.makedirs(documents_dir, exist_ok=True, mode=0o755)

        # Determine file extension
        file_ext = 'xlsx'  # default to xlsx
        if original_filename and '.' in original_filename:
            extracted_ext = original_filename.split('.')[-1].lower()
            valid_extensions = ['xlsx', 'xls', 'csv', 'ods', 'xlsm', 'xlsb']
            if extracted_ext in valid_extensions:
                file_ext = extracted_ext

        # Create the document filename with the document key
        document_filename = f"{document_key}.{file_ext}"
        document_path = os.path.join(documents_dir, document_filename)

        if file_ext == 'csv':
            # Convert CSV to Excel for better OnlyOffice compatibility
            import pandas as pd
            df = pd.read_csv(file_path)
            document_filename = f"{document_key}.xlsx"
            document_path = os.path.join(documents_dir, document_filename)
            df.to_excel(document_path, index=False, sheet_name='Sheet1')
        else:
            # Copy Excel file to documents folder
            shutil.copy2(file_path, document_path)

        # Set proper file permissions
        try:
            os.chmod(document_path, 0o644)
        except Exception as perm_error:
            print(f"Warning: Could not set file permissions: {perm_error}")

        # Create document URL (served by Flask)
        document_url = f"{request.host_url}documents/{document_filename}"

        return document_url, document_key

    except Exception as e:
        print(f"Error creating OnlyOffice document: {e}")
        raise e

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


def analyze_data(df):
    """Analyze data - copied from app.py to avoid circular import"""

    # Convert dtypes to string representation
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
    """
    def convert_pandas_types(obj):
        if isinstance(obj, str):
            return obj
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        if hasattr(obj, 'dtype'):
            if pd.api.types.is_integer_dtype(obj.dtype):
                return int(obj) if pd.notna(obj) else None
            elif pd.api.types.is_float_dtype(obj.dtype):
                return float(obj) if pd.notna(obj) else None
            elif pd.api.types.is_bool_dtype(obj.dtype):
                return bool(obj) if pd.notna(obj) else None
            else:
                return str(obj) if pd.notna(obj) else None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    """

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
    if summary['numeric_columns']:
        numeric_stats = {}
        try:
            stats_df = df[summary['numeric_columns']].describe()
            for col in stats_df.columns:
                numeric_stats[str(col)] = {}
                for stat in stats_df.index:
                    numeric_stats[col][stat] = convert_pandas_types(stats_df.loc[stat, col])
            summary['numeric_stats'] = numeric_stats
        except Exception as e:
            print(f"Warning: Could not generate numeric stats: {e}")
            summary['numeric_stats'] = {}

    # Add value counts for categorical columns
    summary['categorical_stats'] = {}
    for col in summary['categorical_columns'][:20]:  # Limit to first 20 categorical columns
        try:
            value_counts = df[col].value_counts().head()
            summary['categorical_stats'][str(col)] = {
                    str(k): convert_pandas_types(v) for k, v in value_counts.to_dict().items()
            }
        except Exception as e:
            print(f"Warning: Could not generate categorical stats for {col}: {e}")
            summary['categorical_stats'][str(col)] = {}

    return summary


def register_database_routes(app):
    """Register database-related routes with the Flask app"""
    
    @app.route('/database')
    def database_mode():
        """Database mode main page"""
        try:
            return render_template('database.html')
        except Exception as e:
            logger.error(f"Error loading database page: {e}")
            return f"Error loading database page: {e}", 500
    
    @app.route('/api/database/configs')
    def get_database_configs():
        """Get available database configurations with supplemental context info"""
        try:
            config_manager = get_config_manager()
            configs = config_manager.get_safe_configs(enabled_only=True)
            
            # ENHANCED: Add supplemental context information to each config
            context_manager = get_context_manager()
            enhanced_configs = []
            
            for config in configs:
                enhanced_config = config.copy()
                
                # Check if supplemental context exists
                context_summary = context_manager.get_context_summary(config['id'])
                if context_summary:
                    enhanced_config['has_supplemental_context'] = True
                    enhanced_config['context_summary'] = {
                        'has_database_description': context_summary.get('has_database_description', False),
                        'has_global_instructions': context_summary.get('has_global_instructions', False),
                        'schema_architecture_type': context_summary.get('schema_architecture', {}).get('schema_type'),
                        'hierarchy_count': len(context_summary.get('hierarchy_definitions_summary', {})),
                        'glossary_terms_count': len(context_summary.get('business_glossary_terms', [])),
                        'kpi_count': len(context_summary.get('kpi_definitions', [])),
                        'table_metadata_count': len(context_summary.get('table_metadata', []))
                    }
                else:
                    enhanced_config['has_supplemental_context'] = False
                    enhanced_config['context_summary'] = {}
                
                enhanced_configs.append(enhanced_config)
            
            return jsonify({
                'success': True,
                'configs': enhanced_configs,
                'count': len(enhanced_configs)
            })
            
        except Exception as e:
            logger.error(f"Error getting database configs: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'configs': []
            }), 500
    
    @app.route('/api/database/test_connection', methods=['POST'])
    def test_database_connection():
        """Test database connection with enhanced handler"""
        try:
            data = request.get_json()
            config_id = data.get('config_id')
            
            if not config_id:
                return jsonify({
                    'success': False,
                    'error': 'No configuration ID provided'
                }), 400
            
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({
                    'success': False,
                    'error': f'Configuration {config_id} not found'
                }), 404
            
            # ENHANCED: Test connection using enhanced handler with context support
            success, message = config_manager.test_config(config)
            
            # ENHANCED: Add supplemental context status to response
            context_manager = get_context_manager()
            context_summary = context_manager.get_context_summary(config_id)
            
            response_data = {
                'success': success,
                'message': message,
                'config_id': config_id
            }
            
            if context_summary:
                response_data['supplemental_context'] = {
                    'available': True,
                    'summary': context_summary
                }
            else:
                response_data['supplemental_context'] = {
                    'available': False,
                    'message': 'No supplemental context configured. Consider adding business context for better query results.'
                }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error testing database connection: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/database/schema_info')
    def get_database_schema():
        """Get database schema information with enhanced context integration"""
        try:
            config_id = request.args.get('config_id')
            
            if not config_id:
                return jsonify({
                    'success': False,
                    'error': 'No configuration ID provided'
                }), 400
            
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({
                    'success': False,
                    'error': f'Configuration {config_id} not found'
                }), 404
            
            # ENHANCED: Create handler with context support
            handler = EnhancedDatabricksHandler(
                server_hostname=config.server_hostname,
                http_path=config.http_path,
                catalog=config.catalog,
                schema=config.schema,
                config_id=config.id  # ENHANCED: Pass config_id for supplemental context
            )
            
            # Set environment variables temporarily for authentication
            original_env = {}
            env_vars = {
                'AZURE_TENANT_ID': config.azure_tenant_id,
                'AZURE_CLIENT_ID': config.azure_client_id,
                'AZURE_CLIENT_SECRET': config.azure_client_secret
            }
            
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Get schema information
                catalogs = handler.get_catalogs()
                schemas = handler.get_schemas(config.catalog)
                tables = handler.get_tables(config.catalog, config.schema)
                
                # ENHANCED: Get database context (now includes supplemental context automatically)
                db_context = handler.get_database_context(
                    config.catalog, 
                    config.schema, 
                    include_sample_data=True
                )
                
                # ENHANCED: Get supplemental context summary
                context_manager = get_context_manager()
                context_summary = context_manager.get_context_summary(config_id)
                
                response_data = {
                    'success': True,
                    'catalog': config.catalog,
                    'schema': config.schema,
                    'catalogs': catalogs,
                    'schemas': schemas,
                    'tables': tables,
                    'table_count': len(tables),
                    'database_context': db_context
                }
                
                # Add supplemental context information
                if context_summary:
                    response_data['supplemental_context'] = {
                        'available': True,
                        'enhanced': True,
                        'summary': context_summary,
                        'message': 'Database context enhanced with business information'
                    }
                else:
                    response_data['supplemental_context'] = {
                        'available': False,
                        'enhanced': False,
                        'message': 'Using Unity Catalog schema only. Add supplemental context for better results.'
                    }
                
                return jsonify(response_data)
                
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                
                handler.close_connection()
                
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/database/text2sql', methods=['POST'])
    def text_to_sql():
        """Convert natural language question to SQL with enhanced context and retry capabilities"""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            config_id = data.get('config_id')
            
            if not question:
                return jsonify({
                    'success': False,
                    'error': 'No question provided'
                }), 400
            
            if not config_id:
                return jsonify({
                    'success': False,
                    'error': 'No database configuration provided'
                }), 400
            
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({
                    'success': False,
                    'error': f'Configuration {config_id} not found'
                }), 404
            
            # Create Enhanced Text-to-SQL agent with retry capabilities
            text2sql_agent = create_enhanced_text2sql_agent(max_retry_attempts=3)
            
            # ENHANCED: Create handler with context support
            handler = EnhancedDatabricksHandler(
                server_hostname=config.server_hostname,
                http_path=config.http_path,
                catalog=config.catalog,
                schema=config.schema,
                config_id=config.id  # ENHANCED: Pass config_id for supplemental context
            )
            
            # Set environment variables temporarily
            original_env = {}
            env_vars = {
                'AZURE_TENANT_ID': config.azure_tenant_id,
                'AZURE_CLIENT_ID': config.azure_client_id,
                'AZURE_CLIENT_SECRET': config.azure_client_secret
            }
            
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # ENHANCED: Get database context (now automatically includes supplemental context)
                db_context = handler.get_database_context(
                    config.catalog, 
                    config.schema, 
                    include_sample_data=True
                )
                
                # Get previous queries for context
                previous_queries = text2sql_agent.get_query_history()
                
                # Use enhanced generation with retry and error correction
                result = text2sql_agent.generate_sql_with_retry(
                    question, 
                    db_context, 
                    previous_queries,
                    database_handler=handler  # Pass handler for execution testing
                )
                
                # ENHANCED: Get supplemental context info for response
                context_manager = get_context_manager()
                context_summary = context_manager.get_context_summary(config_id)
                
                if result.success:
                    response_data = {
                        'success': True,
                        'sql_query': result.final_sql,
                        'explanation': result.explanation,
                        'is_valid': True,
                        'validation_message': "Query generated and validated successfully",
                        'config_id': config_id,
                        'attempts_made': result.total_attempts,
                        'retry_details': [
                            {
                                'attempt': attempt.attempt_number,
                                'success': attempt.success,
                                'error_type': attempt.error_type.value if attempt.error_type else None,
                                'error_message': attempt.error_message if not attempt.success else None
                            }
                            for attempt in result.attempts
                        ]
                    }
                    
                    # Add context enhancement info
                    if context_summary:
                        response_data['context_enhanced'] = True
                        response_data['context_info'] = f"Enhanced with {len(context_summary.get('business_glossary_terms', []))} business terms, {len(context_summary.get('kpi_definitions', []))} KPIs, and {len(context_summary.get('table_metadata', []))} enhanced tables"
                    else:
                        response_data['context_enhanced'] = False
                        response_data['context_info'] = "Using Unity Catalog schema only"
                    
                    return jsonify(response_data)
                else:
                    # Get enhanced suggestions from the agent
                    suggestions = text2sql_agent.suggest_query_improvements("", result.final_error)
                    
                    response_data = {
                        'success': False,
                        'error': result.final_error,
                        'sql_query': '',
                        'config_id': config_id,
                        'attempts_made': result.total_attempts,
                        'suggestions': suggestions,
                        'retry_details': [
                            {
                                'attempt': attempt.attempt_number,
                                'success': attempt.success,
                                'error_type': attempt.error_type.value if attempt.error_type else None,
                                'error_message': attempt.error_message,
                                'sql_attempted': attempt.sql_query
                            }
                            for attempt in result.attempts
                        ]
                    }
                    
                    # Add context suggestion if no supplemental context exists
                    if not context_summary:
                        response_data['context_suggestion'] = "Consider adding supplemental business context for better query generation accuracy"
                    
                    return jsonify(response_data)
                    
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                
                handler.close_connection()
                
        except Exception as e:
            logger.error(f"Error in enhanced text-to-SQL conversion: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/database/execute_sql', methods=['POST'])
    def execute_sql_query():
        """Execute SQL query and return results with enhanced context integration"""
        try:
            data = request.get_json()
            sql_query = data.get('sql_query', '').strip()
            config_id = data.get('config_id')
            question = data.get('question', '')
            max_rows = data.get('max_rows', 100000)
            
            if not sql_query:
                return jsonify({
                    'success': False,
                    'error': 'No SQL query provided'
                }), 400
            
            if not config_id:
                return jsonify({
                    'success': False,
                    'error': 'No database configuration provided'
                }), 400
            
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({
                    'success': False,
                    'error': f'Configuration {config_id} not found'
                }), 404
            
            # ENHANCED: Create handler with context support
            handler = EnhancedDatabricksHandler(
                server_hostname=config.server_hostname,
                http_path=config.http_path,
                catalog=config.catalog,
                schema=config.schema,
                config_id=config.id  # ENHANCED: Pass config_id for supplemental context
            )
            
            # Set environment variables temporarily
            original_env = {}
            env_vars = {
                'AZURE_TENANT_ID': config.azure_tenant_id,
                'AZURE_CLIENT_ID': config.azure_client_id,
                'AZURE_CLIENT_SECRET': config.azure_client_secret
            }
            
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Execute query
                success, df, error = handler.execute_query(sql_query, max_rows)
                
                if success and df is not None:
                    # Save result to CSV for OnlyOffice integration
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    query_description = question[:50] if question else "database_query"
                    safe_description = "".join(c for c in query_description if c.isalnum() or c in (' ', '_', '-')).strip()
                    
                    csv_filename = f"db_result_{safe_description}_{timestamp}.csv"
                    csv_path = os.path.join(current_app.config['UPLOAD_FOLDER'], csv_filename)
                    
                    # Save to CSV
                    df.to_csv(csv_path, index=False)
                    
                    # Create OnlyOffice document
                    from app import create_onlyoffice_document  # Import here to avoid circular import
                    document_url, document_key = create_onlyoffice_document(csv_path, csv_filename)
                    
                    # Update global variables (this follows the existing pattern from app.py)
                    import app
                    app.current_data = df
                    app.current_filename = csv_filename
                    app.data_summary = app.analyze_data(df)
                    app.worksheet_data = {'Sheet1': df}
                    app.worksheet_summaries = {'Sheet1': app.data_summary}
                    app.active_worksheet = 'Sheet1'
                    app.current_document_key = document_key
                    app.current_document_url = document_url
                    
                    # ENHANCED: Get supplemental context info for response
                    context_manager = get_context_manager()
                    context_summary = context_manager.get_context_summary(config_id)
                    
                    response_data = {
                        'success': True,
                        'filename': csv_filename,
                        'shape': list(df.shape),
                        'columns': list(df.columns),
                        'summary': app.data_summary,
                        'document_url': document_url,
                        'document_key': document_key,
                        'has_multiple_worksheets': False,
                        'active_worksheet': 'Sheet1',
                        'query_executed': sql_query,
                        'rows_returned': len(df)
                    }
                    
                    # Add context enhancement info
                    if context_summary:
                        response_data['context_enhanced'] = True
                        response_data['context_info'] = "Query executed with enhanced business context"
                    else:
                        response_data['context_enhanced'] = False
                        response_data['context_info'] = "Query executed with Unity Catalog schema only"
                    
                    return jsonify(response_data)
                    
                else:
                    # Use enhanced Text2SQL agent for error analysis and suggestions
                    text2sql_agent = create_enhanced_text2sql_agent()
                    suggestions = text2sql_agent.suggest_query_improvements(sql_query, error)
                    
                    response_data = {
                        'success': False,
                        'error': error or 'Query execution failed',
                        'suggestions': suggestions,
                        'sql_query': sql_query
                    }
                    
                    # ENHANCED: Add context suggestion if no supplemental context exists
                    context_manager = get_context_manager()
                    context_summary = context_manager.get_context_summary(config_id)
                    if not context_summary:
                        response_data['context_suggestion'] = "Consider adding supplemental business context to help with query disambiguation and error prevention"
                    
                    return jsonify(response_data)
                    
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                
                handler.close_connection()
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/database/query', methods=['POST'])
    def database_query():
        """
        ENHANCED two-step database query processing with supplemental context integration:
        Step 1: Fetch data from database via enhanced SQL generation with retry and context
        Step 2: Call data analysis agent with the fetched data
        """
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            config_id = data.get('config_id')

            if not question:
                return jsonify({
                    'success': False,
                    'error': 'No question provided',
                    'step': 'validation'
                }), 400

            if not config_id:
                return jsonify({
                    'success': False,
                    'error': 'No database configuration provided',
                    'step': 'validation'
                }), 400

            logger.info(f"Starting ENHANCED two-step processing for question: '{question[:100]}...'")

            # Get database configuration
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)

            if not config:
                return jsonify({
                    'success': False,
                    'error': f'Configuration {config_id} not found',
                    'step': 'configuration'
                }), 404

            # ENHANCED STEP 1: FETCH DATA FROM DATABASE WITH CONTEXT AND RETRY
            logger.info("=== ENHANCED STEP 1: Fetching data from database with supplemental context and retry capabilities ===")

            step1_result = execute_enhanced_step1_fetch_data(question, config)

            if not step1_result['success']:
                return jsonify({
                    'success': False,
                    'error': step1_result['error'],
                    'sql_query': step1_result.get('sql_query', ''),
                    'suggestions': step1_result.get('suggestions', ''),
                    'step': 'step1_sql_execution',
                    'retry_details': step1_result.get('retry_details', []),
                    'attempts_made': step1_result.get('attempts_made', 0),
                    'context_enhanced': step1_result.get('context_enhanced', False),
                    'context_info': step1_result.get('context_info', '')
                })

            # STEP 2: ANALYZE DATA WITH AI AGENT
            logger.info("=== STEP 2: Analyzing data with AI agent ===")

            step2_result = execute_step2_analyze_data(question, step1_result['data'])

            # Combine results from both steps with enhanced context information
            combined_result = {
                'success': True,
                'step1_result': {
                    'sql_query': step1_result['sql_query'],
                    'explanation': step1_result['explanation'],
                    'filename': step1_result['filename'],
                    'shape': step1_result['shape'],
                    'columns': step1_result['columns'],
                    'document_url': step1_result['document_url'],
                    'document_key': step1_result['document_key'],
                    'rows_returned': step1_result['rows_returned'],
                    'attempts_made': step1_result.get('attempts_made', 1),
                    'retry_details': step1_result.get('retry_details', []),
                    'context_enhanced': step1_result.get('context_enhanced', False),
                    'context_info': step1_result.get('context_info', '')
                },
                'step2_result': step2_result,
                'message': f'✅ Enhanced two-step processing completed! Fetched {step1_result["rows_returned"]} rows in {step1_result.get("attempts_made", 1)} attempt(s) and analyzed the data.',
                'data_source': 'enhanced_two_step_process_with_context'
            }

            logger.info("Enhanced two-step processing with supplemental context completed successfully")
            combined_result['step2_result'] = remove_plot_objects_from_result(combined_result['step2_result'])
            combined_result = convert_pandas_types(combined_result)

            return jsonify(combined_result)

        except Exception as e:
            logger.error(f"Critical error in enhanced two-step database query: {e}")
            return jsonify({
                'success': False,
                'error': f'Critical system error: {str(e)}',
                'suggestion': 'Please try again or contact support',
                'step': 'system_error'
            }), 500

    def execute_enhanced_step1_fetch_data(question: str, config) -> dict:
        """
        ENHANCED Step 1: Generate SQL with supplemental context, retry, and fetch data from database

        Returns: Dictionary with success, data, error, retry details, context info, and metadata
        """
        try:
            # Create Enhanced Text-to-SQL agent with retry capabilities
            text2sql_agent = create_enhanced_text2sql_agent(max_retry_attempts=3)

            # ENHANCED: Create handler with context support
            handler = EnhancedDatabricksHandler(
                server_hostname=config.server_hostname,
                http_path=config.http_path,
                catalog=config.catalog,
                schema=config.schema,
                config_id=config.id  # ENHANCED: Pass config_id for supplemental context
            )

            # Set environment variables temporarily
            original_env = {}
            env_vars = {
                'AZURE_TENANT_ID': config.azure_tenant_id,
                'AZURE_CLIENT_ID': config.azure_client_id,
                'AZURE_CLIENT_SECRET': config.azure_client_secret
            }

            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            try:
                # ENHANCED: Get database context (now automatically includes supplemental context)
                db_context = handler.get_database_context(
                    config.catalog,
                    config.schema,
                    include_sample_data=True
                )

                # ENHANCED: Get supplemental context info
                context_manager = get_context_manager()
                context_summary = context_manager.get_context_summary(config.id)
                
                # Get previous queries for context
                previous_queries = text2sql_agent.get_query_history()

                # Use enhanced generation with retry and automatic error correction
                logger.info("Generating SQL with enhanced context and retry mechanism...")
                result = text2sql_agent.generate_sql_with_retry(
                    question,
                    db_context,
                    previous_queries,
                    database_handler=handler  # Pass handler for execution testing during generation
                )

                if not result.success:
                    # Enhanced error reporting with retry details and context info
                    suggestions = text2sql_agent.suggest_query_improvements("", result.final_error)
                    
                    response_data = {
                        'success': False,
                        'error': f'Enhanced SQL generation failed after {result.total_attempts} attempts: {result.final_error}',
                        'sql_query': '',
                        'suggestions': suggestions,
                        'attempts_made': result.total_attempts,
                        'retry_details': [
                            {
                                'attempt': attempt.attempt_number,
                                'success': attempt.success,
                                'error_type': attempt.error_type.value if attempt.error_type else None,
                                'error_message': attempt.error_message,
                                'sql_attempted': attempt.sql_query
                            }
                            for attempt in result.attempts
                        ]
                    }
                    
                    # Add context information
                    if context_summary:
                        response_data['context_enhanced'] = True
                        response_data['context_info'] = f"Enhanced with business context, but SQL generation still failed"
                        response_data['context_suggestion'] = "Consider reviewing or expanding your supplemental context definitions"
                    else:
                        response_data['context_enhanced'] = False
                        response_data['context_info'] = "No supplemental context available"
                        response_data['context_suggestion'] = "Add supplemental business context to improve SQL generation accuracy"
                    
                    return response_data

                # Success! The retry mechanism has already validated and tested the query
                final_sql = result.final_sql

                # Execute the final query with full row limit for actual data retrieval
                logger.info(f"Executing validated SQL query (attempt {result.total_attempts} succeeded with context)")
                execution_success, df, execution_error = handler.execute_query(final_sql, 100000)

                if not execution_success or df is None:
                    # This shouldn't happen since the retry mechanism already tested it, but handle just in case
                    suggestions = text2sql_agent.suggest_query_improvements(final_sql, execution_error)
                    
                    response_data = {
                        'success': False,
                        'error': f'Final execution failed despite successful retry validation: {execution_error}',
                        'sql_query': final_sql,
                        'suggestions': suggestions,
                        'attempts_made': result.total_attempts,
                        'retry_details': [
                            {
                                'attempt': attempt.attempt_number,
                                'success': attempt.success,
                                'error_type': attempt.error_type.value if attempt.error_type else None,
                                'error_message': attempt.error_message,
                                'sql_attempted': attempt.sql_query
                            }
                            for attempt in result.attempts
                        ]
                    }
                    
                    # Add context information
                    if context_summary:
                        response_data['context_enhanced'] = True
                        response_data['context_info'] = "Enhanced with business context, but final execution failed"
                    else:
                        response_data['context_enhanced'] = False
                        response_data['context_info'] = "No supplemental context available"
                    
                    return response_data

                # Save result and create OnlyOffice document
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_question = "".join(c for c in question[:50] if c.isalnum() or c in (' ', '_', '-')).strip()
                csv_filename = f"db_result_{safe_question}_{timestamp}.csv"
                csv_path = os.path.join(current_app.config['UPLOAD_FOLDER'], csv_filename)

                df.to_csv(csv_path, index=False)

                # Create OnlyOffice document
                document_url, document_key = create_onlyoffice_document(csv_path, csv_filename)

                # Analyze data
                data_summary = analyze_data(df)

                # Update shared state with new data
                shared_state.update_data(df, csv_filename, data_summary, document_key, document_url)

                # Log success with context and retry information
                context_message = ""
                if context_summary:
                    context_parts = []
                    if context_summary.get('has_database_description'):
                        context_parts.append("database description")
                    if context_summary.get('business_glossary_terms'):
                        context_parts.append(f"{len(context_summary['business_glossary_terms'])} business terms")
                    if context_summary.get('kpi_definitions'):
                        context_parts.append(f"{len(context_summary['kpi_definitions'])} KPIs")
                    if context_summary.get('table_metadata'):
                        context_parts.append(f"{len(context_summary['table_metadata'])} enhanced tables")
                    
                    if context_parts:
                        context_message = f" with {', '.join(context_parts)}"

                if result.total_attempts > 1:
                    logger.info(f"✅ Enhanced Step 1 succeeded after {result.total_attempts} attempts with error correction{context_message}")
                else:
                    logger.info(f"✅ Enhanced Step 1 succeeded on first attempt{context_message}")

                response_data = {
                    'success': True,
                    'data': df,
                    'data_summary': data_summary,
                    'sql_query': final_sql,
                    'explanation': result.explanation,
                    'filename': csv_filename,
                    'shape': list(df.shape),
                    'columns': list(df.columns),
                    'document_url': document_url,
                    'document_key': document_key,
                    'rows_returned': len(df),
                    'attempts_made': result.total_attempts,
                    'retry_details': [
                        {
                            'attempt': attempt.attempt_number,
                            'success': attempt.success,
                            'error_type': attempt.error_type.value if attempt.error_type else None,
                            'error_message': attempt.error_message if not attempt.success else "Success",
                            'sql_attempted': attempt.sql_query,
                            'rows_returned': attempt.rows_returned
                        }
                        for attempt in result.attempts
                    ]
                }
                
                # Add context information
                if context_summary:
                    response_data['context_enhanced'] = True
                    response_data['context_info'] = f"Enhanced with business context{context_message}"
                else:
                    response_data['context_enhanced'] = False
                    response_data['context_info'] = "Using Unity Catalog schema only"

                return response_data

            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

                handler.close_connection()

        except Exception as e:
            logger.error(f"Error in Enhanced Step 1 (fetch data): {e}")
            return {
                'success': False,
                'error': f'Enhanced data fetching failed: {str(e)}',
                'sql_query': '',
                'suggestions': 'Please try again or check your database connection.',
                'attempts_made': 0,
                'retry_details': [],
                'context_enhanced': False,
                'context_info': 'Error occurred before context could be applied'
            }

    def execute_step2_analyze_data(question: str, df) -> dict:
        """
        Step 2: Analyze the fetched data with AI agent (unchanged from original)

        Returns: Dictionary with analysis results
        """
        try:
            # Create context for DataAnalysisAgent
            data_summary = analyze_data(df)

            context = f"""
    Dataset from Enhanced Database Query (Step 1):
    Shape: {df.shape[0]} rows × {df.shape[1]} columns

    Columns: {', '.join(df.columns)}

    Data Types:
    {chr(10).join([f"- {col}: {dtype}" for col, dtype in data_summary['dtypes'].items()])}

    Numeric columns: {', '.join(data_summary['numeric_columns'])}
    Categorical columns: {', '.join(data_summary['categorical_columns'])}

    Sample Values:
    {chr(10).join([f"- {col}: {data_summary['sample_data'].get(col, [])}" for col in list(data_summary['sample_data'].keys())[:5]])}

    DataFrame variable names:
    - current_data: Active dataset from enhanced database query (Step 1)
    - worksheet_data: Dictionary containing the dataset (key: 'Sheet1')

    This data was fetched from the database using an enhanced Text-to-SQL agent with automatic error correction, retry capabilities, and supplemental business context integration.
    """

            # Get the existing agent from app module
            try:
                from app import agent
            except ImportError:
                from agent import DataAnalysisAgent
                agent = DataAnalysisAgent()

            # Import merge function if needed
            try:
                from app import merge_worksheets
            except ImportError:
                merge_worksheets = None

            # Prepare worksheet data format
            worksheet_data = {'Sheet1': df}

            # Process with DataAnalysisAgent
            result = agent.process_query(
                question,
                context,
                current_data=df,
                worksheet_data=worksheet_data,
                merge_worksheets_func=merge_worksheets
            )

            # Ensure result has required fields
            if result.get('success', False):
                if 'output_id' not in result:
                    result['output_id'] = f"analysis_{int(datetime.now().timestamp())}"

                if 'timestamp' not in result:
                    result['timestamp'] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Error in Step 2 (analyze data): {e}")
            return {
                'success': False,
                'error': f'Data analysis failed: {str(e)}',
                'result': 'Analysis could not be completed, but the data was fetched successfully using enhanced retry mechanisms with supplemental context integration.'
            }

    @app.route('/api/database/agent_stats', methods=['GET'])
    def get_agent_statistics():
        """Get statistics about the enhanced Text2SQL agent performance"""
        try:
            # Create agent instance to get stats
            text2sql_agent = create_enhanced_text2sql_agent()
            
            stats = text2sql_agent.get_error_statistics()
            history = text2sql_agent.get_query_history()
            
            return jsonify({
                'success': True,
                'error_statistics': stats,
                'query_history_count': len(history),
                'recent_queries': history[-5:] if history else []  # Last 5 queries
            })
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/database/clear_agent_history', methods=['POST'])
    def clear_agent_history():
        """Clear the enhanced Text2SQL agent history and error patterns"""
        try:
            text2sql_agent = create_enhanced_text2sql_agent()
            text2sql_agent.clear_history()
            
            return jsonify({
                'success': True,
                'message': 'Agent history and error patterns cleared successfully'
            })
            
        except Exception as e:
            logger.error(f"Error clearing agent history: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # ENHANCED: New route to get context status for a configuration
    @app.route('/api/database/context_status/<config_id>')
    def get_context_status(config_id):
        """Get supplemental context status for a database configuration"""
        try:
            context_manager = get_context_manager()
            context_summary = context_manager.get_context_summary(config_id)
            
            if context_summary:
                return jsonify({
                    'success': True,
                    'has_context': True,
                    'context_summary': context_summary,
                    'config_id': config_id
                })
            else:
                return jsonify({
                    'success': True,
                    'has_context': False,
                    'message': 'No supplemental context configured for this database',
                    'config_id': config_id
                })
                
        except Exception as e:
            logger.error(f"Error getting context status for {config_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500