"""
Flask routes for Supplemental Context Management
Provides API endpoints for managing business context that supplements Unity Catalog schema information
"""

from flask import request, jsonify, render_template
from database_config import get_config_manager
from supplemental_context_manager import get_context_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_supplemental_context_routes(app):
    """Register supplemental context management routes with the Flask app"""
    
    @app.route('/database-context-manager')
    def database_context_manager():
        """Serve the database context manager interface"""
        try:
            return render_template('database_context_manager.html')
        except Exception as e:
            logger.error(f"Error loading database context manager page: {e}")
            return f"Error loading database context manager: {e}", 500
    
    @app.route('/api/database-configs')
    def get_database_configs_for_context():
        """Get available database configurations for context management"""
        try:
            config_manager = get_config_manager()
            configs = config_manager.get_safe_configs(enabled_only=True)
            
            return jsonify(configs)
            
        except Exception as e:
            logger.error(f"Error getting database configs for context: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/database-configs/<config_id>')
    def get_database_config_details(config_id):
        """Get detailed information about a specific database configuration"""
        try:
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({'error': f'Configuration {config_id} not found'}), 404
            
            # Return safe configuration details (without sensitive info)
            return jsonify(config.get_safe_dict())
            
        except Exception as e:
            logger.error(f"Error getting database config details for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context')
    def get_all_supplemental_contexts():
        """Get list of all supplemental contexts with summary information"""
        try:
            context_manager = get_context_manager()
            contexts = context_manager.list_all_contexts()
            
            return jsonify(contexts)
            
        except Exception as e:
            logger.error(f"Error getting all supplemental contexts: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context/<config_id>')
    def get_supplemental_context(config_id):
        """Get supplemental context for a specific database configuration"""
        try:
            context_manager = get_context_manager()
            context = context_manager.get_context(config_id)
            
            if not context:
                return jsonify({'error': f'No supplemental context found for configuration {config_id}'}), 404
            
            return jsonify(context)
            
        except Exception as e:
            logger.error(f"Error getting supplemental context for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context/<config_id>', methods=['POST'])
    def save_supplemental_context(config_id):
        """Save or update supplemental context for a database configuration"""
        try:
            # Validate that the configuration exists
            config_manager = get_config_manager()
            config = config_manager.get_config(config_id)
            
            if not config:
                return jsonify({'error': f'Database configuration {config_id} not found'}), 404
            
            # Get context data from request
            context_data = request.get_json()
            if not context_data:
                return jsonify({'error': 'No context data provided'}), 400
            
            # Validate basic structure
            if not isinstance(context_data, dict):
                return jsonify({'error': 'Context data must be a JSON object'}), 400
            
            # Save the context
            context_manager = get_context_manager()
            success = context_manager.save_context(config_id, context_data)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Supplemental context saved successfully for {config_id}',
                    'config_id': config_id
                })
            else:
                return jsonify({'error': 'Failed to save supplemental context'}), 500
                
        except Exception as e:
            logger.error(f"Error saving supplemental context for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context/<config_id>', methods=['DELETE'])
    def delete_supplemental_context(config_id):
        """Delete supplemental context for a database configuration"""
        try:
            context_manager = get_context_manager()
            success = context_manager.delete_context(config_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Supplemental context deleted successfully for {config_id}',
                    'config_id': config_id
                })
            else:
                return jsonify({'error': f'No supplemental context found to delete for {config_id}'}), 404
                
        except Exception as e:
            logger.error(f"Error deleting supplemental context for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context/<config_id>/summary')
    def get_supplemental_context_summary(config_id):
        """Get a summary of supplemental context for a database configuration"""
        try:
            context_manager = get_context_manager()
            summary = context_manager.get_context_summary(config_id)
            
            if not summary:
                return jsonify({'error': f'No supplemental context found for configuration {config_id}'}), 404
            
            return jsonify(summary)
            
        except Exception as e:
            logger.error(f"Error getting supplemental context summary for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/supplemental-context/<config_id>/validate', methods=['POST'])
    def validate_supplemental_context(config_id):
        """Validate supplemental context data without saving"""
        try:
            # Get context data from request
            context_data = request.get_json()
            if not context_data:
                return jsonify({'error': 'No context data provided'}), 400
            
            # Basic validation
            validation_errors = []
            warnings = []
            
            # Check basic structure
            if not isinstance(context_data, dict):
                validation_errors.append('Context data must be a JSON object')
                return jsonify({
                    'valid': False,
                    'errors': validation_errors,
                    'warnings': warnings
                })
            
            # Validate schema architecture if provided
            schema_arch = context_data.get('schema_architecture', {})
            if schema_arch:
                valid_schema_types = ['normalized', 'star', 'flattened_hierarchical', 'hybrid']
                if schema_arch.get('schema_type') and schema_arch['schema_type'] not in valid_schema_types:
                    validation_errors.append(f"Invalid schema type. Must be one of: {', '.join(valid_schema_types)}")
            
            # Validate hierarchy definitions
            hierarchy_defs = context_data.get('hierarchy_definitions', {})
            if hierarchy_defs:
                for hierarchy_type, hierarchies in hierarchy_defs.items():
                    if not isinstance(hierarchies, dict):
                        validation_errors.append(f"Hierarchy definitions for {hierarchy_type} must be a dictionary")
                        continue
                    
                    for hierarchy_name, hierarchy_data in hierarchies.items():
                        if not isinstance(hierarchy_data, dict):
                            validation_errors.append(f"Hierarchy data for {hierarchy_name} must be a dictionary")
                            continue
                        
                        # Check required fields
                        if not hierarchy_data.get('name'):
                            warnings.append(f"Hierarchy {hierarchy_name} is missing a name")
                        if not hierarchy_data.get('code_column') and not hierarchy_data.get('name_column'):
                            warnings.append(f"Hierarchy {hierarchy_name} should have at least a code or name column")
            
            # Validate temporal aggregation rules
            temporal_rules = context_data.get('temporal_aggregation_rules', {})
            if temporal_rules:
                aggregation_types = temporal_rules.get('aggregation_types', {})
                if aggregation_types:
                    for agg_code, agg_data in aggregation_types.items():
                        if not isinstance(agg_data, dict):
                            validation_errors.append(f"Temporal aggregation data for {agg_code} must be a dictionary")
                            continue
                        
                        if not agg_data.get('name'):
                            warnings.append(f"Temporal aggregation {agg_code} is missing a descriptive name")
            
            # Validate table metadata
            table_metadata = context_data.get('table_metadata', {})
            if table_metadata:
                for table_name, table_data in table_metadata.items():
                    if not isinstance(table_data, dict):
                        validation_errors.append(f"Table metadata for {table_name} must be a dictionary")
                        continue
                    
                    if not table_data.get('description'):
                        warnings.append(f"Table {table_name} is missing a description")
                    
                    # Validate column metadata
                    column_metadata = table_data.get('column_metadata', {})
                    if column_metadata:
                        valid_data_types = ['absolute_values', 'percentages', 'index_values', 'currency', 
                                          'dates', 'categorical', 'hierarchical_code', 'hierarchical_name']
                        
                        for col_name, col_data in column_metadata.items():
                            if not isinstance(col_data, dict):
                                validation_errors.append(f"Column metadata for {table_name}.{col_name} must be a dictionary")
                                continue
                            
                            data_type_info = col_data.get('data_type_info')
                            if data_type_info and data_type_info not in valid_data_types:
                                validation_errors.append(f"Invalid data type info for {table_name}.{col_name}. Must be one of: {', '.join(valid_data_types)}")
            
            # Validate business glossary
            business_glossary = context_data.get('business_glossary', {})
            if business_glossary:
                if not isinstance(business_glossary, dict):
                    validation_errors.append('Business glossary must be a dictionary')
                else:
                    for term, definition in business_glossary.items():
                        if not isinstance(term, str) or not isinstance(definition, str):
                            validation_errors.append(f"Business glossary term '{term}' and definition must be strings")
                        if not definition.strip():
                            warnings.append(f"Business glossary term '{term}' has an empty definition")
            
            # Validate KPI definitions
            kpi_definitions = context_data.get('kpi_definitions', {})
            if kpi_definitions:
                if not isinstance(kpi_definitions, dict):
                    validation_errors.append('KPI definitions must be a dictionary')
                else:
                    for kpi, definition in kpi_definitions.items():
                        if not isinstance(kpi, str) or not isinstance(definition, str):
                            validation_errors.append(f"KPI '{kpi}' and definition must be strings")
                        if not definition.strip():
                            warnings.append(f"KPI '{kpi}' has an empty definition")
            
            is_valid = len(validation_errors) == 0
            
            return jsonify({
                'valid': is_valid,
                'errors': validation_errors,
                'warnings': warnings,
                'config_id': config_id
            })
            
        except Exception as e:
            logger.error(f"Error validating supplemental context for {config_id}: {e}")
            return jsonify({'error': str(e)}), 500