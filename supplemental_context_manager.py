"""
Supplemental Database Context Manager
Handles storage and retrieval of business context that supplements Unity Catalog schema information
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupplementalContextManager:
    """Manages supplemental business context for database configurations"""
    
    def __init__(self, context_dir: str = "db_context"):
        """
        Initialize the context manager
        
        Args:
            context_dir: Directory to store context files
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Supplemental context directory: {self.context_dir.absolute()}")
    
    def _get_context_file_path(self, config_id: str) -> Path:
        """Get the file path for a specific configuration's context"""
        return self.context_dir / f"{config_id}_supplemental_context.json"
    
    def save_context(self, config_id: str, context_data: Dict[str, Any]) -> bool:
        """
        Save supplemental context for a database configuration
        
        Args:
            config_id: Database configuration ID
            context_data: Context data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            # Add metadata
            context_data['_metadata'] = {
                'config_id': config_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # If context already exists, preserve creation time
            existing_context = self.get_context(config_id)
            if existing_context and '_metadata' in existing_context:
                context_data['_metadata']['created_at'] = existing_context['_metadata'].get('created_at')
            
            file_path = self._get_context_file_path(config_id)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved supplemental context for config: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving supplemental context for {config_id}: {e}")
            return False
    
    def get_context(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get supplemental context for a database configuration
        
        Args:
            config_id: Database configuration ID
            
        Returns:
            Optional[Dict]: Context data or None if not found
        """
        try:
            file_path = self._get_context_file_path(config_id)
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            
            logger.info(f"Retrieved supplemental context for config: {config_id}")
            return context_data
            
        except Exception as e:
            logger.error(f"Error retrieving supplemental context for {config_id}: {e}")
            return None
    
    def delete_context(self, config_id: str) -> bool:
        """
        Delete supplemental context for a database configuration
        
        Args:
            config_id: Database configuration ID
            
        Returns:
            bool: Success status
        """
        try:
            file_path = self._get_context_file_path(config_id)
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted supplemental context for config: {config_id}")
                return True
            else:
                logger.warning(f"No supplemental context found to delete for config: {config_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting supplemental context for {config_id}: {e}")
            return False
    
    def list_all_contexts(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available supplemental contexts
        
        Returns:
            Dict: Dictionary mapping config_id to context metadata
        """
        contexts = {}
        
        try:
            for file_path in self.context_dir.glob("*_supplemental_context.json"):
                config_id = file_path.stem.replace("_supplemental_context", "")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        context_data = json.load(f)
                    
                    # Extract summary information
                    contexts[config_id] = {
                        'metadata': context_data.get('_metadata', {}),
                        'has_database_description': bool(context_data.get('database_description')),
                        'has_global_instructions': bool(context_data.get('global_instructions')),
                        'schema_architecture_type': context_data.get('schema_architecture', {}).get('schema_type'),
                        'hierarchy_count': sum(len(hierarchies) for hierarchies in context_data.get('hierarchy_definitions', {}).values()),
                        'glossary_terms_count': len(context_data.get('business_glossary', {})),
                        'kpi_count': len(context_data.get('kpi_definitions', {})),
                        'table_metadata_count': len(context_data.get('table_metadata', {})),
                        'temporal_aggregation_types': list(context_data.get('temporal_aggregation_rules', {}).get('aggregation_types', {}).keys()),
                        'disambiguation_rules_count': len(context_data.get('disambiguation_rules', {}).get('column_rules', {})),
                        'cross_table_relationships_count': len(context_data.get('cross_table_relationships', {}))
                    }
                    
                except Exception as e:
                    logger.error(f"Error reading context file {file_path}: {e}")
                    contexts[config_id] = {'error': str(e)}
            
            logger.info(f"Listed {len(contexts)} supplemental contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error listing supplemental contexts: {e}")
            return {}
    
    def get_context_summary(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of supplemental context for a configuration
        
        Args:
            config_id: Database configuration ID
            
        Returns:
            Optional[Dict]: Context summary or None if not found
        """
        context_data = self.get_context(config_id)
        if not context_data:
            return None
        
        return {
            'config_id': config_id,
            'metadata': context_data.get('_metadata', {}),
            'has_database_description': bool(context_data.get('database_description')),
            'has_global_instructions': bool(context_data.get('global_instructions')),
            'schema_architecture': context_data.get('schema_architecture', {}),
            'hierarchy_definitions_summary': {
                hierarchy_type: list(hierarchies.keys()) 
                for hierarchy_type, hierarchies in context_data.get('hierarchy_definitions', {}).items()
                if hierarchies
            },
            'temporal_aggregation_rules_summary': {
                'default_behavior': context_data.get('temporal_aggregation_rules', {}).get('default_behavior'),
                'aggregation_types': list(context_data.get('temporal_aggregation_rules', {}).get('aggregation_types', {}).keys())
            },
            'business_glossary_terms': list(context_data.get('business_glossary', {}).keys()),
            'kpi_definitions': list(context_data.get('kpi_definitions', {}).keys()),
            'table_metadata': list(context_data.get('table_metadata', {}).keys()),
            'disambiguation_rules_summary': {
                'column_rules': list(context_data.get('disambiguation_rules', {}).get('column_rules', {}).keys()),
                'table_preferences': len(context_data.get('disambiguation_rules', {}).get('table_preferences', {}))
            },
            'cross_table_relationships': list(context_data.get('cross_table_relationships', {}).keys())
        }
    
    def combine_with_database_context(self, config_id: str, database_context: str) -> str:
        """
        Combine supplemental context with database schema context
        
        Args:
            config_id: Database configuration ID
            database_context: Base database context from Unity Catalog
            
        Returns:
            str: Combined context for SQL generation
        """
        supplemental_context = self.get_context(config_id)
        if not supplemental_context:
            logger.info(f"No supplemental context found for {config_id}, using database context only")
            return database_context
        
        try:
            combined_context = f"""
{database_context}

===============================
SUPPLEMENTAL CONTEXT:
===============================

"""
            
            # Add database description
            if supplemental_context.get('database_description'):
                combined_context += f"""
DATABASE BUSINESS DESCRIPTION:
{supplemental_context['database_description']}

"""
            
            # Add global instructions
            if supplemental_context.get('global_instructions'):
                combined_context += f"""
GLOBAL QUERY INSTRUCTIONS:
{supplemental_context['global_instructions']}

"""
            
            # Add schema architecture information
            schema_arch = supplemental_context.get('schema_architecture', {})
            if schema_arch.get('schema_type'):
                combined_context += f"""
SCHEMA ARCHITECTURE:
- Type: {schema_arch['schema_type']}
"""
                if schema_arch.get('hierarchical_complexity'):
                    combined_context += f"- Hierarchical Complexity: {schema_arch['hierarchical_complexity']}\n"
                
                if schema_arch.get('has_temporal_aggregations'):
                    combined_context += f"- Contains Multiple Temporal Aggregations: Yes\n"
                
                if schema_arch.get('has_mixed_actual_forecast'):
                    combined_context += f"- Mixes Actual and Forecasted Data: Yes\n"
                
                if schema_arch.get('has_column_ambiguity'):
                    combined_context += f"- Has Column Name Ambiguities: Yes\n"
                
                combined_context += "\n"
            
            # Add hierarchy definitions
            hierarchy_defs = supplemental_context.get('hierarchy_definitions', {})
            if any(hierarchy_defs.values()):
                combined_context += "HIERARCHY DEFINITIONS:\n"
                
                for hierarchy_type, hierarchies in hierarchy_defs.items():
                    if hierarchies:
                        combined_context += f"\n{hierarchy_type.upper()} HIERARCHIES:\n"
                        for name, hierarchy in hierarchies.items():
                            combined_context += f"- {name}:\n"
                            combined_context += f"  Code Column: {hierarchy.get('code_column', 'N/A')}\n"
                            combined_context += f"  Name Column: {hierarchy.get('name_column', 'N/A')}\n"
                            if hierarchy.get('levels'):
                                combined_context += f"  Levels: {' → '.join(hierarchy['levels'])}\n"
                            if hierarchy.get('description'):
                                combined_context += f"  Description: {hierarchy['description']}\n"
                combined_context += "\n"
            
            # Add temporal aggregation rules
            temporal_rules = supplemental_context.get('temporal_aggregation_rules', {})
            if temporal_rules:
                combined_context += "TEMPORAL AGGREGATION RULES:\n"
                
                if temporal_rules.get('default_behavior'):
                    combined_context += f"Default Behavior: {temporal_rules['default_behavior']}\n"
                
                if temporal_rules.get('actual_vs_forecast_logic'):
                    combined_context += f"Actual vs Forecast Logic: {temporal_rules['actual_vs_forecast_logic']}\n"
                
                aggregation_types = temporal_rules.get('aggregation_types', {})
                if aggregation_types:
                    combined_context += "\nTemporal Aggregation Types:\n"
                    for code, agg_info in aggregation_types.items():
                        combined_context += f"- {code} ({agg_info.get('name', code)}): "
                        combined_context += f"Column: {agg_info.get('column', 'N/A')}, "
                        combined_context += f"Description: {agg_info.get('description', 'N/A')}\n"
                
                combined_context += "\n"
            
            # Add disambiguation rules
            disambig_rules = supplemental_context.get('disambiguation_rules', {})
            if disambig_rules:
                combined_context += "DISAMBIGUATION RULES:\n"
                
                column_rules = disambig_rules.get('column_rules', {})
                if column_rules:
                    combined_context += "\nColumn Disambiguation:\n"
                    for column, rule in column_rules.items():
                        combined_context += f"- {column}: Found in tables {', '.join(rule.get('tables', []))}\n"
                        if rule.get('logic'):
                            combined_context += f"  Logic: {rule['logic']}\n"
                
                table_prefs = disambig_rules.get('table_preferences', {})
                if table_prefs:
                    combined_context += "\nTable Preferences:\n"
                    for pref_key, pref in table_prefs.items():
                        combined_context += f"- Keywords {', '.join(pref.get('keywords', []))}: Prefer {pref.get('table', 'N/A')}\n"
                        if pref.get('logic'):
                            combined_context += f"  Logic: {pref['logic']}\n"
                
                combined_context += "\n"
            
            # Add cross-table relationships
            cross_table_rels = supplemental_context.get('cross_table_relationships', {})
            if cross_table_rels:
                combined_context += "CROSS-TABLE RELATIONSHIPS:\n"
                for rel_key, relationship in cross_table_rels.items():
                    combined_context += f"- {relationship.get('table1', 'N/A')} ↔ {relationship.get('table2', 'N/A')} "
                    combined_context += f"({relationship.get('type', 'unknown')})\n"
                    if relationship.get('logic'):
                        combined_context += f"  Logic: {relationship['logic']}\n"
                combined_context += "\n"
            
            # Add business glossary
            business_glossary = supplemental_context.get('business_glossary', {})
            if business_glossary:
                combined_context += "BUSINESS GLOSSARY:\n"
                for term, definition in business_glossary.items():
                    combined_context += f"- {term}: {definition}\n"
                combined_context += "\n"
            
            # Add KPI definitions
            kpi_definitions = supplemental_context.get('kpi_definitions', {})
            if kpi_definitions:
                combined_context += "KPI DEFINITIONS:\n"
                for kpi, definition in kpi_definitions.items():
                    combined_context += f"- {kpi}: {definition}\n"
                combined_context += "\n"
            
            # Add table metadata
            table_metadata = supplemental_context.get('table_metadata', {})
            if table_metadata:
                combined_context += "ENHANCED TABLE METADATA:\n"
                for table_name, metadata in table_metadata.items():
                    combined_context += f"\nTable: {table_name}\n"
                    if metadata.get('description'):
                        combined_context += f"Description: {metadata['description']}\n"
                    if metadata.get('join_instructions'):
                        combined_context += f"Join Instructions: {metadata['join_instructions']}\n"
                    
                    # Column metadata
                    column_metadata = metadata.get('column_metadata', {})
                    if column_metadata:
                        combined_context += "Enhanced Column Info:\n"
                        for col_name, col_meta in column_metadata.items():
                            combined_context += f"  - {col_name}: {col_meta.get('description', 'N/A')}"
                            if col_meta.get('data_type_info'):
                                combined_context += f" (Type: {col_meta['data_type_info']})"
                            if col_meta.get('allowed_aggregations'):
                                combined_context += f" (Aggregations: {', '.join(col_meta['allowed_aggregations'])})"
                            combined_context += "\n"
                    
                    # Derived KPIs
                    derived_kpis = metadata.get('derived_kpis', {})
                    if derived_kpis:
                        combined_context += "Derived KPIs:\n"
                        for kpi_name, formula in derived_kpis.items():
                            combined_context += f"  - {kpi_name}: {formula}\n"
                    
                    combined_context += "\n"
            
            combined_context += """
===============================
END SUPPLEMENTAL CONTEXT
===============================

IMPORTANT: Use the supplemental context above to enhance your understanding of the database structure, business rules, and user intent when generating SQL queries. Pay special attention to hierarchy definitions, temporal aggregation rules, and disambiguation rules when column names appear in multiple tables.
"""
            
            logger.info(f"Combined database context with supplemental context for config: {config_id}")
            return combined_context
            
        except Exception as e:
            logger.error(f"Error combining contexts for {config_id}: {e}")
            return database_context


# Global instance
_context_manager = None

def get_context_manager() -> SupplementalContextManager:
    """Get global supplemental context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = SupplementalContextManager()
    return _context_manager