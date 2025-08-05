"""
Database Configuration Manager for Azure Databricks Unity Catalog
Manages multiple database configurations for different users
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for Azure Databricks"""
    id: str
    name: str
    description: str
    server_hostname: str
    http_path: str
    catalog: str
    schema: str
    azure_tenant_id: str
    azure_client_id: str
    azure_client_secret: str
    enabled: bool = True
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'server_hostname': self.server_hostname,
            'http_path': self.http_path,
            'catalog': self.catalog,
            'schema': self.schema,
            'azure_tenant_id': self.azure_tenant_id,
            'azure_client_id': self.azure_client_id,
            'azure_client_secret': self.azure_client_secret,
            'enabled': self.enabled,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DatabaseConfig':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            server_hostname=data['server_hostname'],
            http_path=data['http_path'],
            catalog=data['catalog'],
            schema=data['schema'],
            azure_tenant_id=data['azure_tenant_id'],
            azure_client_id=data['azure_client_id'],
            azure_client_secret=data['azure_client_secret'],
            enabled=data.get('enabled', True),
            tags=data.get('tags', [])
        )

    def get_safe_dict(self) -> Dict:
        """Get dictionary without sensitive information"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'server_hostname': self.server_hostname,
            'http_path': self.http_path,
            'catalog': self.catalog,
            'schema': self.schema,
            'enabled': self.enabled,
            'tags': self.tags
        }


class DatabaseConfigManager:
    """Manages multiple database configurations"""
    
    def __init__(self, config_file: str = None):
        """Initialize database config manager"""
        self.config_file = config_file or os.getenv('DATABASE_CONFIG_FILE', 'database_configs.json')
        self.configs: Dict[str, DatabaseConfig] = {}
        self.load_configs()
    
    def load_configs(self):
        """Load database configurations from file and environment variables"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for config_data in data.get('databases', []):
                        config = DatabaseConfig.from_dict(config_data)
                        self.configs[config.id] = config
                logger.info(f"Loaded {len(self.configs)} database configurations from file")
            
            # Load from environment variables
            self._load_from_environment()
            
            # Ensure we have at least one configuration
            if not self.configs:
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"Error loading database configurations: {e}")
            self._create_default_config()
    
    def _load_from_environment(self):
        """Load configurations from environment variables"""
        try:
            # Load main database configuration from environment
            server_hostname = os.getenv('DATABRICKS_SERVER_HOSTNAME')
            http_path = os.getenv('DATABRICKS_HTTP_PATH')
            catalog = os.getenv('DATABRICKS_CATALOG')
            schema = os.getenv('DATABRICKS_SCHEMA')
            tenant_id = os.getenv('AZURE_TENANT_ID')
            client_id = os.getenv('AZURE_CLIENT_ID')
            client_secret = os.getenv('AZURE_CLIENT_SECRET')
            
            if all([server_hostname, http_path, catalog, schema, tenant_id, client_id, client_secret]):
                config = DatabaseConfig(
                    id='default',
                    name='Default Database',
                    description='Default database configuration from environment variables',
                    server_hostname=server_hostname,
                    http_path=http_path,
                    catalog=catalog,
                    schema=schema,
                    azure_tenant_id=tenant_id,
                    azure_client_id=client_id,
                    azure_client_secret=client_secret,
                    tags=['default', 'environment']
                )
                self.configs['default'] = config
                logger.info("Loaded default database configuration from environment variables")
            
            # Load additional configurations if they exist
            # Format: DB_CONFIG_<ID>_<FIELD>=value
            env_configs = {}
            for key, value in os.environ.items():
                if key.startswith('DB_CONFIG_'):
                    parts = key.split('_', 3)
                    if len(parts) >= 4:
                        config_id = parts[2].lower()
                        field = parts[3].lower()
                        
                        if config_id not in env_configs:
                            env_configs[config_id] = {}
                        env_configs[config_id][field] = value
            
            # Create configurations from environment variables
            for config_id, config_data in env_configs.items():
                required_fields = [
                    'name', 'server_hostname', 'http_path', 'catalog', 'schema',
                    'azure_tenant_id', 'azure_client_id', 'azure_client_secret'
                ]
                
                if all(field in config_data for field in required_fields):
                    config = DatabaseConfig(
                        id=config_id,
                        name=config_data['name'],
                        description=config_data.get('description', f'Database configuration for {config_id}'),
                        server_hostname=config_data['server_hostname'],
                        http_path=config_data['http_path'],
                        catalog=config_data['catalog'],
                        schema=config_data['schema'],
                        azure_tenant_id=config_data['azure_tenant_id'],
                        azure_client_id=config_data['azure_client_id'],
                        azure_client_secret=config_data['azure_client_secret'],
                        tags=['environment']
                    )
                    self.configs[config_id] = config
                    logger.info(f"Loaded database configuration '{config_id}' from environment variables")
                    
        except Exception as e:
            logger.warning(f"Error loading configurations from environment: {e}")
    
    def _create_default_config(self):
        """Create a default configuration as example"""
        logger.warning("No database configurations found, creating example configuration")
        
        example_config = DatabaseConfig(
            id='example',
            name='Example Database',
            description='Example configuration - please update with your actual database details',
            server_hostname='your-workspace.azuredatabricks.net',
            http_path='/sql/1.0/warehouses/your-warehouse-id',
            catalog='your_catalog',
            schema='your_schema',
            azure_tenant_id='your-tenant-id',
            azure_client_id='your-client-id',
            azure_client_secret='your-client-secret',
            enabled=False,
            tags=['example']
        )
        
        self.configs['example'] = example_config
    
    def save_configs(self):
        """Save configurations to file"""
        try:
            data = {
                'databases': [config.to_dict() for config in self.configs.values()]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.configs)} database configurations to file")
            
        except Exception as e:
            logger.error(f"Error saving database configurations: {e}")
    
    def get_config(self, config_id: str) -> Optional[DatabaseConfig]:
        """Get database configuration by ID"""
        return self.configs.get(config_id)
    
    def get_all_configs(self, enabled_only: bool = True) -> List[DatabaseConfig]:
        """Get all database configurations"""
        configs = list(self.configs.values())
        if enabled_only:
            configs = [config for config in configs if config.enabled]
        return configs
    
    def get_safe_configs(self, enabled_only: bool = True) -> List[Dict]:
        """Get all configurations without sensitive information"""
        configs = self.get_all_configs(enabled_only)
        return [config.get_safe_dict() for config in configs]
    
    def add_config(self, config: DatabaseConfig):
        """Add new database configuration"""
        self.configs[config.id] = config
        self.save_configs()
        logger.info(f"Added database configuration: {config.name}")
    
    def update_config(self, config: DatabaseConfig):
        """Update existing database configuration"""
        if config.id in self.configs:
            self.configs[config.id] = config
            self.save_configs()
            logger.info(f"Updated database configuration: {config.name}")
        else:
            raise ValueError(f"Configuration {config.id} not found")
    
    def delete_config(self, config_id: str):
        """Delete database configuration"""
        if config_id in self.configs:
            del self.configs[config_id]
            self.save_configs()
            logger.info(f"Deleted database configuration: {config_id}")
        else:
            raise ValueError(f"Configuration {config_id} not found")
    
    def validate_config(self, config: DatabaseConfig) -> tuple[bool, str]:
        """Validate database configuration"""
        try:
            # Check required fields
            required_fields = [
                'id', 'name', 'server_hostname', 'http_path', 'catalog', 'schema',
                'azure_tenant_id', 'azure_client_id', 'azure_client_secret'
            ]
            
            for field in required_fields:
                value = getattr(config, field, None)
                if not value or not str(value).strip():
                    return False, f"Missing required field: {field}"
            
            # Validate hostname format
            if not config.server_hostname.endswith('.azuredatabricks.net'):
                return False, "Server hostname should end with '.azuredatabricks.net'"
            
            # Validate HTTP path format
            if not config.http_path.startswith('/sql/'):
                return False, "HTTP path should start with '/sql/'"
            
            # Check for duplicate IDs (excluding the current config if updating)
            existing_ids = [cfg.id for cfg in self.configs.values() if cfg.id != config.id]
            if config.id in existing_ids:
                return False, f"Configuration ID '{config.id}' already exists"
            
            return True, "Configuration is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def test_config(self, config: DatabaseConfig) -> tuple[bool, str]:
        """Test database configuration by attempting connection"""
        try:
            # Import here to avoid circular dependency
            from databricks_handler_dbapi import DatabricksHandler
            
            # Create handler with the configuration
            handler = DatabricksHandler(
                server_hostname=config.server_hostname,
                http_path=config.http_path,
                catalog=config.catalog,
                schema=config.schema
            )
            
            # Set the Azure credentials temporarily
            original_env = {}
            env_vars = {
                'AZURE_TENANT_ID': config.azure_tenant_id,
                'AZURE_CLIENT_ID': config.azure_client_id,
                'AZURE_CLIENT_SECRET': config.azure_client_secret
            }
            
            # Temporarily set environment variables
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Test connection
                success, message = handler.test_connection()
                return success, message
            finally:
                # Restore original environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                
                # Close handler connection
                handler.close_connection()
                
        except Exception as e:
            return False, f"Connection test error: {str(e)}"


# Global instance
_config_manager = None

def get_config_manager() -> DatabaseConfigManager:
    """Get global database configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DatabaseConfigManager()
    return _config_manager