"""
Enhanced Azure Databricks Unity Catalog Database Handler
Handles connection, schema browsing, and query execution for Databricks
ENHANCED: Now integrates supplemental business context for improved Text-to-SQL generation
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import tempfile
import time
from dotenv import load_dotenv

# Azure authentication imports
from azure.identity import ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError

# Databricks SQL Connector
from databricks import sql
from databricks.sql.client import Connection, Cursor

# ENHANCED: Import supplemental context manager
from supplemental_context_manager import get_context_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Databricks Azure AD Application ID (fixed for all Azure Databricks instances)
DATABRICKS_RESOURCE_ID = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d"


class EnhancedDatabricksHandler:
    """Enhanced Databricks handler with supplemental business context integration"""
    
    def __init__(self, server_hostname: str = None, http_path: str = None, access_token: str = None, 
                 catalog: str = None, schema: str = None, config_id: str = None):
        """
        Initialize Enhanced Databricks connection
        
        Args:
            server_hostname: Databricks workspace hostname (e.g., adb-xxxxx.azuredatabricks.net)
            http_path: SQL warehouse HTTP path (e.g., /sql/1.0/warehouses/xxxxx)
            access_token: Personal access token or service principal token (optional - will be generated if not provided)
            catalog: Unity Catalog name (optional)
            schema: Schema name within catalog (optional)
            config_id: Database configuration ID for supplemental context lookup (optional)
        """
        # Get configuration from environment variables if not provided
        self.server_hostname = server_hostname or os.getenv('DATABRICKS_SERVER_HOSTNAME')
        self.http_path = http_path or os.getenv('DATABRICKS_HTTP_PATH')
        self.catalog = catalog or os.getenv('DATABRICKS_CATALOG')
        self.schema = schema or os.getenv('DATABRICKS_SCHEMA')
        
        # ENHANCED: Store config_id for supplemental context
        self.config_id = config_id
        
        # Validate required configuration
        if not self.server_hostname or not self.http_path:
            raise ValueError(
                "Missing required Databricks configuration. Please set: "
                "DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH or provide them as parameters"
            )
        
        # Initialize access token (will be generated if not provided)
        self.access_token = access_token
        
        # Token cache for Service Principal authentication
        self._token_cache = {
            'token': None,
            'expires_at': None
        }
        
        # Azure Service Principal configuration
        self._azure_tenant_id = os.getenv('AZURE_TENANT_ID')
        self._azure_client_id = os.getenv('AZURE_CLIENT_ID')
        self._azure_client_secret = os.getenv('AZURE_CLIENT_SECRET')
        
        # Databricks connection objects
        self.connection: Optional[Connection] = None
        self.cursor: Optional[Cursor] = None
        
        # Connection test flag
        self._connection_tested = False
        self._connection_valid = False
        
        # Get access token if not provided
        if not self.access_token:
            self.access_token = self.get_access_token()

    def get_access_token(self):
        """Get access token using Service Principal authentication with caching"""
        
        # Check if cached token is still valid (refresh 5 minutes before expiry)
        if (self._token_cache['token'] and self._token_cache['expires_at'] and
                datetime.now() < self._token_cache['expires_at'] - timedelta(minutes=5)):
            logger.info("Using cached access token")
            return self._token_cache['token']

        try:
            # Validate configuration
            if not all([self._azure_tenant_id, self._azure_client_id, self._azure_client_secret]):
                raise ValueError(
                    "Service Principal credentials not configured. Check AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET")

            # Create Service Principal credential
            credential = ClientSecretCredential(
                tenant_id=self._azure_tenant_id,
                client_id=self._azure_client_id,
                client_secret=self._azure_client_secret
            )

            # Get token for Databricks resource
            logger.info("Requesting new access token via Service Principal")
            token = credential.get_token(f"{DATABRICKS_RESOURCE_ID}/.default")

            # Cache the token
            expires_at = datetime.now() + timedelta(seconds=token.expires_on - time.time())
            self._token_cache['token'] = token.token
            self._token_cache['expires_at'] = expires_at

            logger.info(f"Successfully obtained token. Expires at: {expires_at}")
            return token.token

        except ClientAuthenticationError as e:
            logger.error(f"Authentication failed - check your Service Principal credentials: {str(e)}")
            raise Exception(f"Authentication failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to obtain access token: {str(e)}")
            raise Exception(f"Token acquisition failed: {str(e)}")

    def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection to Databricks"""
        try:
            # Check if connection exists and is valid
            if self.connection and self.cursor:
                return True
            
            # Ensure we have a valid access token
            if not self.access_token:
                self.access_token = self.get_access_token()
            
            # Create new connection
            logger.info("Creating new Databricks connection...")
            self.connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token
            )
            
            # Create cursor
            self.cursor = self.connection.cursor()
            
            logger.info("Databricks connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish connection: {str(e)}")
            self.connection = None
            self.cursor = None
            return False
        
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the database connection
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info("Testing Databricks connection...")
            
            if not self._ensure_connection():
                self._connection_tested = True
                self._connection_valid = False
                return False, "Failed to establish connection"
            
            # Test with simple query
            test_query = "SELECT 1 as test"
            self.cursor.execute(test_query)
            result = self.cursor.fetchall()
            
            if result and result[0][0] == 1:
                self._connection_tested = True
                self._connection_valid = True
                logger.info("Databricks connection test successful")
                return True, "Connection successful"
            else:
                self._connection_tested = True
                self._connection_valid = False
                return False, "Connection test failed - unexpected result"
                
        except Exception as e:
            self._connection_tested = True
            self._connection_valid = False
            error_msg = f"Connection test error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _execute_query_internal(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """Internal method to execute SQL query using cursor"""
        try:
            if not self._ensure_connection():
                return False, None, "Failed to establish connection"
            
            # Execute query
            self.cursor.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
            
            # Fetch all results
            rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            if rows:
                df = pd.DataFrame(rows, columns=columns)
            else:
                df = pd.DataFrame(columns=columns)
            
            return True, df, None
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a token expiry issue
            if "token" in error_msg.lower() or "auth" in error_msg.lower():
                logger.warning("Token may have expired, refreshing...")
                # Clear cached token and try to get new one
                self._token_cache['token'] = None
                self._token_cache['expires_at'] = None
                self.access_token = self.get_access_token()
                
                # Close existing connection and retry
                if self.connection:
                    self.connection.close()
                self.connection = None
                self.cursor = None
                
                # Retry once with new token
                try:
                    if self._ensure_connection():
                        self.cursor.execute(query)
                        columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
                        rows = self.cursor.fetchall()
                        
                        if rows:
                            df = pd.DataFrame(rows, columns=columns)
                        else:
                            df = pd.DataFrame(columns=columns)
                        
                        return True, df, None
                except Exception as retry_e:
                    return False, None, f"Retry failed: {str(retry_e)}"
            
            return False, None, error_msg
    
    def get_catalogs(self) -> List[str]:
        """Get list of available catalogs"""
        try:
            if not self._connection_valid:
                success, msg = self.test_connection()
                if not success:
                    logger.error(f"Cannot get catalogs: {msg}")
                    return []
            
            query = "SHOW CATALOGS"
            success, df, error = self._execute_query_internal(query)
            
            if success and df is not None and not df.empty:
                # Extract catalog names
                if 'catalog' in df.columns:
                    catalogs = df['catalog'].tolist()
                elif 'name' in df.columns:
                    catalogs = df['name'].tolist()
                else:
                    # Fallback - use first column
                    catalogs = df.iloc[:, 0].tolist()
                
                logger.info(f"Found {len(catalogs)} catalogs")
                return catalogs
            else:
                logger.error(f"Failed to get catalogs: {error}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting catalogs: {str(e)}")
            return []
    
    def get_schemas(self, catalog_name: str = None) -> List[str]:
        """Get list of schemas in a catalog"""
        try:
            if not self._connection_valid:
                success, msg = self.test_connection()
                if not success:
                    logger.error(f"Cannot get schemas: {msg}")
                    return []
            
            catalog = catalog_name or self.catalog
            if not catalog:
                logger.error("No catalog specified")
                return []
            
            query = f"SHOW SCHEMAS IN {catalog}"
            success, df, error = self._execute_query_internal(query)
            
            if success and df is not None and not df.empty:
                # Extract schema names
                if 'schemaName' in df.columns:
                    schemas = df['schemaName'].tolist()
                elif 'schema' in df.columns:
                    schemas = df['schema'].tolist()
                elif 'name' in df.columns:
                    schemas = df['name'].tolist()
                else:
                    # Fallback - use first column
                    schemas = df.iloc[:, 0].tolist()
                
                logger.info(f"Found {len(schemas)} schemas in catalog {catalog}")
                return schemas
            else:
                logger.error(f"Failed to get schemas: {error}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting schemas: {str(e)}")
            return []
    
    def get_tables(self, catalog_name: str = None, schema_name: str = None) -> List[Dict]:
        """Get list of tables in a schema"""
        try:
            if not self._connection_valid:
                success, msg = self.test_connection()
                if not success:
                    logger.error(f"Cannot get tables: {msg}")
                    return []
            
            catalog = catalog_name or self.catalog
            schema = schema_name or self.schema
            
            if not catalog or not schema:
                logger.error("Both catalog and schema must be specified")
                return []
            
            query = f"SHOW TABLES IN {catalog}.{schema}"
            success, df, error = self._execute_query_internal(query)
            
            if success and df is not None and not df.empty:
                tables = []
                for _, row in df.iterrows():
                    table_info = {
                        'name': row.get('tableName', row.get('name', '')),
                        'type': row.get('tableType', row.get('type', 'TABLE')),
                        'catalog': catalog,
                        'schema': schema
                    }
                    tables.append(table_info)
                
                logger.info(f"Found {len(tables)} tables in {catalog}.{schema}")
                return tables
            else:
                logger.error(f"Failed to get tables: {error}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting tables: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str, catalog_name: str = None, schema_name: str = None) -> List[Dict]:
        """Get table schema information"""
        try:
            if not self._connection_valid:
                success, msg = self.test_connection()
                if not success:
                    logger.error(f"Cannot get table schema: {msg}")
                    return []
            
            catalog = catalog_name or self.catalog
            schema = schema_name or self.schema
            
            if not catalog or not schema:
                logger.error("Both catalog and schema must be specified")
                return []
            
            full_table_name = f"{catalog}.{schema}.{table_name}"
            query = f"DESCRIBE {full_table_name}"
            
            success, df, error = self._execute_query_internal(query)
            
            if success and df is not None and not df.empty:
                schema_info = []
                for _, row in df.iterrows():
                    column_info = {
                        'name': row.get('col_name', row.get('column_name', '')),
                        'type': row.get('data_type', row.get('type', '')),
                        'comment': row.get('comment', '')
                    }
                    schema_info.append(column_info)
                
                logger.info(f"Retrieved schema for {full_table_name} with {len(schema_info)} columns")
                return schema_info
            else:
                logger.error(f"Failed to get table schema: {error}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return []
    
    def execute_query(self, query: str, max_rows: int = 100000) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            max_rows: Maximum number of rows to return
            
        Returns:
            Tuple of (success: bool, dataframe: pd.DataFrame or None, error: str or None)
        """
        try:
            if not self._connection_valid:
                success, msg = self.test_connection()
                if not success:
                    return False, None, f"Connection failed: {msg}"
            
            logger.info(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Add LIMIT if not present and max_rows is specified
            if max_rows and max_rows > 0:
                query_upper = query.upper().strip()
                if not any(keyword in query_upper for keyword in ['LIMIT', 'TOP']):
                    query = f"{query.rstrip(';')} LIMIT {max_rows}"
            
            success, df, error = self._execute_query_internal(query)
            
            if success:
                logger.info(f"Query executed successfully. Retrieved {len(df) if df is not None else 0} rows")
                return True, df, None
            else:
                logger.error(f"Query execution failed: {error}")
                return False, None, error
                
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def save_query_result_to_csv(self, df: pd.DataFrame, query_description: str = "") -> str:
        """Save query result to temporary CSV file"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_description = "".join(c for c in query_description if c.isalnum() or c in (' ', '_', '-')).strip()[:50]
            if safe_description:
                filename = f"databricks_query_{safe_description}_{timestamp}.csv"
            else:
                filename = f"databricks_query_{timestamp}.csv"
            
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            logger.info(f"Query result saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving query result to CSV: {str(e)}")
            raise e
    
    def get_database_context(self, catalog_name: str = None, schema_name: str = None, 
                           include_sample_data: bool = True) -> str:
        """
        ENHANCED: Get comprehensive database context including supplemental business context
        
        Args:
            catalog_name: Catalog name (optional)
            schema_name: Schema name (optional)
            include_sample_data: Whether to include sample data
            
        Returns:
            String containing combined database and business context
        """
        try:
            catalog = catalog_name or self.catalog
            schema = schema_name or self.schema
            
            if not catalog or not schema:
                return "No catalog or schema specified"
            
            # Get basic database context from Unity Catalog
            base_context = self._get_base_database_context(catalog, schema, include_sample_data)
            
            # ENHANCED: Combine with supplemental context if available
            if self.config_id:
                try:
                    context_manager = get_context_manager()
                    combined_context = context_manager.combine_with_database_context(
                        self.config_id, base_context
                    )
                    logger.info(f"Combined database context with supplemental context for config: {self.config_id}")
                    return combined_context
                except Exception as e:
                    logger.warning(f"Failed to combine with supplemental context for {self.config_id}: {e}")
                    logger.info("Falling back to base database context only")
                    return base_context
            else:
                logger.info("No config_id provided, using base database context only")
                return base_context
            
        except Exception as e:
            logger.error(f"Error getting database context: {str(e)}")
            return f"Error getting database context: {str(e)}"
    
    def _get_base_database_context(self, catalog: str, schema: str, include_sample_data: bool = True) -> str:
        """
        Get base database context from Unity Catalog (original implementation)
        
        Args:
            catalog: Catalog name
            schema: Schema name  
            include_sample_data: Whether to include sample data
            
        Returns:
            String containing base database schema information
        """
        try:
            context = f"""
DATABASE CONTEXT FOR TEXT-TO-SQL (Unity Catalog):
Catalog: {catalog}
Schema: {schema}
Full Namespace: {catalog}.{schema}

AVAILABLE TABLES:
"""
            
            # Get tables
            tables = self.get_tables(catalog, schema)
            
            for table in tables[:25]:  # Limit to first 25 tables
                table_name = table['name']
                context += f"\nTable: {table_name}\n"
                context += f"Type: {table.get('type', 'TABLE')}\n"
                
                # Get table schema
                schema_info = self.get_table_schema(table_name, catalog, schema)
                
                if schema_info:
                    context += "Columns:\n"
                    for col in schema_info[:50]:  # Limit to first 50 columns
                        context += f"  - {col['name']} ({col['type']})"
                        if col.get('comment'):
                            context += f" - {col['comment']}"
                        context += "\n"
                
                # Get sample data if requested
                if include_sample_data:
                    try:
                        sample_query = f"SELECT * FROM {catalog}.{schema}.{table_name} LIMIT 3"
                        success, sample_df, _ = self.execute_query(sample_query)
                        
                        if success and sample_df is not None and not sample_df.empty:
                            context += "Sample Data:\n"
                            for _, row in sample_df.head(2).iterrows():
                                context += f"  {dict(row)}\n"
                    except:
                        pass  # Skip sample data if query fails
                
                context += "\n" + "="*50 + "\n"
            
            if len(tables) > 25:
                context += f"\n... and {len(tables) - 25} more tables\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting base database context: {str(e)}")
            return f"Error getting base database context: {str(e)}"
    
    def close_connection(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
                logger.info("Database cursor closed")
            if self.connection:
                self.connection.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")
        finally:
            self.cursor = None
            self.connection = None
            self._connection_valid = False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()


def create_enhanced_databricks_handler_from_config(config_id: str = None) -> EnhancedDatabricksHandler:
    """Create Enhanced DatabricksHandler from environment variables with optional config_id"""
    return EnhancedDatabricksHandler(config_id=config_id)


# For backward compatibility - maintain the original class name as an alias
DatabricksHandler = EnhancedDatabricksHandler