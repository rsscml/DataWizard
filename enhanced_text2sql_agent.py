"""
Enhanced Text-to-SQL Agent with Error Correction and Retry Capabilities
Leverages the existing LLM infrastructure from agent.py with advanced error handling
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from langchain.schema import SystemMessage
import re
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of SQL execution errors for targeted handling"""
    SYNTAX_ERROR = "syntax_error"
    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    PERMISSION_ERROR = "permission_error"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    FUNCTION_NOT_FOUND = "function_not_found"
    AGGREGATION_ERROR = "aggregation_error"
    JOIN_ERROR = "join_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class QueryAttempt:
    """Record of a single query attempt"""
    attempt_number: int
    sql_query: str
    error_type: Optional[ErrorType]
    error_message: str
    success: bool
    execution_time: Optional[float] = None
    rows_returned: Optional[int] = None


@dataclass
class SQLGenerationResult:
    """Complete result of SQL generation process"""
    success: bool
    final_sql: str
    explanation: str
    attempts: List[QueryAttempt]
    total_attempts: int
    final_error: Optional[str] = None


class EnhancedText2SQLAgent:
    """Enhanced AI Agent for converting natural language to SQL queries with error correction"""
    
    def __init__(self, llm=None, max_retry_attempts=3):
        """
        Initialize Enhanced Text2SQL agent
        
        Args:
            llm: LangChain LLM instance (if None, will be created from existing infrastructure)
            max_retry_attempts: Maximum number of retry attempts for SQL generation
        """
        self.llm = llm
        if self.llm is None:
            self._initialize_llm()
        
        # Retry configuration
        self.max_retry_attempts = max_retry_attempts
        
        # Track query history for context and learning
        self.query_history = []
        self.error_patterns = {}  # Store common error patterns and fixes
        self.max_history = 20
        
        # Error classification patterns
        self.error_classifiers = self._initialize_error_classifiers()
        
    def _initialize_llm(self):
        """Initialize LLM using existing infrastructure from agent.py"""
        try:
            from agent import create_langchain_llm_with_auto_refresh
            self.llm = create_langchain_llm_with_auto_refresh()
            logger.info("Enhanced Text2SQL agent initialized with auto-refresh LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for Enhanced Text2SQL: {e}")
            raise
    
    def _initialize_error_classifiers(self) -> Dict[ErrorType, List[str]]:
        """Initialize error classification patterns"""
        return {
            ErrorType.SYNTAX_ERROR: [
                "syntax error", "invalid syntax", "mismatched input", "unexpected token",
                "missing", "expected", "parse error"
            ],
            ErrorType.TABLE_NOT_FOUND: [
                "table not found", "table or view not found", "table does not exist",
                "unknown table", "no such table", "relation does not exist"
            ],
            ErrorType.COLUMN_NOT_FOUND: [
                "column not found", "column does not exist", "unknown column",
                "no such column", "invalid column name", "column not available"
            ],
            ErrorType.PERMISSION_ERROR: [
                "permission denied", "access denied", "insufficient privileges",
                "not authorized", "unauthorized", "forbidden"
            ],
            ErrorType.DATA_TYPE_MISMATCH: [
                "data type mismatch", "type mismatch", "cannot convert",
                "invalid type", "incompatible types", "cast error"
            ],
            ErrorType.FUNCTION_NOT_FOUND: [
                "function not found", "unknown function", "no such function",
                "function does not exist", "undefined function"
            ],
            ErrorType.AGGREGATION_ERROR: [
                "group by", "aggregation", "must appear in group by",
                "not in group by clause", "aggregate function"
            ],
            ErrorType.JOIN_ERROR: [
                "join", "ambiguous column", "column reference is ambiguous",
                "multiple tables", "table alias"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "timeout", "query timeout", "execution timeout", "time limit exceeded"
            ]
        }
    
    def generate_sql_with_retry(self, user_question: str, database_context: str, 
                               previous_queries: List[Dict] = None,
                               database_handler=None) -> SQLGenerationResult:
        """
        Generate SQL query with automatic error correction and retry mechanism
        
        Args:
            user_question: User's natural language question
            database_context: Database schema and context information
            previous_queries: List of previous queries for context
            database_handler: Database handler for executing test queries
            
        Returns:
            SQLGenerationResult with complete attempt history
        """
        logger.info(f"Starting SQL generation with retry for: {user_question[:100]}...")
        
        attempts = []
        error_context = ""
        
        for attempt_num in range(1, self.max_retry_attempts + 1):
            logger.info(f"Attempt {attempt_num}/{self.max_retry_attempts}")
            
            try:
                # Generate SQL query (with error context from previous attempts)
                success, sql_query, explanation = self._generate_sql_attempt(
                    user_question, 
                    database_context, 
                    previous_queries or [], 
                    error_context,
                    attempt_num
                )
                
                if not success:
                    # SQL generation failed - record and continue
                    attempt = QueryAttempt(
                        attempt_number=attempt_num,
                        sql_query="",
                        error_type=ErrorType.UNKNOWN_ERROR,
                        error_message=explanation,
                        success=False
                    )
                    attempts.append(attempt)
                    error_context += f"\nAttempt {attempt_num} generation failed: {explanation}"
                    continue
                
                # Basic validation
                is_valid, validation_error = self.validate_sql_query(
                    sql_query, 
                    database_context
                )
                
                if not is_valid:
                    # Validation failed - record and continue
                    attempt = QueryAttempt(
                        attempt_number=attempt_num,
                        sql_query=sql_query,
                        error_type=ErrorType.SYNTAX_ERROR,
                        error_message=validation_error,
                        success=False
                    )
                    attempts.append(attempt)
                    error_context += f"\nAttempt {attempt_num} validation failed: {validation_error}"
                    continue
                
                # Execute query if database handler is provided
                if database_handler:
                    execution_success, result_df, execution_error = database_handler.execute_query(
                        sql_query, 
                        1000  # Limit rows for testing
                    )
                    
                    if execution_success and result_df is not None:
                        # Success! Record successful attempt
                        attempt = QueryAttempt(
                            attempt_number=attempt_num,
                            sql_query=sql_query,
                            error_type=None,
                            error_message="",
                            success=True,
                            rows_returned=len(result_df)
                        )
                        attempts.append(attempt)
                        
                        # Add to query history for learning
                        self._add_successful_query_to_history(user_question, sql_query, explanation)
                        
                        return SQLGenerationResult(
                            success=True,
                            final_sql=sql_query,
                            explanation=explanation,
                            attempts=attempts,
                            total_attempts=attempt_num
                        )
                    
                    else:
                        # Execution failed - analyze error and prepare for retry
                        error_type = self._classify_error(execution_error)
                        attempt = QueryAttempt(
                            attempt_number=attempt_num,
                            sql_query=sql_query,
                            error_type=error_type,
                            error_message=execution_error,
                            success=False
                        )
                        attempts.append(attempt)
                        
                        # Build error context for next attempt
                        error_analysis = self._analyze_execution_error(
                            sql_query, 
                            execution_error, 
                            error_type, 
                            database_context
                        )
                        error_context += f"\nAttempt {attempt_num} execution failed: {error_analysis}"
                        
                        # Store error pattern for learning
                        self._store_error_pattern(user_question, sql_query, execution_error, error_type)
                        
                        logger.warning(f"Attempt {attempt_num} failed: {execution_error}")
                        
                        # If this is the last attempt, break out of loop
                        if attempt_num == self.max_retry_attempts:
                            break
                            
                        # Otherwise, continue to next attempt
                        continue
                
                else:
                    # No database handler provided - return after validation
                    attempt = QueryAttempt(
                        attempt_number=attempt_num,
                        sql_query=sql_query,
                        error_type=None,
                        error_message="",
                        success=True
                    )
                    attempts.append(attempt)
                    
                    return SQLGenerationResult(
                        success=True,
                        final_sql=sql_query,
                        explanation=explanation,
                        attempts=attempts,
                        total_attempts=attempt_num
                    )
                    
            except Exception as e:
                # Unexpected error during attempt
                logger.error(f"Unexpected error in attempt {attempt_num}: {e}")
                attempt = QueryAttempt(
                    attempt_number=attempt_num,
                    sql_query="",
                    error_type=ErrorType.UNKNOWN_ERROR,
                    error_message=str(e),
                    success=False
                )
                attempts.append(attempt)
                error_context += f"\nAttempt {attempt_num} unexpected error: {str(e)}"
        
        # All attempts failed
        final_error = attempts[-1].error_message if attempts else "No attempts made"
        
        return SQLGenerationResult(
            success=False,
            final_sql="",
            explanation=f"Failed after {len(attempts)} attempts",
            attempts=attempts,
            total_attempts=len(attempts),
            final_error=final_error
        )
    
    def _generate_sql_attempt(self, user_question: str, database_context: str, 
                             previous_queries: List[Dict], error_context: str,
                             attempt_number: int) -> Tuple[bool, str, str]:
        """Generate SQL for a single attempt with error context"""
        
        # Create enhanced prompt with error correction context
        system_prompt = self._create_enhanced_text2sql_prompt(
            user_question, 
            database_context, 
            previous_queries,
            error_context,
            attempt_number
        )
        
        try:
            # Use existing LLM infrastructure with retry logic
            response = self._invoke_llm_with_retry([SystemMessage(content=system_prompt)])
            
            # Parse response
            success, sql_query, explanation = self._parse_sql_response(response.content)
            
            if success:
                logger.info(f"SQL generated successfully for attempt {attempt_number}")
                return True, sql_query, explanation
            else:
                logger.error(f"Failed to parse SQL response for attempt {attempt_number}: {explanation}")
                return False, "", explanation
                
        except Exception as e:
            error_msg = f"Error generating SQL for attempt {attempt_number}: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def _create_enhanced_text2sql_prompt(self, user_question: str, database_context: str, 
                                        previous_queries: List[Dict], error_context: str,
                                        attempt_number: int) -> str:
        """Create enhanced Text-to-SQL prompt with error correction guidance"""
        
        # Base prompt from original implementation
        base_prompt = f"""
You are an expert SQL query generator for Azure Databricks Unity Catalog. 
Convert the user's natural language question into a precise SQL query.
CRITICAL: Generate only Apache Spark SQL or Standard SQL. Do not use highly specialized, database specific functions.

{database_context}

USER QUESTION: {user_question}
"""
        
        # Add error correction context if this is a retry attempt
        if attempt_number > 1 and error_context:
            base_prompt += f"""

🔄 ERROR CORRECTION MODE - ATTEMPT {attempt_number}:
Previous attempts have failed. Learn from these errors and generate an improved query:

{error_context}

CRITICAL ERROR CORRECTION GUIDELINES:
1. Carefully analyze the error messages from previous attempts
2. If table/column not found: Double-check names in the database context above
3. If syntax error: Ensure proper Databricks/Spark SQL syntax
4. If join issues: Verify table relationships and use proper aliases
5. If aggregation error: Ensure all non-aggregate columns are in GROUP BY
6. If data type issues: Use proper CAST() functions
7. If function not found: Use standard Spark SQL functions only
8. If timeout: Add WHERE clauses to limit data processing

SPECIFIC FIXES BASED ON ERROR PATTERNS:
"""
            
            # Add specific guidance based on error patterns learned
            if hasattr(self, 'error_patterns') and self.error_patterns:
                for pattern, fixes in self.error_patterns.items():
                    if any(error_term in error_context.lower() for error_term in pattern.split()):
                        base_prompt += f"- {pattern}: {'; '.join(fixes[:3])}\n"
        
        # Add query history context
        if previous_queries:
            base_prompt += f"\nPREVIOUS SUCCESSFUL QUERIES FOR CONTEXT:\n"
            for i, query in enumerate(previous_queries[-3:], 1):  # Show last 3 successful queries
                base_prompt += f"{i}. Question: {query.get('question', '')}\n"
                base_prompt += f"   SQL: {query.get('sql', '')}\n\n"
        
        # Add the rest of the original prompt rules
        base_prompt += """

CRITICAL SQL GENERATION RULES:

1. TABLE REFERENCES:
   - For user tables: Always use fully qualified names: catalog.schema.table_name
   - For system tables: Use catalog.information_schema.table_name
   - Never use just table_name without catalog prefix
   - For names with spaces, use backticks: catalog.schema.`table name`

2. QUERY OPTIMIZATION:
   - Add LIMIT clause for queries that might return large result sets (>10,000 rows)
   - Use appropriate WHERE clauses for filtering
   - Consider using aggregate functions when asking for summaries

3. DATA TYPES & FUNCTIONS:
   - Use appropriate Databricks/Spark SQL functions only
   - Handle date/time operations with proper date functions
   - Use CAST() for type conversions when needed

4. COMMON PATTERNS:
   - "How many..." → COUNT(*)
   - "What is the average..." → AVG(column)
   - "Show me the top..." → ORDER BY ... DESC LIMIT N
   - "Compare..." → GROUP BY with aggregations
   - "Trend over time..." → GROUP BY date/time columns

5. ERROR PREVENTION:
   - Always check that referenced columns exist in the schema
   - Use proper SQL syntax for Databricks/Spark SQL
   - Handle NULL values appropriately
   - Use table aliases to avoid ambiguous column references

6. RESPONSE FORMAT:
   You must respond with EXACTLY this JSON format:
   {
       "sql_query": "your SQL query here",
       "explanation": "explanation of what the query does and how it addresses any previous errors",
       "confidence": "high|medium|low",
       "assumptions": ["list", "of", "assumptions", "made"],
       "error_corrections": ["list", "of", "corrections", "made", "from", "previous", "attempts"]
   }

Now generate the SQL query for the user's question, incorporating all error corrections if this is a retry attempt.
"""
        
        return base_prompt
    
    def _classify_error(self, error_message: str) -> ErrorType:
        """Classify error type based on error message"""
        if not error_message:
            return ErrorType.UNKNOWN_ERROR
        
        error_lower = error_message.lower()
        
        for error_type, patterns in self.error_classifiers.items():
            if any(pattern in error_lower for pattern in patterns):
                return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    def _analyze_execution_error(self, sql_query: str, error_message: str, 
                                error_type: ErrorType, database_context: str) -> str:
        """Analyze execution error and provide specific guidance for correction"""
        
        analysis = f"Error Type: {error_type.value}\nError: {error_message}\n"
        
        if error_type == ErrorType.TABLE_NOT_FOUND:
            # Extract table name from query and suggest corrections
            table_matches = re.findall(r'FROM\s+([^\s\)]+)', sql_query, re.IGNORECASE)
            if table_matches:
                analysis += f"Problematic table reference: {table_matches[0]}\n"
                analysis += "Correction needed: Verify table name exists in database context and use fully qualified name (catalog.schema.table)\n"
        
        elif error_type == ErrorType.COLUMN_NOT_FOUND:
            # Extract column name and suggest corrections
            column_matches = re.findall(r"column['\s]*([^'\s]+)", error_message, re.IGNORECASE)
            if column_matches:
                analysis += f"Problematic column: {column_matches[0]}\n"
                analysis += "Correction needed: Check column name in database schema and ensure proper table qualification\n"
        
        elif error_type == ErrorType.SYNTAX_ERROR:
            analysis += "Correction needed: Review SQL syntax for Databricks/Spark SQL compatibility\n"
            
        elif error_type == ErrorType.AGGREGATION_ERROR:
            analysis += "Correction needed: Ensure all non-aggregate columns are included in GROUP BY clause\n"
            
        elif error_type == ErrorType.JOIN_ERROR:
            analysis += "Correction needed: Use table aliases and qualify all column references to avoid ambiguity\n"
            
        elif error_type == ErrorType.FUNCTION_NOT_FOUND:
            # Extract function name and suggest alternatives
            func_matches = re.findall(r"function['\s]*([^'\s\)]+)", error_message, re.IGNORECASE)
            if func_matches:
                analysis += f"Problematic function: {func_matches[0]}\n"
                analysis += "Correction needed: Use standard Spark SQL functions only. Check Databricks documentation for alternatives.\n"
        
        return analysis
    
    def _store_error_pattern(self, question: str, sql_query: str, error_message: str, error_type: ErrorType):
        """Store error patterns for learning and future correction"""
        
        # Create a simple pattern key
        pattern_key = f"{error_type.value}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = []
        
        # Store correction suggestion
        correction = self._generate_correction_suggestion(sql_query, error_message, error_type)
        if correction and correction not in self.error_patterns[pattern_key]:
            self.error_patterns[pattern_key].append(correction)
            
        # Limit stored patterns to prevent memory issues
        if len(self.error_patterns[pattern_key]) > 10:
            self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-10:]
    
    def _generate_correction_suggestion(self, sql_query: str, error_message: str, error_type: ErrorType) -> str:
        """Generate a correction suggestion based on error type"""
        
        if error_type == ErrorType.TABLE_NOT_FOUND:
            return "Always use fully qualified table names: catalog.schema.table_name"
        elif error_type == ErrorType.COLUMN_NOT_FOUND:
            return "Verify column names in database schema and use proper table qualification"
        elif error_type == ErrorType.FUNCTION_NOT_FOUND:
            return "Use only standard Spark SQL functions - avoid database-specific functions"
        elif error_type == ErrorType.AGGREGATION_ERROR:
            return "Include all non-aggregate columns in GROUP BY clause"
        elif error_type == ErrorType.SYNTAX_ERROR:
            return "Review query syntax for Databricks/Spark SQL compatibility"
        else:
            return "Review query structure and database schema"
    
    def _add_successful_query_to_history(self, question: str, sql_query: str, explanation: str):
        """Add successful query to history for learning"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'sql': sql_query,
            'explanation': explanation,
            'success': True
        }
        
        self.query_history.append(history_entry)
        
        # Maintain history limit
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
    
    def validate_sql_query(self, sql_query: str, database_context: str) -> Tuple[bool, str]:
        """Enhanced SQL query validation"""
        try:
            sql_upper = sql_query.upper().strip()
            
            # Basic SQL validation checks
            validation_errors = []
            
            # Check if it's a SELECT query (we only allow SELECT for safety)
            if not sql_upper.startswith('SELECT'):
                validation_errors.append("Only SELECT queries are allowed")
            
            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    validation_errors.append(f"Dangerous operation '{keyword}' not allowed")
            
            # Check for balanced parentheses
            if sql_query.count('(') != sql_query.count(')'):
                validation_errors.append("Unbalanced parentheses in query")
            
            # Check for basic SQL structure
            required_keywords = ['SELECT']
            for keyword in required_keywords:
                if keyword not in sql_upper:
                    validation_errors.append(f"Missing required keyword: {keyword}")
            
            # Enhanced validation: Check for common syntax issues
            if 'GROUP_CONCAT' in sql_upper:
                validation_errors.append("GROUP_CONCAT is MySQL-specific. Use CONCAT_WS or array functions in Spark SQL")
            
            if 'LIMIT' not in sql_upper and 'TOP' not in sql_upper:
                # This is just a warning, not an error
                pass
            
            if validation_errors:
                return False, "; ".join(validation_errors)
            else:
                return True, "Query validation passed"
                
        except Exception as e:
            return False, f"Error during query validation: {str(e)}"
    
    def _parse_sql_response(self, response_content: str) -> Tuple[bool, str, str]:
        """Parse the LLM response to extract SQL query and explanation (enhanced)"""
        try:
            # Try to parse as JSON first
            if '{' in response_content and '}' in response_content:
                # Extract JSON from response
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                json_str = response_content[json_start:json_end]
                
                try:
                    parsed = json.loads(json_str)
                    sql_query = parsed.get('sql_query', '').strip()
                    explanation = parsed.get('explanation', '').strip()
                    confidence = parsed.get('confidence', 'medium')
                    assumptions = parsed.get('assumptions', [])
                    error_corrections = parsed.get('error_corrections', [])
                    
                    if sql_query:
                        # Enhanced explanation with corrections info
                        if error_corrections:
                            explanation += f"\n\nError corrections made: {', '.join(error_corrections)}"
                        if assumptions:
                            explanation += f"\n\nAssumptions: {', '.join(assumptions)}"
                        
                        return True, sql_query, explanation
                    else:
                        return False, "", "No SQL query found in response"
                        
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Try to extract SQL using patterns (same as original)
            sql_patterns = [
                r'```sql\s*(.*?)\s*```',
                r'```\s*(SELECT.*?)\s*```',
                r'(SELECT\s+.*?)(?:\n\n|\nExplanation|\nNote|$)',
            ]
            
            for pattern in sql_patterns:
                matches = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    sql_query = matches[0].strip()
                    if sql_query.upper().startswith('SELECT'):
                        explanation = "SQL query extracted from response (fallback parsing)"
                        return True, sql_query, explanation
            
            # If no SQL found, return the full response as explanation
            return False, "", f"Could not extract valid SQL from response: {response_content[:500]}"
            
        except Exception as e:
            return False, "", f"Error parsing SQL response: {str(e)}"
    
    def _invoke_llm_with_retry(self, messages, max_retries=2):
        """Invoke LLM with retry logic (same as original)"""
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries and ('401' in error_str or 'unauthorized' in error_str or 'authentication' in error_str):
                    logger.warning(f"Authentication error on attempt {attempt + 1}, refreshing LLM connection...")
                    try:
                        self._reinitialize_llm()
                        logger.info(f"LLM connection refreshed, retrying... (attempt {attempt + 2})")
                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh LLM connection: {refresh_error}")
                        if attempt == max_retries:
                            raise Exception(f"Authentication failed and could not refresh connection: {refresh_error}")
                else:
                    logger.error(f"LLM invocation failed: {e}")
                    raise
        
        raise Exception("Max retries exceeded for LLM invocation")
    
    def _reinitialize_llm(self):
        """Reinitialize LLM connection (same as original)"""
        try:
            from agent import create_langchain_llm_with_auto_refresh
            self.llm = create_langchain_llm_with_auto_refresh()
        except Exception as e:
            logger.error(f"Failed to reinitialize LLM: {e}")
            raise
    
    def get_query_history(self) -> List[Dict]:
        """Get the query history"""
        return self.query_history.copy()
    
    def clear_history(self):
        """Clear the query history and error patterns"""
        self.query_history = []
        self.error_patterns = {}
        logger.info("Query history and error patterns cleared")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about error patterns and corrections"""
        stats = {
            'total_patterns': len(self.error_patterns),
            'error_types': {},
            'most_common_errors': []
        }
        
        for pattern, corrections in self.error_patterns.items():
            stats['error_types'][pattern] = len(corrections)
        
        # Sort by frequency
        sorted_errors = sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True)
        stats['most_common_errors'] = sorted_errors[:5]
        
        return stats

    # Backward compatibility methods - delegate to new retry method
    def generate_sql(self, user_question: str, database_context: str, 
                    previous_queries: List[Dict] = None) -> Tuple[bool, str, str]:
        """
        Backward compatibility method - now uses retry mechanism by default
        """
        result = self.generate_sql_with_retry(
            user_question, database_context, previous_queries
        )
        
        return result.success, result.final_sql, result.explanation
    
    def suggest_query_improvements(self, sql_query: str, execution_error: str = None) -> str:
        """Enhanced query improvement suggestions"""
        suggestions = []
        
        sql_upper = sql_query.upper() if sql_query else ""
        
        # Performance suggestions
        if sql_query and 'LIMIT' not in sql_upper and 'TOP' not in sql_upper:
            suggestions.append("Consider adding LIMIT clause to prevent large result sets")
        
        if sql_query and 'WHERE' not in sql_upper and 'SELECT *' in sql_upper:
            suggestions.append("Consider adding WHERE clause for filtering to improve performance")
        
        # Error-specific suggestions (enhanced)
        if execution_error:
            error_type = self._classify_error(execution_error)
            error_lower = execution_error.lower()
            
            if error_type == ErrorType.TABLE_NOT_FOUND:
                suggestions.extend([
                    "Check table name and ensure catalog.schema.table format is used",
                    "Verify the table exists in the specified catalog and schema",
                    "Use backticks for table names with spaces: `table name`"
                ])
            
            elif error_type == ErrorType.COLUMN_NOT_FOUND:
                suggestions.extend([
                    "Verify column names exist in the table schema",
                    "Check for typos in column names",
                    "Use table aliases to qualify column references"
                ])
            
            elif error_type == ErrorType.SYNTAX_ERROR:
                suggestions.extend([
                    "Check SQL syntax for Databricks/Spark SQL compatibility",
                    "Ensure proper keywords and punctuation are used",
                    "Verify function names are correct for Spark SQL"
                ])
            
            elif error_type == ErrorType.FUNCTION_NOT_FOUND:
                suggestions.extend([
                    "Use standard Spark SQL functions only",
                    "Check Databricks SQL function documentation",
                    "Replace database-specific functions with Spark SQL equivalents"
                ])
            
            elif error_type == ErrorType.AGGREGATION_ERROR:
                suggestions.extend([
                    "Include all non-aggregate columns in GROUP BY clause",
                    "Use appropriate aggregate functions (COUNT, SUM, AVG, etc.)",
                    "Check column references in SELECT vs GROUP BY"
                ])
            
            elif error_type == ErrorType.PERMISSION_ERROR:
                suggestions.extend([
                    "Check access permissions for the referenced tables",
                    "Verify your user account has appropriate database privileges",
                    "Contact database administrator if permissions are needed"
                ])
        
        # Add learned patterns
        if hasattr(self, 'error_patterns') and execution_error:
            error_type = self._classify_error(execution_error)
            if error_type.value in self.error_patterns:
                pattern_suggestions = self.error_patterns[error_type.value][:3]  # Top 3
                suggestions.extend([f"Learned pattern: {s}" for s in pattern_suggestions])
        
        return "\n".join([f"• {suggestion}" for suggestion in suggestions])


def create_enhanced_text2sql_agent(max_retry_attempts=3) -> EnhancedText2SQLAgent:
    """Factory function to create Enhanced Text2SQL agent"""
    try:
        agent = EnhancedText2SQLAgent(max_retry_attempts=max_retry_attempts)
        logger.info("Enhanced Text2SQL agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create Enhanced Text2SQL agent: {e}")
        raise
