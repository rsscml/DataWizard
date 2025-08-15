"""
Session Context Enhancer Module
===============================

This module improves how session history is formatted and presented to the LLM
to enable better follow-up question support and contextual understanding.

Usage in app.py:
    from session_context_enhancer import enhance_session_context
    
    # Replace the session history context creation with:
    enhanced_context = enhance_session_context(context, current_session_history, user_query)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SessionContextEnhancer:
    """Enhances session context for better follow-up question support"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration for session context enhancement"""
        return {
            'max_history_entries': 5,
            'recent_entries_weight': 5,  # Number of most recent entries to emphasize
            'context_relevance_threshold': 0.3,
            'max_code_lines_in_context': 500,
            'max_result_summary_length': 16000,
            'follow_up_keywords': [
                'also', 'additionally', 'furthermore', 'what about', 'how about',
                'compare', 'versus', 'vs', 'difference', 'similar', 'like that',
                'same', 'previous', 'last', 'earlier', 'before', 'above',
                'it', 'this', 'that', 'them', 'those', 'these',
                'drill down', 'drill into', 'expand on', 'tell me more',
                'break down', 'details', 'specifically', 'focus on'
            ],
            'reference_keywords': [
                'that analysis', 'the chart', 'the table', 'the result',
                'that result', 'those findings', 'the data', 'that data',
                'the values', 'those numbers', 'the metrics'
            ]
        }
    
    def enhance_session_context(self, base_context: str, session_history: List[Dict], 
                               current_query: str) -> str:
        """
        Main function to enhance session context with improved history formatting
        
        Args:
            base_context: The base data context string
            session_history: List of previous session interactions
            current_query: The current user query
            
        Returns:
            Enhanced context string with better session history
        """
        if not session_history:
            return base_context
        
        logger.info(f"Enhancing context with {len(session_history)} history entries")
        
        # Analyze current query for follow-up indicators
        query_analysis = self._analyze_query_for_followup(current_query, session_history)
        
        # Filter and rank history entries based on relevance
        relevant_history = self._filter_relevant_history(session_history, current_query, query_analysis)
        
        # Create enhanced history context
        history_context = self._create_enhanced_history_context(relevant_history, query_analysis)
        
        # Combine with base context
        enhanced_context = self._combine_contexts(base_context, history_context, query_analysis)
        
        return enhanced_context

    def _analyze_query_for_followup(self, query: str, session_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Simplified follow-up analysis using the simple rule:
        - If no history exists, it's a new query
        - If history exists, it's a follow-up query
        """
        # Simple rule: if we have session history, it's a follow-up
        is_followup = len(session_history) > 0 if session_history else False

        analysis = {
            'is_likely_followup': is_followup,
            'has_references': is_followup,  # If it's a follow-up, assume it may reference previous results
            'followup_indicators': ['session_history'] if is_followup else [],
            'reference_phrases': ['previous_context'] if is_followup else [],
            'query_type': 'followup' if is_followup else 'new',
            'contains_pronouns': is_followup,  # Assume follow-ups may use pronouns
            'temporal_references': ['previous'] if is_followup else []
        }

        return analysis
    
    def _analyze_query_for_followup_old(self, query: str) -> Dict[str, Any]:
        """Analyze query to identify follow-up patterns and references"""
        query_lower = query.lower()
        
        analysis = {
            'is_likely_followup': False,
            'has_references': False,
            'followup_indicators': [],
            'reference_phrases': [],
            'query_type': 'new',  # 'new', 'followup', 'reference', 'comparison'
            'contains_pronouns': False,
            'temporal_references': []
        }
        
        # Check for follow-up keywords
        for keyword in self.config['follow_up_keywords']:
            if keyword in query_lower:
                analysis['followup_indicators'].append(keyword)
                analysis['is_likely_followup'] = True
        
        # Check for reference keywords
        for ref_keyword in self.config['reference_keywords']:
            if ref_keyword in query_lower:
                analysis['reference_phrases'].append(ref_keyword)
                analysis['has_references'] = True
        
        # Check for pronouns that likely refer to previous content
        pronouns = ['it', 'this', 'that', 'them', 'those', 'these', 'they']
        for pronoun in pronouns:
            if f' {pronoun} ' in f' {query_lower} ' or query_lower.startswith(f'{pronoun} '):
                analysis['contains_pronouns'] = True
                break
        
        # Check for temporal references
        temporal_words = ['previous', 'last', 'earlier', 'before', 'above', 'recent', 'latest']
        for temp_word in temporal_words:
            if temp_word in query_lower:
                analysis['temporal_references'].append(temp_word)
        
        # Determine query type
        if analysis['has_references'] or analysis['contains_pronouns']:
            analysis['query_type'] = 'reference'
        elif 'compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower:
            analysis['query_type'] = 'comparison'
        elif analysis['is_likely_followup']:
            analysis['query_type'] = 'followup'
        
        logger.debug(f"Query analysis: {analysis}")
        return analysis
    
    def _filter_relevant_history(self, session_history: List[Dict], current_query: str, 
                                query_analysis: Dict) -> List[Dict]:
        """Filter and rank history entries based on relevance to current query"""
        if not session_history:
            return []
        
        # Always include recent entries for context continuity
        recent_count = min(self.config['recent_entries_weight'], len(session_history))
        recent_entries = session_history[-recent_count:]
        
        # If it's clearly a follow-up query, emphasize recent history
        if query_analysis['query_type'] in ['reference', 'followup', 'comparison']:
            # Take more recent entries for follow-up queries
            max_entries = min(self.config['max_history_entries'], len(session_history))
            return session_history[-max_entries:]
        
        # For new queries, take standard recent history
        max_entries = min(self.config['max_history_entries'], len(session_history))
        return session_history[-max_entries:]
    
    def _create_enhanced_history_context(self, relevant_history: List[Dict], 
                                       query_analysis: Dict) -> str:
        """Create enhanced history context string"""
        if not relevant_history:
            return ""
        
        context_parts = []
        context_parts.append(f"\n\n{'='*60}")
        context_parts.append("SESSION CONTEXT - PREVIOUS ANALYSIS")
        context_parts.append(f"{'='*60}")
        
        # Add analysis type context
        if query_analysis['query_type'] == 'reference':
            context_parts.append(" User's query refers to previous results - provide context-aware response")
        elif query_analysis['query_type'] == 'comparison':
            context_parts.append(" User wants comparison - reference previous analysis for comparison")
        elif query_analysis['query_type'] == 'followup':
            context_parts.append(" User's query builds on previous analysis - maintain context continuity")
        
        context_parts.append(f"\nPREVIOUS {len(relevant_history)} INTERACTIONS:")
        
        # Process each history entry
        for i, entry in enumerate(relevant_history, 1):
            entry_age = self._calculate_entry_age(entry.get('timestamp'))
            context_parts.append(f"\n[{i}] {entry_age}")
            
            # Query
            query = entry.get('query', '')
            context_parts.append(f"USER ASKED: {query}")
            
            # Code (abbreviated for context)
            code = entry.get('code', '')
            if code:
                abbreviated_code = self._abbreviate_code(code)
                context_parts.append(f"CODE USED: {abbreviated_code}")
            
            # Result summary (enhanced)
            result_summary = entry.get('result_summary', '')
            if result_summary:
                enhanced_summary = self._enhance_result_summary(result_summary)
                context_parts.append(f"RESULT: {enhanced_summary}")
            
            # Add relevance markers for recent entries
            if i > len(relevant_history) - self.config['recent_entries_weight']:
                context_parts.append("   RECENT - Highly relevant for follow-up questions")
        
        context_parts.append(f"\n{'='*60}")
        context_parts.append("CONTEXT USAGE GUIDELINES:")
        
        if query_analysis['contains_pronouns']:
            context_parts.append("- User uses pronouns ('it', 'this', 'that') - resolve references to previous results")
        
        if query_analysis['has_references']:
            context_parts.append("- User references previous analysis - connect to specific prior results")
        
        if query_analysis['temporal_references']:
            context_parts.append(f"- User mentions temporal references: {query_analysis['temporal_references']}")
        
        context_parts.append("- Maintain consistency with previous analysis approaches")
        context_parts.append("- Build upon previous insights rather than repeating them")
        context_parts.append("- Reference specific previous results when relevant")
        
        context_parts.append(f"{'='*60}")
        
        return "\n".join(context_parts)
    
    def _combine_contexts(self, base_context: str, history_context: str, 
                         query_analysis: Dict) -> str:
        """Combine base context with enhanced history context"""
        if not history_context:
            return base_context
        
        # Add follow-up specific instructions
        followup_instructions = ""
        if query_analysis['query_type'] != 'new':
            followup_instructions = f"""

FOLLOW-UP QUERY DETECTED (Type: {query_analysis['query_type']})
- This query likely builds on or references previous analysis
- Use session context to understand references and maintain continuity
- When user says "it", "this", "that", refer to most recent relevant result
- Provide context-aware responses that acknowledge previous work
"""
        
        return base_context + history_context + followup_instructions
    
    def _calculate_entry_age(self, timestamp_str: Optional[str]) -> str:
        """Calculate and format how long ago an entry was made"""
        if not timestamp_str:
            return "Previously"
        
        try:
            entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now(entry_time.tzinfo) if entry_time.tzinfo else datetime.now()
            age = now - entry_time
            
            if age.total_seconds() < 60:
                return "Just now"
            elif age.total_seconds() < 3600:
                minutes = int(age.total_seconds() / 60)
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif age.total_seconds() < 86400:
                hours = int(age.total_seconds() / 3600)
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                days = age.days
                return f"{days} day{'s' if days != 1 else ''} ago"
        except Exception:
            return "Previously"
    
    def _abbreviate_code(self, code: str) -> str:
        """Abbreviate code for context while keeping key operations visible"""
        if not code:
            return ""
        
        lines = code.strip().split('\n')
        
        # If short enough, keep as is
        if len(lines) <= self.config['max_code_lines_in_context']:
            return code.strip()
        
        # Extract key lines (assignments, function calls, etc.)
        key_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Keep important lines
            if any(keyword in stripped for keyword in ['=', 'df', 'pd.', 'plt.', 'fig', 'result']):
                key_lines.append(stripped)
                if len(key_lines) >= self.config['max_code_lines_in_context'] - 1:
                    break
        
        if len(key_lines) < len(lines):
            key_lines.append(f"... ({len(lines) - len(key_lines)} more lines)")
        
        return '\n'.join(key_lines)
    
    def _enhance_result_summary(self, result_summary: str) -> str:
        """Enhance result summary for better context"""
        if not result_summary:
            return "No result summary available"
        
        # Truncate if too long
        if len(result_summary) > self.config['max_result_summary_length']:
            result_summary = result_summary[:self.config['max_result_summary_length']] + "..."
        
        # Add structure indicators
        if 'table' in result_summary.lower() or 'dataframe' in result_summary.lower():
            result_summary = f"📊 {result_summary}"
        elif 'chart' in result_summary.lower() or 'plot' in result_summary.lower():
            result_summary = f"📈 {result_summary}"
        elif any(word in result_summary.lower() for word in ['number', 'count', 'sum', 'average']):
            result_summary = f"🔢 {result_summary}"
        
        return result_summary
    
    def create_follow_up_hints(self, session_history: List[Dict]) -> List[str]:
        """Generate suggested follow-up questions based on session history"""
        if not session_history:
            return []
        
        hints = []
        recent_entry = session_history[-1] if session_history else None
        
        if recent_entry:
            query = recent_entry.get('query', '').lower()
            result = recent_entry.get('result_summary', '').lower()
            
            # Suggest related analyses based on last query
            if 'correlation' in query:
                hints.append("Show me a scatter plot of the most correlated variables")
                hints.append("What other variables are strongly correlated?")
            
            elif 'summary' in query or 'describe' in query:
                hints.append("Create a visualization of this data")
                hints.append("Show me the distribution of the numeric columns")
            
            elif 'plot' in query or 'chart' in query:
                hints.append("Show me the underlying data for this chart")
                hints.append("Create a different type of visualization")
            
            elif 'filter' in query or 'where' in query:
                hints.append("Show me summary statistics for this filtered data")
                hints.append("Compare this subset with the full dataset")
            
            # Generic follow-ups
            if len(session_history) >= 2:
                hints.append("Compare this with the previous analysis")
                hints.append("What's the relationship between these results?")
        
        return hints[:3]  # Return top 3 hints


# Global enhancer instance
_enhancer = SessionContextEnhancer()

############################### New additions
def is_follow_up_query(query: str, session_history: List[Dict]) -> bool:
    """
    Simplified follow-up detection: any query after the first is a follow-up

    Args:
        query: Current user query
        session_history: List of previous interactions

    Returns:
        Boolean indicating if this is a follow-up query
    """
    return len(session_history) > 0  # Simple rule: if history exists, it's a follow-up

def is_follow_up_query_old(query: str, session_history: List[Dict]) -> bool:
    """
    Detect if a query is a follow-up to previous questions

    Args:
        query: Current user query
        session_history: List of previous interactions

    Returns:
        Boolean indicating if this is likely a follow-up query
    """
    if not session_history:
        return False

    global _enhancer
    analysis = _enhancer._analyze_query_for_followup(query)

    # Strong indicators of follow-up
    if analysis['query_type'] in ['reference', 'followup', 'comparison']:
        return True

    if analysis['contains_pronouns'] or analysis['has_references']:
        return True

    if analysis['temporal_references']:
        return True

    # Check if query references specific values from last result
    if session_history:
        last_result = session_history[-1].get('result_summary', '')
        # Extract potential values from last result (numbers, percentages, names)
        values = re.findall(r'\b(?:\d+\.?\d*%?|\$[\d,]+\.?\d*|[A-Z][a-z]+)\b', last_result)
        query_lower = query.lower()
        for value in values[:100]:  # Check first 100 values
            if value.lower() in query_lower:
                return True

    return False


def create_result_summary(result: Any, max_length: int = 1000) -> str:
    """
    Create a comprehensive result summary for session context

    Args:
        result: The result from agent processing
        max_length: Maximum length of summary (default 1000)

    Returns:
        String summary of the result
    """
    # Import pandas at the function level to handle DataFrames/Series
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if result is None:
        return "No result returned"

    if isinstance(result, str):
        # For string results, preserve more content
        if len(result) <= max_length:
            return result
        else:
            # Smart truncation - try to break at sentence boundary
            truncated = result[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # If we have a period in last 30%
                return truncated[:last_period + 1]
            return truncated + "..."

    elif isinstance(result, (int, float)):
        return f"Numeric value: {result:,.2f}" if isinstance(result, float) else f"Numeric value: {result:,}"

    elif isinstance(result, dict):
        # Summarize dictionary structure and content
        if not result:
            return "Empty dictionary"

        summary_parts = []
        total_keys = len(result)

        for i, (key, value) in enumerate(list(result.items())):
            if isinstance(value, (list, dict)):
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # List of dicts (common for table data)
                    summary_parts.append(f"{key}: Table with {len(value)} rows")
                else:
                    summary_parts.append(f"{key}: {type(value).__name__} with {len(value)} items")
            elif isinstance(value, (int, float)):
                formatted_val = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                summary_parts.append(f"{key}: {formatted_val}")
            else:
                val_str = str(value)
                summary_parts.append(f"{key}: {val_str}")

        summary = f"Dictionary with {total_keys} keys: " + "; ".join(summary_parts)

        return summary

    elif isinstance(result, list):
        if not result:
            return "Empty list"

        # Check if it's a list of dictionaries (table data)
        if isinstance(result[0], dict):
            keys = list(result[0].keys())
            num_rows = len(result)
            num_cols = len(keys)

            # Create informative summary
            summary = f"Table with {num_rows} rows and {num_cols} columns"

            # Add column names (up to 30)
            if keys:
                cols_preview = keys[:30]
                summary += f"\nColumns: {', '.join(cols_preview)}"
                if len(keys) > 30:
                    summary += f" ... and {len(keys) - 30} more"

            # Add sample of first 5 rows
            rows_to_show = min(5, len(result))  # Show up to 5 rows
            if rows_to_show > 0:
                summary += f"\n\nFirst {rows_to_show} rows:"

                for row_idx in range(rows_to_show):
                    row = result[row_idx]
                    row_items = []

                    # Show up to 10 key-value pairs per row
                    items_to_show = min(10, len(keys))
                    for k in keys[:items_to_show]:
                        v = row.get(k, '')
                        # Format value based on type
                        if isinstance(v, float):
                            val_str = f"{v:.2f}"
                        elif isinstance(v, int):
                            val_str = f"{v:,}" if v > 999 else str(v)
                        else:
                            val_str = str(v)[:50]  # Limit string length
                        row_items.append(f"{k}={val_str}")

                    row_summary = f"  Row {row_idx + 1}: {', '.join(row_items)}"
                    if len(keys) > items_to_show:
                        row_summary += f" ... +{len(keys) - items_to_show} more fields"
                    summary += f"\n{row_summary}"

                if num_rows > rows_to_show:
                    summary += f"\n  ... and {num_rows - rows_to_show} more rows"

            return summary

        # Regular list
        elif isinstance(result[0], (int, float, str)):
            sample_size = min(5, len(result))
            sample_items = result[:sample_size]

            # Format based on type
            if isinstance(result[0], float):
                sample_str = ', '.join([f"{x:.2f}" for x in sample_items])
            elif isinstance(result[0], int):
                sample_str = ', '.join([f"{x:,}" for x in sample_items])
            else:
                sample_str = ', '.join([str(x)[:30] for x in sample_items])

            summary = f"List with {len(result)} items: [{sample_str}"
            if len(result) > sample_size:
                summary += f", ... and {len(result) - sample_size} more"
            summary += "]"

            return summary

        else:
            # Complex objects in list
            return f"List with {len(result)} {type(result[0]).__name__} objects"

    elif pd and isinstance(result, pd.DataFrame):
        shape = result.shape
        cols = list(result.columns)
        summary = f"DataFrame with {shape[0]} rows and {shape[1]} columns"
        summary += f"\nColumns: {', '.join(map(str, cols))}"

        # Add first 5 rows of actual data
        rows_to_show = min(5, len(result))
        if rows_to_show > 0:
            summary += f"\n\nFirst {rows_to_show} rows:"

            # Convert first 5 rows to dict format for easier processing
            sample_data = result.head(rows_to_show).to_dict('records')

            for idx, row_dict in enumerate(sample_data):
                row_items = []
                cols_to_show = len(cols)  # Show up to 4 columns per row

                for col in cols[:cols_to_show]:
                    val = row_dict.get(col)

                    # Format value based on type
                    if pd.isna(val):
                        val_str = "NaN"
                    elif isinstance(val, float):
                        val_str = f"{val:.2f}"
                    elif isinstance(val, (int, np.integer)):
                        val_str = f"{val:,}" if abs(val) > 999 else str(val)
                    elif isinstance(val, bool):
                        val_str = str(val)
                    else:
                        val_str = str(val)[:50]  # Truncate long strings

                    row_items.append(f"{col}={val_str}")

                row_summary = f"  Row {idx + 1}: {', '.join(row_items)}"
                if len(result.columns) > cols_to_show:
                    row_summary += f" ... +{len(result.columns) - cols_to_show} more cols"
                summary += f"\n{row_summary}"

            if shape[0] > rows_to_show:
                summary += f"\n  ... and {shape[0] - rows_to_show} more rows"

        return summary

    elif pd and isinstance(result, pd.Series):
        summary = f"Series '{result.name}' with {len(result)} values, dtype: {result.dtype}"

        # Show first 20 values with their index
        sample_size = min(20, len(result))
        if sample_size > 0:
            summary += f"\n\nFirst {sample_size} values:"

            for idx in range(sample_size):
                index_label = result.index[idx]
                value = result.iloc[idx]

                # Format value
                if pd.isna(value):
                    val_str = "NaN"
                elif isinstance(value, float):
                    val_str = f"{value:.2f}"
                elif isinstance(value, (int, np.integer)):
                    val_str = f"{value:,}" if abs(value) > 999 else str(value)
                else:
                    val_str = str(value)[:50]

                summary += f"\n  {index_label}: {val_str}"

            if len(result) > sample_size:
                summary += f"\n  ... and {len(result) - sample_size} more values"

        return summary

    else:
        # Generic fallback
        result_str = str(result)
        if len(result_str) <= max_length:
            return result_str

        return result_str[:max_length] + "..."

#####################################################

def enhance_session_context(base_context: str, session_history: List[Dict], 
                          current_query: str, config=None) -> str:
    """
    Main function to enhance session context
    
    Args:
        base_context: The base data context string
        session_history: List of previous session interactions
        current_query: The current user query
        config: Optional configuration dict
        
    Returns:
        Enhanced context string
    """
    global _enhancer
    if config:
        _enhancer = SessionContextEnhancer(config)
    
    return _enhancer.enhance_session_context(base_context, session_history, current_query)


def analyze_query_context_needs(query: str) -> Dict[str, Any]:
    """
    Analyze what kind of context a query needs
    
    Args:
        query: The user query to analyze
        
    Returns:
        Analysis dict with context needs
    """
    return _enhancer._analyze_query_for_followup(query)


def get_follow_up_suggestions(session_history: List[Dict]) -> List[str]:
    """
    Get suggested follow-up questions based on session history
    
    Args:
        session_history: List of previous session interactions
        
    Returns:
        List of suggested follow-up questions
    """
    return _enhancer.create_follow_up_hints(session_history)

