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

logger = logging.getLogger(__name__)

class SessionContextEnhancer:
    """Enhances session context for better follow-up question support"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration for session context enhancement"""
        return {
            'max_history_entries': 10,
            'recent_entries_weight': 5,  # Number of most recent entries to emphasize
            'context_relevance_threshold': 0.3,
            'max_code_lines_in_context': 50,
            'max_result_summary_length': 200,
            'follow_up_keywords': [
                'also', 'additionally', 'furthermore', 'what about', 'how about',
                'compare', 'versus', 'vs', 'difference', 'similar', 'like that',
                'same', 'previous', 'last', 'earlier', 'before', 'above',
                'it', 'this', 'that', 'them', 'those', 'these'
            ],
            'reference_keywords': [
                'that analysis', 'the chart', 'the table', 'the result',
                'that result', 'those findings', 'the data', 'that data'
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
        query_analysis = self._analyze_query_for_followup(current_query)
        
        # Filter and rank history entries based on relevance
        relevant_history = self._filter_relevant_history(session_history, current_query, query_analysis)
        
        # Create enhanced history context
        history_context = self._create_enhanced_history_context(relevant_history, query_analysis)
        
        # Combine with base context
        enhanced_context = self._combine_contexts(base_context, history_context, query_analysis)
        
        return enhanced_context
    
    def _analyze_query_for_followup(self, query: str) -> Dict[str, Any]:
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
            context_parts.append("🔗 User's query refers to previous results - provide context-aware response")
        elif query_analysis['query_type'] == 'comparison':
            context_parts.append("⚖️ User wants comparison - reference previous analysis for comparison")
        elif query_analysis['query_type'] == 'followup':
            context_parts.append("📈 User's query builds on previous analysis - maintain context continuity")
        
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
                context_parts.append("   ⭐ RECENT - Highly relevant for follow-up questions")
        
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
