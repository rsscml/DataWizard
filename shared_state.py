"""
Updated Shared state manager for the Excel Analytics Platform
Now session-aware to support multiple users
"""

import threading
from typing import Optional, Dict, Any
import pandas as pd

class SharedState:
    """Session-aware shared state manager"""
    
    def __init__(self):
        # Import here to avoid circular imports
        pass
        
    def _get_session(self):
        """Get current user session"""
        from session_manager import session_manager
        return session_manager.get_current_session()
    
    def update_data(self, data: pd.DataFrame, filename: str, summary: Dict, 
                   document_key: str, document_url: str):
        """Update current data and related information"""
        session = self._get_session()
        session.current_data = data
        session.current_filename = filename
        session.data_summary = summary
        session.current_document_key = document_key
        session.current_document_url = document_url
        
        # Update worksheet data for single sheet
        session.worksheet_data = {'Sheet1': data}
        session.worksheet_summaries = {'Sheet1': summary}
        session.active_worksheet = 'Sheet1'
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        return self._get_session().current_data
    
    def get_current_filename(self) -> Optional[str]:
        return self._get_session().current_filename
            
    def get_data_summary(self) -> Optional[Dict]:
        return self._get_session().data_summary
            
    def get_worksheet_data(self) -> Dict[str, pd.DataFrame]:
        return self._get_session().worksheet_data.copy()
            
    def get_worksheet_summaries(self) -> Dict[str, Dict]:
        return self._get_session().worksheet_summaries.copy()
            
    def get_active_worksheet(self) -> Optional[str]:
        return self._get_session().active_worksheet
            
    def get_document_info(self) -> tuple:
        session = self._get_session()
        return session.current_document_key, session.current_document_url
            
    def reset(self):
        """Reset all state for current session"""
        self._get_session().reset()

# Global shared state instance (now session-aware)
shared_state = SharedState()
