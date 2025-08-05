"""
Session Manager for User Session Separation
===========================================

This module provides session-aware storage for all global variables,
allowing multiple users to have separate data without interfering with each other.
"""

import threading
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import uuid
from flask import session
import logging

logger = logging.getLogger(__name__)

class UserSession:
    """Stores all data for a single user session"""
    
    def __init__(self):
        # File upload mode data (previously global variables)
        self.current_data = None
        self.current_filename = None
        self.data_summary = None
        self.worksheet_data = {}
        self.worksheet_summaries = {}
        self.analyzable_worksheets = {}
        self.excluded_worksheets = {}
        self.active_worksheet = None
        self.current_document_key = None
        self.current_document_url = None
        
        # Session history
        self.session_history = []
        self.max_session_history = 20
        
        # Metadata
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.session_id = str(uuid.uuid4())

        # Template-related data
        self.detected_template = None
        self.template_validation_results = None
        self.manual_template_override = None
        
        logger.info(f"Created new user session: {self.session_id}")
    
    def update_access_time(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def reset(self):
        """Reset all session data"""
        self.current_data = None
        self.current_filename = None
        self.data_summary = None
        self.worksheet_data = {}
        self.worksheet_summaries = {}
        self.analyzable_worksheets = {}
        self.excluded_worksheets = {}
        self.active_worksheet = None
        self.current_document_key = None
        self.current_document_url = None
        self.session_history = []
        
        # Clear template data (add after existing resets)
        self.detected_template = None
        self.template_validation_results = None
        self.manual_template_override = None
        
        logger.info(f"Reset session data for: {self.session_id}")


class SessionManager:
    """Thread-safe session manager for user data separation"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, UserSession] = {}
        self._cleanup_threshold = timedelta(hours=24)  # Clean up after 24 hours
        self._max_sessions = 1000  # Prevent memory issues
    
    def get_current_session_id(self) -> str:
        """Get or create session ID for current Flask session"""
        if 'user_session_id' not in session:
            session['user_session_id'] = str(uuid.uuid4())
            session.permanent = True  # Make session persistent
        return session['user_session_id']
    
    def get_session(self, session_id: str = None) -> UserSession:
        """Get or create session for session_id"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = UserSession()
                # Cleanup old sessions if we have too many
                if len(self._sessions) > self._max_sessions:
                    self._cleanup_old_sessions_internal()
            
            user_session = self._sessions[session_id]
            user_session.update_access_time()
            return user_session
    
    def get_current_session(self) -> UserSession:
        """Get session for current Flask user"""
        return self.get_session()
    
    def _cleanup_old_sessions_internal(self):
        """Internal cleanup method (should be called with lock held)"""
        cutoff_time = datetime.now() - self._cleanup_threshold
        old_sessions = [sid for sid, session in self._sessions.items() 
                       if session.last_accessed < cutoff_time]

        # Clean up files for old sessions
        for sid in old_sessions:
            try:
                # Import here to avoid circular imports
                from file_cleanup_manager import file_cleanup_manager
                if 'file_cleanup_manager' in globals():
                    file_cleanup_manager.cleanup_session_files(sid)
            except Exception as e:
                logger.error(f"Error cleaning files for session {sid}: {e}")

            del self._sessions[sid]

        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions and their files")
    
    def cleanup_old_sessions(self):
        """Remove old sessions to prevent memory leaks"""
        with self._lock:
            self._cleanup_old_sessions_internal()
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self._lock:
            return len(self._sessions)
    
    def reset_session(self, session_id: str = None):
        """Reset specific session or current session"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        with self._lock:
            if session_id in self._sessions:
                # Clean up files for this session
                try:
                    from file_cleanup_manager import file_cleanup_manager
                    if 'file_cleanup_manager' in globals():
                        file_cleanup_manager.cleanup_session_files(session_id)
                except Exception as e:
                    logger.error(f"Error cleaning files during session reset: {e}")
                    
                self._sessions[session_id].reset()

# Global session manager instance
session_manager = SessionManager()


# Helper functions to maintain compatibility with existing code
def get_current_data():
    """Get current_data for current session"""
    return session_manager.get_current_session().current_data

def set_current_data(data):
    """Set current_data for current session"""
    session_manager.get_current_session().current_data = data

def get_current_filename():
    """Get current_filename for current session"""
    return session_manager.get_current_session().current_filename

def set_current_filename(filename):
    """Set current_filename for current session"""
    session_manager.get_current_session().current_filename = filename

def get_data_summary():
    """Get data_summary for current session"""
    return session_manager.get_current_session().data_summary

def set_data_summary(summary):
    """Set data_summary for current session"""
    session_manager.get_current_session().data_summary = summary

def get_worksheet_data():
    """Get worksheet_data for current session"""
    return session_manager.get_current_session().worksheet_data

def set_worksheet_data(data):
    """Set worksheet_data for current session"""
    session_manager.get_current_session().worksheet_data = data

def get_worksheet_summaries():
    """Get worksheet_summaries for current session"""
    return session_manager.get_current_session().worksheet_summaries

def set_worksheet_summaries(summaries):
    """Set worksheet_summaries for current session"""
    session_manager.get_current_session().worksheet_summaries = summaries

def get_analyzable_worksheets():
    """Get analyzable_worksheets for current session"""
    return session_manager.get_current_session().analyzable_worksheets

def set_analyzable_worksheets(worksheets):
    """Set analyzable_worksheets for current session"""
    session_manager.get_current_session().analyzable_worksheets = worksheets

def get_excluded_worksheets():
    """Get excluded_worksheets for current session"""
    return session_manager.get_current_session().excluded_worksheets

def set_excluded_worksheets(worksheets):
    """Set excluded_worksheets for current session"""
    session_manager.get_current_session().excluded_worksheets = worksheets

def get_active_worksheet():
    """Get active_worksheet for current session"""
    return session_manager.get_current_session().active_worksheet

def set_active_worksheet(worksheet):
    """Set active_worksheet for current session"""
    session_manager.get_current_session().active_worksheet = worksheet

def get_current_document_key():
    """Get current_document_key for current session"""
    return session_manager.get_current_session().current_document_key

def set_current_document_key(key):
    """Set current_document_key for current session"""
    session_manager.get_current_session().current_document_key = key

def get_current_document_url():
    """Get current_document_url for current session"""
    return session_manager.get_current_session().current_document_url

def set_current_document_url(url):
    """Set current_document_url for current session"""
    session_manager.get_current_session().current_document_url = url

def get_session_history():
    """Get session_history for current session"""
    return session_manager.get_current_session().session_history

def set_session_history(history):
    """Set session_history for current session"""
    session_manager.get_current_session().session_history = history

def add_to_session_history(entry):
    """Add entry to session history"""
    session = session_manager.get_current_session()
    session.session_history.append(entry)
    if len(session.session_history) > session.max_session_history:
        session.session_history = session.session_history[-session.max_session_history:]

def clear_session_history():
    """Clear session history for current session"""
    session_manager.get_current_session().session_history = []

def reset_current_session():
    """Reset all data for current session"""
    session_manager.reset_session()


# Automatic cleanup function for periodic maintenance
def setup_session_cleanup(app):
    """Setup automatic session cleanup"""
    import atexit
    import threading
    import time
    
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Clean up every hour
            try:
                session_manager.cleanup_old_sessions()
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    # Cleanup on app shutdown
    atexit.register(session_manager.cleanup_old_sessions)
    
    logger.info("Session cleanup worker started")
