"""
File Cleanup Manager for Excel Analytics Platform
Handles automatic cleanup of uploaded files, documents, and downloads
"""

import os
import json
import time
import threading
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

class FileCleanupManager:
    """Manages automatic cleanup of application files"""
    
    def __init__(self, app):
        self.app = app
        self.upload_folder = Path(app.config['UPLOAD_FOLDER'])
        self.download_folder = Path(app.config['DOWNLOAD_FOLDER']) 
        self.documents_folder = Path(app.config['DOCUMENTS_FOLDER'])
        self.session_files_db = Path('session_files.json')
        self.cleanup_lock = threading.Lock()
        
        # Configuration
        self.max_file_age_hours = 24  # Files older than 24h get deleted
        self.emergency_cleanup_threshold = 90  # Trigger emergency cleanup at 90% disk usage
        self.warning_threshold = 80  # Warn at 80% disk usage
        
        # Ensure directories exist
        for folder in [self.upload_folder, self.download_folder, self.documents_folder]:
            folder.mkdir(exist_ok=True)
    
    def associate_file_with_session(self, file_path: str, session_id: str, file_type: str = 'upload'):
        """Track which files belong to which session"""
        with self.cleanup_lock:
            try:
                session_files = self._load_session_files()
                
                if session_id not in session_files:
                    session_files[session_id] = []
                
                session_files[session_id].append({
                    'file_path': str(file_path),
                    'file_type': file_type,  # 'upload', 'download', 'document', 'temp'
                    'created_at': time.time(),
                    'last_accessed': time.time()
                })
                
                self._save_session_files(session_files)
                logger.debug(f"Associated file {file_path} with session {session_id}")
                
            except Exception as e:
                logger.error(f"Error associating file with session: {e}")
    
    def update_file_access(self, file_path: str, session_id: str):
        """Update last access time for a file"""
        with self.cleanup_lock:
            try:
                session_files = self._load_session_files()
                
                if session_id in session_files:
                    for file_info in session_files[session_id]:
                        if file_info['file_path'] == str(file_path):
                            file_info['last_accessed'] = time.time()
                            break
                    
                    self._save_session_files(session_files)
                    
            except Exception as e:
                logger.error(f"Error updating file access: {e}")
    
    def cleanup_session_files(self, session_id: str):
        """Clean up all files associated with an expired session"""
        with self.cleanup_lock:
            try:
                session_files = self._load_session_files()
                
                if session_id in session_files:
                    deleted_count = 0
                    for file_info in session_files[session_id]:
                        file_path = Path(file_info['file_path'])
                        if file_path.exists():
                            try:
                                file_path.unlink()
                                deleted_count += 1
                                logger.info(f"Cleaned up session file: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to delete session file {file_path}: {e}")
                    
                    del session_files[session_id]
                    self._save_session_files(session_files)
                    logger.info(f"Cleaned up {deleted_count} files for session {session_id}")
                    
            except Exception as e:
                logger.error(f"Error cleaning session files: {e}")
    
    def cleanup_old_files(self, max_age_hours: int = None):
        """Clean up files older than specified age across all directories"""
        if max_age_hours is None:
            max_age_hours = self.max_file_age_hours
            
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        directories_to_clean = [
            (self.upload_folder, "uploads"),
            (self.download_folder, "downloads"), 
            (self.documents_folder, "documents")
        ]
        
        total_deleted = 0
        total_size_freed = 0
        
        for directory, dir_name in directories_to_clean:
            if directory.exists():
                deleted, size_freed = self._clean_directory(directory, cutoff_time, dir_name)
                total_deleted += deleted
                total_size_freed += size_freed
        
        # Clean orphaned entries from session files DB
        self._clean_orphaned_session_entries()
        
        if total_deleted > 0:
            logger.info(f"Cleanup complete: {total_deleted} files deleted, {self._format_size(total_size_freed)} freed")
        
        return total_deleted, total_size_freed
    
    def emergency_cleanup(self):
        """Emergency cleanup when disk space is critically low"""
        logger.warning("Initiating emergency cleanup due to low disk space")
        
        # Clean files older than 1 hour in emergency
        self.cleanup_old_files(max_age_hours=1)
        
        # Clean up any remaining large files
        self._clean_large_files()
        
        # Force cleanup of all temp files
        self._clean_temp_files()
    
    def check_disk_space(self) -> Dict[str, float]:
        """Check disk space and trigger cleanup if needed"""
        try:
            total, used, free = shutil.disk_usage(self.upload_folder)
            
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            used_percent = (used / total) * 100
            
            disk_info = {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'used_percent': used_percent
            }
            
            if used_percent > self.emergency_cleanup_threshold:
                logger.critical(f"DISK SPACE CRITICAL: {used_percent:.1f}% used, initiating emergency cleanup")
                self.emergency_cleanup()
            elif used_percent > self.warning_threshold:
                logger.warning(f"DISK SPACE LOW: {used_percent:.1f}% used")
            
            return disk_info
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return {}
    
    def get_cleanup_stats(self) -> Dict:
        """Get statistics about files and cleanup"""
        try:
            stats = {
                'total_files': 0,
                'total_size_mb': 0,
                'sessions_tracked': 0,
                'old_files_count': 0
            }
            
            cutoff_time = time.time() - (self.max_file_age_hours * 3600)
            
            for directory in [self.upload_folder, self.download_folder, self.documents_folder]:
                if directory.exists():
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            stats['total_files'] += 1
                            stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                            
                            if file_path.stat().st_mtime < cutoff_time:
                                stats['old_files_count'] += 1
            
            session_files = self._load_session_files()
            stats['sessions_tracked'] = len(session_files)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cleanup stats: {e}")
            return {}
    
    def _clean_directory(self, directory: Path, cutoff_time: float, dir_name: str):
        """Clean files older than cutoff_time in directory"""
        deleted_count = 0
        size_freed = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        size_freed += file_size
                        logger.debug(f"Deleted old {dir_name} file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
        
        return deleted_count, size_freed
    
    def _clean_large_files(self):
        """Clean up unusually large files during emergency"""
        large_file_threshold = 100 * 1024 * 1024  # 100MB
        
        for directory in [self.upload_folder, self.download_folder, self.documents_folder]:
            if directory.exists():
                for file_path in directory.rglob('*'):
                    if file_path.is_file() and file_path.stat().st_size > large_file_threshold:
                        try:
                            file_path.unlink()
                            logger.warning(f"Emergency cleanup: deleted large file {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete large file {file_path}: {e}")
    
    def _clean_temp_files(self):
        """Clean up temporary files"""
        temp_patterns = ['temp_*', '*_temp.*', 'combined_csvs_*', '*_cleaned_*']
        
        for directory in [self.upload_folder, self.documents_folder]:
            if directory.exists():
                for pattern in temp_patterns:
                    for file_path in directory.glob(pattern):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                logger.debug(f"Cleaned temp file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete temp file {file_path}: {e}")
    
    def _clean_orphaned_session_entries(self):
        """Remove entries from session files DB where files no longer exist"""
        try:
            session_files = self._load_session_files()
            modified = False
            
            for session_id in list(session_files.keys()):
                files_list = session_files[session_id]
                existing_files = []
                
                for file_info in files_list:
                    if Path(file_info['file_path']).exists():
                        existing_files.append(file_info)
                    else:
                        modified = True
                
                if existing_files:
                    session_files[session_id] = existing_files
                else:
                    del session_files[session_id]
                    modified = True
            
            if modified:
                self._save_session_files(session_files)
                logger.debug("Cleaned orphaned session file entries")
                
        except Exception as e:
            logger.error(f"Error cleaning orphaned session entries: {e}")
    
    def _load_session_files(self) -> Dict:
        """Load session files database"""
        try:
            if self.session_files_db.exists():
                with open(self.session_files_db, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session files DB: {e}")
        
        return {}
    
    def _save_session_files(self, session_files: Dict):
        """Save session files database"""
        try:
            with open(self.session_files_db, 'w') as f:
                json.dump(session_files, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session files DB: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

def setup_file_cleanup(app):
    """Setup automatic file cleanup worker"""
    cleanup_manager = FileCleanupManager(app)
    
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Run every hour
            try:
                cleanup_manager.cleanup_old_files()
                cleanup_manager.check_disk_space()
            except Exception as e:
                logger.error(f"File cleanup worker error: {e}")
    
    # Start cleanup worker thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("File cleanup worker started")
    
    return cleanup_manager