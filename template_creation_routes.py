import json
import os
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import render_template, request, jsonify, send_from_directory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_template_management_routes(app):
    """Register all template-related routes with the Flask app"""

    @app.route('/template-creator')
    def template_creator():
        """Serve the template creator utility page"""
        return render_template('json_template_creator.html')
        
    @app.route('/template-manager')
    def template_manager_page():
        """Serve the template management page"""
        return render_template('template_manager.html')
        
    @app.route('/api/templates/files', methods=['GET'])
    def list_template_files():
        """Get list of all template files in the data_templates directory"""
        try:
            templates_dir = Path(app.config.get('TEMPLATES_DIR', 'data_templates'))
            templates_dir.mkdir(exist_ok=True)
            template_files = []
            for file_path in templates_dir.glob('*.json'):
                try:
                    # Get file stats
                    stat = file_path.stat()
                    file_size = stat.st_size
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Try to read template info
                    template_info = {'name': 'Unknown', 'description': 'No description', 'domain': 'Unknown'}
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            template_info = {
                                'name': data.get('name', 'Unknown'),
                                'description': data.get('description', 'No description'),
                                'domain': data.get('domain', 'Unknown'),
                                'version': data.get('version', '1.0'),
                                'template_type': data.get('template_type', 'single_worksheet')
                            }
                    except Exception as e:
                        logger.warning(f"Could not parse template file {file_path.name}: {e}")
                    
                    template_files.append({
                        'filename': file_path.name,
                        'size': file_size,
                        'size_formatted': format_file_size(file_size),
                        'modified': modified_time.isoformat(),
                        'modified_formatted': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'template_info': template_info
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing template file {file_path.name}: {e}")
                    continue
            
            # Sort by modification time (newest first)
            template_files.sort(key=lambda x: x['modified'], reverse=True)
            
            return jsonify({
                'success': True,
                'files': template_files,
                'count': len(template_files),
                'templates_dir': str(templates_dir)
            })
            
        except Exception as e:
            logger.error(f"Error listing template files: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


    @app.route('/api/templates/upload', methods=['POST'])
    def upload_template_file():
        """Upload a new template file to the data_templates directory"""
        try:
            if 'template_file' not in request.files:
                return jsonify({
                'success': False,
                'error': 'No file uploaded'
                }), 400
            file = request.files['template_file']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Validate file extension
            if not file.filename.lower().endswith('.json'):
                return jsonify({
                    'success': False,
                    'error': 'Only JSON files are allowed'
                }), 400
            
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename.endswith('.json'):
                filename += '.json'
            
            templates_dir = Path(app.config.get('TEMPLATES_DIR', 'data_templates'))
            templates_dir.mkdir(exist_ok=True)
            
            file_path = templates_dir / filename
            
            # Check if file already exists
            if file_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'Template file "{filename}" already exists'
                }), 409
            
            # Read and validate JSON content
            try:
                file_content = file.read()
                file.seek(0)  # Reset file pointer
                
                # Validate JSON format
                json_data = json.loads(file_content.decode('utf-8'))
                
                # Basic template validation
                required_fields = ['id', 'name', 'description']
                missing_fields = [field for field in required_fields if field not in json_data]
                if missing_fields:
                    return jsonify({
                        'success': False,
                        'error': f'Template missing required fields: {", ".join(missing_fields)}'
                    }), 400
                
            except json.JSONDecodeError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid JSON format: {str(e)}'
                }), 400
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error validating template: {str(e)}'
                }), 400
            
            # Save the file
            file.save(file_path)
            
            # Reload templates in the template manager
            try:
                template_manager.load_templates()
                logger.info(f"Template file uploaded and loaded: {filename}")
            except Exception as e:
                logger.warning(f"Template uploaded but failed to load: {e}")
            
            return jsonify({
                'success': True,
                'message': f'Template "{filename}" uploaded successfully',
                'filename': filename
            })
            
        except Exception as e:
            logger.error(f"Error uploading template file: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
            
    @app.route('/api/templates/delete', methods=['POST'])
    def delete_template_files():
        """Delete one or more template files"""
        try:
            data = request.get_json()
            filenames = data.get('filenames', [])
            if not filenames:
                return jsonify({
                'success': False,
                'error': 'No filenames provided'
            }), 400
        
            templates_dir = Path(app.config.get('TEMPLATES_DIR', 'data_templates'))
            deleted_files = []
            failed_files = []
            
            for filename in filenames:
                try:
                    # Validate filename to prevent directory traversal
                    secure_name = secure_filename(filename)
                    if secure_name != filename or '..' in filename or '/' in filename:
                        failed_files.append({
                            'filename': filename,
                            'error': 'Invalid filename'
                        })
                        continue
                    
                    file_path = templates_dir / filename
                    
                    # Check if file exists and is a JSON file
                    if not file_path.exists():
                        failed_files.append({
                            'filename': filename,
                            'error': 'File does not exist'
                        })
                        continue
                    
                    if not filename.lower().endswith('.json'):
                        failed_files.append({
                            'filename': filename,
                            'error': 'Not a JSON template file'
                        })
                        continue
                    
                    # Delete the file
                    file_path.unlink()
                    deleted_files.append(filename)
                    logger.info(f"Template file deleted: {filename}")
                    
                except Exception as e:
                    failed_files.append({
                        'filename': filename,
                        'error': str(e)
                    })
                    logger.error(f"Error deleting template file {filename}: {e}")
            
            # Reload templates in the template manager
            try:
                template_manager.load_templates()
                logger.info("Templates reloaded after deletion")
            except Exception as e:
                logger.warning(f"Templates deleted but failed to reload: {e}")
            
            return jsonify({
                'success': True,
                'deleted_files': deleted_files,
                'failed_files': failed_files,
                'message': f'Successfully deleted {len(deleted_files)} file(s)'
            })
            
        except Exception as e:
            logger.error(f"Error deleting template files: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            

    @app.route('/api/templates/download/<filename>')
    def download_template_file(filename):
        """Download a specific template file"""
        try:
            # Validate filename to prevent directory traversal
            secure_name = secure_filename(filename)
            if secure_name != filename or '..' in filename or '/' in filename:
                return jsonify({'error': 'Invalid filename'}), 400
                
            templates_dir = Path(app.config.get('TEMPLATES_DIR', 'data_templates'))
            file_path = templates_dir / filename
            
            if not file_path.exists() or not filename.lower().endswith('.json'):
                return jsonify({'error': 'Template file not found'}), 404
            
            return send_from_directory(
                str(templates_dir),
                filename,
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            logger.error(f"Error downloading template file {filename}: {e}")
            return jsonify({'error': str(e)}), 500
            
            
    def format_file_size(size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"
        
