import os
import logging
import requests
import random
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from datetime import datetime
import uuid
import subprocess

from video_processor import VideoProcessor
from utils import allowed_file, get_file_size, validate_request

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize video processor
video_processor = VideoProcessor(
    upload_folder=app.config['UPLOAD_FOLDER'],
    processed_folder=app.config['PROCESSED_FOLDER']
)

@app.route('/')
def index():
    """Main page with testing interface"""
    return render_template('index.html')

@app.route('/webhook/video/process', methods=['POST'])
def process_video_webhook():
    """
    Main webhook endpoint for n8n video processing
    Accepts video files and processing parameters
    """
    try:
        # Validate request
        validation_error = validate_request(request)
        if validation_error:
            logger.error(f"Request validation failed: {validation_error}")
            return jsonify({
                'success': False,
                'error': validation_error,
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        # Handle file upload
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file format. Supported formats: MP4, AVI, MOV, WebM',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{unique_id}_{name}{ext}"
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Get processing parameters
        processing_params = {
            'output_format': request.form.get('output_format', 'mp4'),
            'quality': request.form.get('quality', 'medium'),
            'resolution': request.form.get('resolution', None),
            'compress': request.form.get('compress', 'false').lower() == 'true'
        }

        logger.info(f"Processing video {unique_filename} with params: {processing_params}")

        # Process the video
        result = video_processor.process_video(unique_filename, processing_params)

        if result['success']:
            logger.info(f"Video processing completed successfully for {unique_filename}")
            return jsonify({
                'success': True,
                'job_id': unique_id,
                'original_file': filename,
                'processed_file': result['output_file'],
                'metadata': result['metadata'],
                'processing_time': result['processing_time'],
                'file_size_reduction': result.get('file_size_reduction'),
                'download_url': f"/download/{result['output_file']}",
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            logger.error(f"Video processing failed for {unique_filename}: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error'],
                'job_id': unique_id,
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in video processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/webhook/video/metadata', methods=['POST'])
def extract_metadata_webhook():
    """
    Webhook endpoint for extracting video metadata only
    """
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        file = request.files['video']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid or missing video file',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{unique_id}_{name}{ext}"
        
        # Save uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Extract metadata
        metadata = video_processor.extract_metadata(unique_filename)

        # Clean up temporary file
        os.remove(file_path)

        if metadata:
            return jsonify({
                'success': True,
                'metadata': metadata,
                'original_file': filename,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to extract video metadata',
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """
    Download processed video files
    """
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': 'File not found',
                'timestamp': datetime.utcnow().isoformat()
            }), 404
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Download failed',
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
@app.route('/webhook/execute-script', methods=['POST'])
def execute_script_webhook():
    """
    Webhook endpoint for executing bash scripts from n8n
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must contain JSON data',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        data = request.get_json()

        if 'script' not in data:
            return jsonify({
                'success': False,
                'error': 'No script provided',
                'timestamp': datetime.utcnow().isoformat()
            }), 400

        script_content = data['script']
        unique_id = str(uuid.uuid4())
        script_filename = f"script_{unique_id}.sh"
        script_path = os.path.join('/tmp', script_filename)

        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)

        result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=600)

        os.remove(script_path)

        if result.returncode == 0:
            return jsonify({
                'success': True,
                'job_id': unique_id,
                'output': result.stdout,
                'video_path': '/tmp/n8n/simple_video/final_output.mp4',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Script failed with code {result.returncode}',
                'output': result.stdout,
                'error_output': result.stderr,
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring
    """
    return jsonify({
        'status': 'healthy',
        'service': 'video-processing-webhook',
        'timestamp': datetime.utcnow().isoformat(),
        'ffmpeg_available': video_processor.check_ffmpeg_availability()
    }), 200

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 500MB.',
        'timestamp': datetime.utcnow().isoformat()
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
