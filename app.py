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

@app.route('/webhook/create-story-video', methods=['POST'])
def create_story_video():
    """
    NEW ENDPOINT: Create a complete story video from audio, video, and subtitles
    This is specifically designed for your n8n workflow
    """
    try:
        logger.info("Received request to create story video")
        
        # Check if we have the required files
        required_files = ['audio', 'background_video']
        missing_files = []
        
        for file_key in required_files:
            if file_key not in request.files:
                missing_files.append(file_key)
        
        if missing_files:
            return jsonify({
                'success': False,
                'error': f'Missing required files: {", ".join(missing_files)}',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save uploaded files
        saved_files = {}
        
        # Save audio file
        audio_file = request.files['audio']
        if audio_file.filename:
            audio_filename = f"{job_id}_audio_{secure_filename(audio_file.filename)}"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            audio_file.save(audio_path)
            saved_files['audio'] = audio_path
            logger.info(f"Saved audio file: {audio_path}")
        
        # Save background video file
        video_file = request.files['background_video']
        if video_file.filename:
            video_filename = f"{job_id}_video_{secure_filename(video_file.filename)}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video_file.save(video_path)
            saved_files['background_video'] = video_path
            logger.info(f"Saved background video: {video_path}")
        
        # Save subtitle file (optional)
        subtitle_path = None
        if 'subtitles' in request.files:
            subtitle_file = request.files['subtitles']
            if subtitle_file.filename:
                subtitle_filename = f"{job_id}_subtitles_{secure_filename(subtitle_file.filename)}"
                subtitle_path = os.path.join(app.config['UPLOAD_FOLDER'], subtitle_filename)
                subtitle_file.save(subtitle_path)
                saved_files['subtitles'] = subtitle_path
                logger.info(f"Saved subtitle file: {subtitle_path}")
        
        # Generate output filename
        output_filename = f"{job_id}_final_story_video.mp4"
        
        # Create the final video
        result = video_processor.create_video_with_audio_and_subtitles(
            background_video_path=saved_files['background_video'],
            audio_path=saved_files['audio'],
            subtitle_path=subtitle_path,
            output_filename=output_filename
        )
        
        # Clean up uploaded files
        for file_path in saved_files.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {str(e)}")
        
        if result['success']:
            logger.info(f"Story video created successfully: {output_filename}")
            return jsonify({
                'success': True,
                'job_id': job_id,
                'output_file': result['output_file'],
                'output_path': result['output_path'],
                'metadata': result['metadata'],
                'processing_time': result['processing_time'],
                'download_url': f"/download/{result['output_file']}",
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            logger.error(f"Story video creation failed: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error'],
                'job_id': job_id,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error creating story video: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

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
        file_path = os.path.join(app.config['PROCESSED_FOLDER
