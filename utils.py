import os
import logging
from typing import Optional
from flask import Request

logger = logging.getLogger(__name__)

# Allowed video file extensions
ALLOWED_EXTENSIONS = {
    'mp4', 'avi', 'mov', 'webm', 'mkv', 'flv', 'm4v', '3gp'
}

def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension
    """
    if not filename:
        return False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def validate_request(request: Request) -> Optional[str]:
    """
    Validate incoming webhook request
    Returns error message if validation fails, None if valid
    """
    # Check content type
    if not request.content_type:
        return "Content-Type header missing"
    
    if not request.content_type.startswith('multipart/form-data'):
        return "Content-Type must be multipart/form-data for file uploads"
    
    # Check if request has file part
    if 'video' not in request.files:
        return "No video file in request"
    
    # Validate output format if provided
    if 'output_format' in request.form:
        output_format = request.form['output_format'].lower()
        valid_formats = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
        if output_format not in valid_formats:
            return f"Invalid output format. Supported: {', '.join(valid_formats)}"
    
    # Validate quality if provided
    if 'quality' in request.form:
        quality = request.form['quality'].lower()
        valid_qualities = {'low', 'medium', 'high', 'ultra'}
        if quality not in valid_qualities:
            return f"Invalid quality setting. Supported: {', '.join(valid_qualities)}"
    
    # Validate resolution if provided
    if 'resolution' in request.form:
        resolution = request.form['resolution'].lower()
        valid_resolutions = {'480p', '720p', '1080p', '1440p', '4k'}
        if resolution not in valid_resolutions:
            return f"Invalid resolution. Supported: {', '.join(valid_resolutions)}"
    
    return None

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace problematic characters
    problematic_chars = '<>:"/\\|?*'
    for char in problematic_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 50:
        name = name[:50]
    
    return name + ext

def validate_video_parameters(params: dict) -> dict:
    """
    Validate and normalize video processing parameters
    """
    validated = {}
    
    # Output format
    output_format = params.get('output_format', 'mp4').lower()
    if output_format in {'mp4', 'avi', 'mov', 'webm', 'mkv'}:
        validated['output_format'] = output_format
    else:
        validated['output_format'] = 'mp4'
    
    # Quality
    quality = params.get('quality', 'medium').lower()
    if quality in {'low', 'medium', 'high', 'ultra'}:
        validated['quality'] = quality
    else:
        validated['quality'] = 'medium'
    
    # Resolution
    resolution = params.get('resolution', '').lower()
    if resolution in {'480p', '720p', '1080p', '1440p', '4k'}:
        validated['resolution'] = resolution
    else:
        validated['resolution'] = None
    
    # Compression
    compress = str(params.get('compress', 'false')).lower()
    validated['compress'] = compress in {'true', '1', 'yes', 'on'}
    
    return validated
