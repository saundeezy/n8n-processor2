import os
import time
import logging
import ffmpeg
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles video processing operations using FFmpeg
    """
    
    def __init__(self, upload_folder: str, processed_folder: str):
        self.upload_folder = upload_folder
        self.processed_folder = processed_folder
        
        # Quality presets for video compression
        self.quality_presets = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'},
            'ultra': {'crf': 15, 'preset': 'veryslow'}
        }
        
        # Resolution presets
        self.resolution_presets = {
            '480p': '854:480',
            '720p': '1280:720',
            '1080p': '1920:1080',
            '1440p': '2560:1440',
            '4k': '3840:2160'
        }

    def check_ffmpeg_availability(self) -> bool:
        """
        Check if FFmpeg is available in the system
        """
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("FFmpeg not found in system PATH")
            return False
        except Exception as e:
            logger.error(f"Error checking FFmpeg availability: {str(e)}")
            return False

    def extract_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from video file
        """
        try:
            file_path = os.path.join(self.upload_folder, filename)
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None

            # Get video information using ffprobe
            probe = ffmpeg.probe(file_path)
            
            # Find video stream
            video_stream = None
            audio_stream = None
            
            for stream in probe['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream

            if not video_stream:
                logger.error("No video stream found in file")
                return None

            # Extract metadata
            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0)),
                'format_name': probe['format'].get('format_name', ''),
                'video': {
                    'codec': video_stream.get('codec_name', ''),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': self._get_fps(video_stream),
                    'bitrate': int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None,
                    'pixel_format': video_stream.get('pix_fmt', '')
                }
            }

            # Add audio metadata if available
            if audio_stream:
                metadata['audio'] = {
                    'codec': audio_stream.get('codec_name', ''),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None
                }

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {filename}: {str(e)}")
            return None

    def _get_fps(self, video_stream: Dict) -> float:
        """
        Extract frame rate from video stream
        """
        try:
            # Try r_frame_rate first
            if 'r_frame_rate' in video_stream:
                fps_str = video_stream['r_frame_rate']
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    return float(num) / float(den)
                return float(fps_str)
            
            # Fall back to avg_frame_rate
            if 'avg_frame_rate' in video_stream:
                fps_str = video_stream['avg_frame_rate']
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    return float(num) / float(den)
                return float(fps_str)
                
            return 0.0
        except:
            return 0.0

    def process_video(self, filename: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video file according to specified parameters
        """
        start_time = time.time()
        
        try:
            input_path = os.path.join(self.upload_folder, filename)
            
            if not os.path.exists(input_path):
                return {
                    'success': False,
                    'error': f'Input file not found: {filename}'
                }

            # Extract original metadata
            original_metadata = self.extract_metadata(filename)
            if not original_metadata:
                return {
                    'success': False,
                    'error': 'Failed to extract video metadata'
                }

            # Generate output filename
            name, _ = os.path.splitext(filename)
            output_format = params.get('output_format', 'mp4')
            output_filename = f"{name}_processed.{output_format}"
            output_path = os.path.join(self.processed_folder, output_filename)

            # Build FFmpeg command
            input_stream = ffmpeg.input(input_path)
            output_args = {}

            # Set video codec based on output format
            if output_format in ['mp4', 'mov']:
                output_args['vcodec'] = 'libx264'
                output_args['acodec'] = 'aac'
            elif output_format == 'webm':
                output_args['vcodec'] = 'libvpx-vp9'
                output_args['acodec'] = 'libvorbis'
            elif output_format == 'avi':
                output_args['vcodec'] = 'libx264'
                output_args['acodec'] = 'mp3'

            # Apply quality settings
            quality = params.get('quality', 'medium')
            if quality in self.quality_presets:
                preset_settings = self.quality_presets[quality]
                output_args['crf'] = preset_settings['crf']
                output_args['preset'] = preset_settings['preset']

            # Apply resolution scaling if specified
            resolution = params.get('resolution')
            if resolution and resolution in self.resolution_presets:
                scale = self.resolution_presets[resolution]
                input_stream = ffmpeg.filter(input_stream, 'scale', scale)

            # Apply compression if requested
            if params.get('compress', False):
                # Additional compression settings
                output_args['crf'] = min(output_args.get('crf', 23) + 5, 32)
                output_args['preset'] = 'fast'

            # Create output stream
            output_stream = ffmpeg.output(input_stream, output_path, **output_args)

            # Run FFmpeg command
            logger.info(f"Starting video processing: {filename} -> {output_filename}")
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Get processed file metadata
            processed_metadata = self.extract_metadata(output_filename)

            # Calculate file size reduction
            original_size = original_metadata.get('size', 0)
            processed_size = processed_metadata.get('size', 0) if processed_metadata else 0
            size_reduction = None
            
            if original_size > 0 and processed_size > 0:
                size_reduction = {
                    'original_size_mb': round(original_size / (1024 * 1024), 2),
                    'processed_size_mb': round(processed_size / (1024 * 1024), 2),
                    'reduction_percent': round(((original_size - processed_size) / original_size) * 100, 2)
                }

            # Clean up original file
            try:
                os.remove(input_path)
            except:
                logger.warning(f"Failed to clean up original file: {input_path}")

            return {
                'success': True,
                'output_file': output_filename,
                'metadata': processed_metadata or original_metadata,
                'processing_time': round(processing_time, 2),
                'file_size_reduction': size_reduction
            }

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error processing {filename}: {str(e)}")
            return {
                'success': False,
                'error': f'Video processing failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error processing {filename}: {str(e)}")
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }

    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old files from upload and processed directories
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for folder in [self.upload_folder, self.processed_folder]:
            try:
                for filename in os.listdir(folder):
                    if filename == '.gitkeep':
                        continue
                        
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            logger.info(f"Cleaned up old file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up files in {folder}: {str(e)}")
