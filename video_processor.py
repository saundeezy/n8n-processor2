import os
import time
import logging
import ffmpeg
from typing import Dict, Optional, Any
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles video processing operations using FFmpeg and MoviePy
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

    def create_video_with_audio_and_subtitles(self, background_video_path: str, audio_path: str, 
                                            subtitle_path: Optional[str] = None, 
                                            output_filename: str = "final_video.mp4") -> Dict[str, Any]:
        """
        Create a final video by combining background video, audio, and subtitles
        This is the main method your n8n workflow will use
        """
        start_time = time.time()
        
        try:
            # Ensure all input files exist
            if not os.path.exists(background_video_path):
                return {'success': False, 'error': f'Background video not found: {background_video_path}'}
            
            if not os.path.exists(audio_path):
                return {'success': False, 'error': f'Audio file not found: {audio_path}'}
            
            # Create output path
            output_path = os.path.join(self.processed_folder, output_filename)
            
            logger.info(f"Creating video: {output_filename}")
            logger.info(f"Background video: {background_video_path}")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Subtitles: {subtitle_path}")
            
            # Load video and audio
            video_clip = VideoFileClip(background_video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Get the duration of the audio (this will be our final video length)
            audio_duration = audio_clip.duration
            
            # Loop the background video if it's shorter than the audio
            if video_clip.duration < audio_duration:
                # Calculate how many times we need to loop
                loop_count = int(audio_duration / video_clip.duration) + 1
                video_clip = video_clip.loop(n=loop_count)
            
            # Cut the video to match audio duration
            video_clip = video_clip.subclip(0, audio_duration)
            
            # Replace the video's audio with our narration
            final_video = video_clip.set_audio(audio_clip)
            
            # Add subtitles if provided
            if subtitle_path and os.path.exists(subtitle_path):
                final_video = self._add_subtitles_to_video(final_video, subtitle_path)
            
            # Write the final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None  # Suppress moviepy logs
            )
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            final_video.close()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get metadata of the final video
            metadata = self.extract_metadata_from_path(output_path)
            
            logger.info(f"Video creation completed: {output_filename}")
            
            return {
                'success': True,
                'output_file': output_filename,
                'output_path': output_path,
                'metadata': metadata,
                'processing_time': round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            return {
                'success': False,
                'error': f'Video creation failed: {str(e)}'
            }

    def _add_subtitles_to_video(self, video_clip, subtitle_path: str):
        """
        Add subtitles to video using SRT file
        """
        try:
            # Parse SRT file
            subtitles = self._parse_srt_file(subtitle_path)
            
            subtitle_clips = []
            
            for subtitle in subtitles:
                # Create text clip for each subtitle
                txt_clip = TextClip(
                    subtitle['text'],
                    fontsize=50,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial-Bold'
                ).set_position(('center', 'bottom')).set_start(subtitle['start']).set_end(subtitle['end'])
                
                subtitle_clips.append(txt_clip)
            
            # Composite video with subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video_clip] + subtitle_clips)
                return final_video
            else:
                return video_clip
                
        except Exception as e:
            logger.warning(f"Failed to add subtitles: {str(e)}")
            return video_clip

    def _parse_srt_file(self, subtitle_path: str) -> list:
        """
        Parse SRT subtitle file
        """
        subtitles = []
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Split by double newlines to separate subtitle blocks
            blocks = content.split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse timestamp line (format: 00:00:01,000 --> 00:00:04,000)
                    timestamp_line = lines[1]
                    if ' --> ' in timestamp_line:
                        start_str, end_str = timestamp_line.split(' --> ')
                        start_time = self._srt_time_to_seconds(start_str)
                        end_time = self._srt_time_to_seconds(end_str)
                        
                        # Get subtitle text (everything after the timestamp line)
                        text = '\n'.join(lines[2:])
                        
                        subtitles.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
            
            return subtitles
            
        except Exception as e:
            logger.error(f"Error parsing SRT file: {str(e)}")
            return []

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """
        Convert SRT time format (HH:MM:SS,mmm) to seconds
        """
        try:
            time_part, ms_part = time_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            
            total_seconds = h * 3600 + m * 60 + s + ms / 1000
            return total_seconds
        except:
            return 0.0

    def extract_metadata_from_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from video file using direct path
        """
        try:
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
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return None

    def extract_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from video file
        """
        try:
            file_path = os.path.join(self.upload_folder, filename)
            return self.extract_metadata_from_path(file_path)
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
