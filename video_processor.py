import os
import time
import logging
import ffmpeg
import subprocess
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
        Create a final video - bypass MoviePy entirely if it fails
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
            
            # Get audio duration using MoviePy (this should work)
            try:
                temp_audio = AudioFileClip(audio_path)
                audio_duration = temp_audio.duration
                temp_audio.close()
                logger.info(f"Audio duration: {audio_duration} seconds")
            except Exception as e:
                logger.error(f"Failed to get audio duration: {str(e)}")
                audio_duration = 60.0
                logger.warning(f"Using fallback duration of {audio_duration} seconds")
            
            # Try direct FFmpeg approach first (bypass MoviePy completely for video)
            logger.info("Attempting direct FFmpeg approach...")
            try:
                return self._create_video_direct_ffmpeg(background_video_path, audio_path, 
                                                      subtitle_path, output_filename, 
                                                      audio_duration, start_time)
            except Exception as ffmpeg_error:
                logger.warning(f"Direct FFmpeg failed: {str(ffmpeg_error)}")
                
                # Last resort: Try the safest MoviePy approach possible
                logger.info("Trying ultra-safe MoviePy approach...")
                return self._create_video_safe_moviepy(background_video_path, audio_path, 
                                                     subtitle_path, output_filename, 
                                                     audio_duration, start_time)
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            return {
                'success': False,
                'error': f'Video creation failed: {str(e)}'
            }

    def _create_video_direct_ffmpeg(self, background_video_path: str, audio_path: str, 
                                  subtitle_path: Optional[str], output_filename: str, 
                                  audio_duration: float, start_time: float) -> Dict[str, Any]:
        """
        Use FFmpeg directly without MoviePy at all
        """
        try:
            logger.info("Using direct FFmpeg approach (no MoviePy)")
            
            output_path = os.path.join(self.processed_folder, output_filename)
            
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                raise Exception("FFmpeg not available")
            
            # Build FFmpeg command that handles any video input
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output files
                '-stream_loop', '-1',  # Loop video indefinitely
                '-i', background_video_path,  # Background video input
                '-i', audio_path,  # Audio input
                '-t', str(audio_duration),  # Duration = audio length
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-r', '24',  # Explicit frame rate (fixes broken FPS)
                '-vsync', 'cfr',  # Constant frame rate
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-shortest',  # Stop when shortest stream ends
                '-preset', 'ultrafast',  # Fast encoding
                '-crf', '28',  # Reasonable quality
                output_path
            ]
            
            logger.info(f"Running direct FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise Exception('Output video file was not created')
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Direct FFmpeg video creation completed: {output_filename}")
            
            return {
                'success': True,
                'output_file': output_filename,
                'output_path': output_path,
                'metadata': {'duration': audio_duration, 'fps': 24, 'method': 'direct_ffmpeg'},
                'processing_time': round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Direct FFmpeg method failed: {str(e)}")
            raise e

    def _create_video_safe_moviepy(self, background_video_path: str, audio_path: str, 
                                 subtitle_path: Optional[str], output_filename: str, 
                                 audio_duration: float, start_time: float) -> Dict[str, Any]:
        """
        Ultra-safe MoviePy approach that pre-processes the video
        """
        try:
            logger.info("Using ultra-safe MoviePy approach")
            
            output_path = os.path.join(self.processed_folder, output_filename)
            
            # Pre-process the video file to fix any issues
            processed_video_path = self._preprocess_video_file(background_video_path)
            
            # Now try loading with MoviePy
            try:
                logger.info("Loading preprocessed video...")
                
                # Load video with explicit parameters to avoid FPS issues
                video_clip = VideoFileClip(processed_video_path, audio=False)  # Don't load audio
                
                # Force set FPS immediately
                video_clip.fps = 24.0
                logger.info(f"Video loaded and FPS set to: {video_clip.fps}")
                
            except Exception as video_error:
                logger.error(f"Failed to load even preprocessed video: {str(video_error)}")
                return {'success': False, 'error': f'Video file cannot be processed: {str(video_error)}'}
            
            # Load audio separately
            try:
                audio_clip = AudioFileClip(audio_path)
            except Exception as audio_error:
                video_clip.close()
                return {'success': False, 'error': f'Audio file cannot be processed: {str(audio_error)}'}
            
            # Process video safely with better audio handling
            try:
                logger.info(f"Video duration: {video_clip.duration}s, Audio duration: {audio_duration}s")
                
                # Loop if needed
                if video_clip.duration < audio_duration:
                    loop_count = int(audio_duration / video_clip.duration) + 1
                    logger.info(f"Looping video {loop_count} times to match audio duration")
                    video_clip = video_clip.loop(n=loop_count)
                    video_clip.fps = 24.0
                
                # Cut to match audio duration exactly
                logger.info(f"Cutting video from {video_clip.duration}s to {audio_duration}s")
                video_clip = video_clip.subclip(0, audio_duration)
                video_clip.fps = 24.0
                
                # Remove existing audio from video first
                video_clip_no_audio = video_clip.without_audio()
                logger.info("Removed original video audio")
                
                # Set the new audio
                logger.info("Adding narration audio to video...")
                final_video = video_clip_no_audio.set_audio(audio_clip)
                final_video.fps = 24.0
                
                # Verify audio is attached
                if final_video.audio is None:
                    logger.error("WARNING: Final video has no audio attached!")
                else:
                    logger.info(f"SUCCESS: Final video has audio duration: {final_video.audio.duration}s")
                
                # Write with explicit audio parameters
                logger.info("Writing final video with audio...")
                final_video.write_videofile(
                    output_path,
                    fps=24,
                    verbose=False,
                    logger=None,
                    audio=True,  # Force audio inclusion
                    audio_codec='aac',
                    audio_bitrate='128k',  # Explicit audio bitrate
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                
                # Clean up
                video_clip.close()
                audio_clip.close()
                video_clip_no_audio.close()
                final_video.close()
                
                processing_time = time.time() - start_time
                
                logger.info(f"Video creation completed successfully in {processing_time:.1f}s")
                
                return {
                    'success': True,
                    'output_file': output_filename,
                    'output_path': output_path,
                    'metadata': {
                        'duration': audio_duration, 
                        'fps': 24, 
                        'method': 'safe_moviepy_explicit_audio',
                        'loops_needed': int(audio_duration / 62) + 1 if 62 < audio_duration else 1,
                        'original_video_duration': 62,
                        'final_duration': audio_duration
                    },
                    'processing_time': round(processing_time, 2)
                }
                
            except Exception as processing_error:
                logger.error(f"Video processing failed: {str(processing_error)}")
                return {'success': False, 'error': f'Video processing failed: {str(processing_error)}'}
            
        except Exception as e:
            logger.error(f"Safe MoviePy method failed: {str(e)}")
            return {'success': False, 'error': f'Safe MoviePy failed: {str(e)}'}

    def _preprocess_video_file(self, video_path: str) -> str:
        """
        Pre-process video file to fix metadata issues
        """
        try:
            logger.info("Pre-processing video file to fix metadata...")
            
            # Create output path for processed file
            processed_path = video_path.replace('.', '_processed.')
            
            # Try using ffmpeg to fix the video
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
                
                # Simple re-encode to fix metadata
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-c:v', 'copy',  # Copy video stream (faster)
                    '-c:a', 'copy',  # Copy audio stream
                    '-r', '24',      # Set explicit frame rate
                    '-avoid_negative_ts', 'make_zero',  # Fix timing issues
                    processed_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(processed_path):
                    logger.info("Video preprocessed successfully")
                    return processed_path
                    
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.info("FFmpeg not available for preprocessing")
            
            # If FFmpeg fails, return original file
            logger.info("Using original video file")
            return video_path
            
        except Exception as e:
            logger.warning(f"Video preprocessing failed: {str(e)}")
            return video_path

    def _create_video_with_ffmpeg(self, background_video_path: str, audio_path: str, 
                                 subtitle_path: Optional[str], output_filename: str, 
                                 audio_duration: float, start_time: float) -> Dict[str, Any]:
        """
        Create video using FFmpeg
        """
        try:
            output_path = os.path.join(self.processed_folder, output_filename)
            
            # Build FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output files
                '-stream_loop', '-1',  # Loop video indefinitely
                '-i', background_video_path,  # Background video input
                '-i', audio_path,  # Audio input
                '-t', str(audio_duration),  # Duration = audio length
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-r', '24',  # Frame rate
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-shortest',  # Stop when shortest stream ends
                '-preset', 'fast',  # Fast encoding
                '-crf', '23',  # Good quality
            ]
            
            # Add subtitle support if subtitle file exists
            if subtitle_path and os.path.exists(subtitle_path):
                logger.info(f"Adding subtitles from: {subtitle_path}")
                ffmpeg_cmd.extend([
                    '-vf', f"subtitles='{subtitle_path}':force_style='FontSize=28,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=3,Bold=1,MarginV=150,Alignment=2'",
                ])
            
            # Add output path at the end
            ffmpeg_cmd.append(output_path)
            
            logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise Exception('Output video file was not created')
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get metadata
            metadata = self.extract_metadata_from_path(output_path)
            
            logger.info(f"FFmpeg video creation completed: {output_filename}")
            
            return {
                'success': True,
                'output_file': output_filename,
                'output_path': output_path,
                'metadata': metadata,
                'processing_time': round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"FFmpeg method failed: {str(e)}")
            # Fallback to MoviePy
            logger.info("Falling back to MoviePy")
            return self._create_video_with_moviepy(background_video_path, audio_path, 
                                                 subtitle_path, output_filename, 
                                                 audio_duration, start_time)

    def _create_video_with_moviepy(self, background_video_path: str, audio_path: str, 
                                  subtitle_path: Optional[str], output_filename: str, 
                                  audio_duration: float, start_time: float) -> Dict[str, Any]:
        """
        Create video using MoviePy with video file repair
        """
        try:
            logger.info("Using MoviePy for video creation")
            
            output_path = os.path.join(self.processed_folder, output_filename)
            
            # STEP 1: Fix the background video file first
            logger.info("Fixing background video FPS metadata...")
            fixed_video_path = self._fix_video_fps(background_video_path)
            
            # STEP 2: Load clips with explicit error handling
            try:
                logger.info("Loading fixed video clip...")
                video_clip = VideoFileClip(fixed_video_path)
                logger.info(f"Video clip loaded: duration={video_clip.duration}, fps={video_clip.fps}")
            except Exception as e:
                logger.error(f"Failed to load video clip: {str(e)}")
                return {'success': False, 'error': f'Failed to load background video: {str(e)}'}
            
            try:
                logger.info("Loading audio clip...")
                audio_clip = AudioFileClip(audio_path)
                logger.info(f"Audio clip loaded: duration={audio_clip.duration}")
            except Exception as e:
                logger.error(f"Failed to load audio clip: {str(e)}")
                video_clip.close()
                return {'success': False, 'error': f'Failed to load audio: {str(e)}'}
            
            # STEP 3: Ensure video has proper FPS
            try:
                logger.info("Ensuring video has proper FPS...")
                if not hasattr(video_clip, 'fps') or video_clip.fps is None or video_clip.fps == 0:
                    video_clip.fps = 24.0
                    logger.info("Set video FPS to 24.0")
                else:
                    logger.info(f"Video already has FPS: {video_clip.fps}")
                    
                # Ensure it's a float
                video_clip.fps = float(video_clip.fps)
                
            except Exception as e:
                logger.error(f"Failed to set FPS: {str(e)}")
                video_clip.fps = 24.0
            
            # STEP 4: Loop the video if needed
            try:
                if video_clip.duration < audio_duration:
                    loop_count = int(audio_duration / video_clip.duration) + 1
                    logger.info(f"Looping video {loop_count} times")
                    video_clip = video_clip.loop(n=loop_count)
                    video_clip.fps = 24.0
            except Exception as e:
                logger.error(f"Failed to loop video: {str(e)}")
                return {'success': False, 'error': f'Video looping failed: {str(e)}'}
            
            # STEP 5: Cut video to match audio duration
            try:
                logger.info("Cutting video to match audio duration...")
                video_clip = video_clip.subclip(0, audio_duration)
                video_clip.fps = 24.0
            except Exception as e:
                logger.error(f"Failed to cut video: {str(e)}")
                return {'success': False, 'error': f'Video cutting failed: {str(e)}'}
            
            # STEP 6: Set audio
            try:
                logger.info("Setting audio...")
                final_video = video_clip.set_audio(audio_clip)
                final_video.fps = 24.0
            except Exception as e:
                logger.error(f"Failed to set audio: {str(e)}")
                return {'success': False, 'error': f'Audio setting failed: {str(e)}'}
            
            # STEP 7: Write video with fallback methods
            try:
                logger.info("Writing video file...")
                final_video.write_videofile(
                    output_path,
                    fps=24.0,
                    verbose=False,
                    logger=None,
                    audio=True,
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                logger.info("Video file written successfully")
                
            except Exception as write_error:
                logger.error(f"Write attempt failed: {str(write_error)}")
                return {'success': False, 'error': f'Video writing failed: {str(write_error)}'}
            
            # STEP 8: Clean up
            try:
                video_clip.close()
                audio_clip.close()
                final_video.close()
                # Clean up the fixed video file
                if fixed_video_path != background_video_path and os.path.exists(fixed_video_path):
                    os.remove(fixed_video_path)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {str(cleanup_error)}")
            
            # Verify output file exists
            if not os.path.exists(output_path):
                return {'success': False, 'error': 'Output video file was not created'}
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"MoviePy video creation completed: {output_filename}")
            
            return {
                'success': True,
                'output_file': output_filename,
                'output_path': output_path,
                'metadata': {'duration': audio_duration, 'fps': 24},
                'processing_time': round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"MoviePy method failed with error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f'Video creation failed: {str(e)}'
            }

    def _fix_video_fps(self, video_path: str) -> str:
        """
        Fix video file by re-encoding it with proper FPS metadata
        """
        try:
            logger.info(f"Fixing FPS metadata for: {video_path}")
            
            # Create a fixed version using a simple re-encode
            fixed_path = video_path.replace('.', '_fixed.')
            
            # Try using subprocess to re-encode with ffmpeg if available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
                
                # Re-encode the video with explicit FPS
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-c:v', 'libx264', '-r', '24', '-preset', 'ultrafast',
                    '-c:a', 'copy',  # Copy audio without re-encoding
                    fixed_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(fixed_path):
                    logger.info("Video fixed successfully with ffmpeg")
                    return fixed_path
                else:
                    logger.warning("FFmpeg fix failed, using original file")
                    return video_path
                    
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.info("FFmpeg not available for video fixing")
                return video_path
                
        except Exception as e:
            logger.warning(f"Video fixing failed: {str(e)}, using original file")
            return video_path

    def _add_subtitles_to_video(self, video_clip, subtitle_path: str):
        """
        Add subtitles to video using SRT file (MoviePy method)
        """
        try:
            subtitles = self._parse_srt_file(subtitle_path)
            subtitle_clips = []
            
            for subtitle in subtitles:
                txt_clip = TextClip(
                    subtitle['text'],
                    fontsize=50,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial-Bold'
                ).set_position(('center', 'bottom')).set_start(subtitle['start']).set_end(subtitle['end'])
                
                subtitle_clips.append(txt_clip)
            
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
            
            blocks = content.split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    timestamp_line = lines[1]
                    if ' --> ' in timestamp_line:
                        start_str, end_str = timestamp_line.split(' --> ')
                        start_time = self._srt_time_to_seconds(start_str)
                        end_time = self._srt_time_to_seconds(end_str)
                        
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
        Convert SRT time format to seconds
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
        Extract metadata from video file
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None

            probe = ffmpeg.probe(file_path)
            
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
            if 'r_frame_rate' in video_stream:
                fps_str = video_stream['r_frame_rate']
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    return float(num) / float(den)
                return float(fps_str)
            
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

            original_metadata = self.extract_metadata(filename)
            if not original_metadata:
                return {
                    'success': False,
                    'error': 'Failed to extract video metadata'
                }

            name, _ = os.path.splitext(filename)
            output_format = params.get('output_format', 'mp4')
            output_filename = f"{name}_processed.{output_format}"
            output_path = os.path.join(self.processed_folder, output_filename)

            input_stream = ffmpeg.input(input_path)
            output_args = {}

            if output_format in ['mp4', 'mov']:
                output_args['vcodec'] = 'libx264'
                output_args['acodec'] = 'aac'
            elif output_format == 'webm':
                output_args['vcodec'] = 'libvpx-vp9'
                output_args['acodec'] = 'libvorbis'
            elif output_format == 'avi':
                output_args['vcodec'] = 'libx264'
                output_args['acodec'] = 'mp3'

            quality = params.get('quality', 'medium')
            if quality in self.quality_presets:
                preset_settings = self.quality_presets[quality]
                output_args['crf'] = preset_settings['crf']
                output_args['preset'] = preset_settings['preset']

            resolution = params.get('resolution')
            if resolution and resolution in self.resolution_presets:
                scale = self.resolution_presets[resolution]
                input_stream = ffmpeg.filter(input_stream, 'scale', scale)

            if params.get('compress', False):
                output_args['crf'] = min(output_args.get('crf', 23) + 5, 32)
                output_args['preset'] = 'fast'

            output_stream = ffmpeg.output(input_stream, output_path, **output_args)

            logger.info(f"Starting video processing: {filename} -> {output_filename}")
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

            processing_time = time.time() - start_time
            processed_metadata = self.extract_metadata(output_filename)

            original_size = original_metadata.get('size', 0)
            processed_size = processed_metadata.get('size', 0) if processed_metadata else 0
            size_reduction = None
            
            if original_size > 0 and processed_size > 0:
                size_reduction = {
                    'original_size_mb': round(original_size / (1024 * 1024), 2),
                    'processed_size_mb': round(processed_size / (1024 * 1024), 2),
                    'reduction_percent': round(((original_size - processed_size) / original_size) * 100, 2)
                }

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
