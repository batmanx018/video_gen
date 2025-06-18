import os
import re
import gc
import logging
import asyncio
import whisper
import requests
import tempfile
import edge_tts
import moviepy.editor as mp
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
import time

# Setup logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_generation.log')
    ]
)

load_dotenv()

# Configuration constants
class Config:
    AUDIO_SUFFIX = ".mp3"
    VIDEO_SUFFIX = ".mp4"
    VIDEO_DIR = "./videos"
    MAX_VIDEOS_PER_KEYWORD = 3
    MAX_TOTAL_VIDEOS = 15
    MIN_ASPECT_RATIO = 1.2
    MAX_ASPECT_RATIO = 2.1
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 480
    MAX_RETRIES = 3
    CHUNK_SIZE = 8192
    
    # Video encoding settings
    VIDEO_CODEC = "libx264"
    AUDIO_CODEC = "aac"
    FPS = 24
    PRESET = "ultrafast"
    BITRATE = "300k"
    
    # Caption settings
    FONT_SIZE = 42
    FONT_COLOR = 'white'
    STROKE_COLOR = 'black'
    STROKE_WIDTH = 2
    FONT_FAMILY = 'Arial'


# Create directories
os.makedirs(Config.VIDEO_DIR, exist_ok=True)

# API Keys validation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
WHISPER_MODEL = whisper.load_model("base")  # global

if not all([GEMINI_API_KEY, PEXELS_API_KEY]):
    raise ValueError("Missing required API keys. Check your .env file.")

# Cloudinary configuration
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
except Exception as e:
    logging.warning(f"Cloudinary configuration failed: {e}")

# Gemini configuration
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

@contextmanager
def temp_file(suffix: str):
    """Context manager for temporary files with automatic cleanup."""
    temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file_obj.name
    temp_file_obj.close()
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logging.warning(f"Failed to remove temp file {temp_path}: {e}")

class VideoGenerationError(Exception):
    """Custom exception for video generation errors."""
    pass

async def text_to_speech(text: str, output_path: str, voice: str = "en-US-AriaNeural") -> None:
    """Convert text to speech using Edge TTS."""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)
        logging.info(f"✅ TTS saved to: {output_path}")
    except Exception as e:
        logging.error(f"TTS generation failed: {e}")
        raise VideoGenerationError(f"Text-to-speech failed: {e}")

def generate_captions(audio_path: str) -> List[Dict]:
    """Generate captions from audio using Whisper."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        logging.info("Loading Whisper model...")
        whisper_model = WHISPER_MODEL 
        
        logging.info("Transcribing audio...")
        result = whisper_model.transcribe(
            audio_path, 
            word_timestamps=True, 
            verbose=False,
            temperature=0.0  # More deterministic results
        )
        
        # Clean up captions
        captions = []
        for seg in result["segments"]:
            cleaned_text = re.sub(r'[\*\[\]]+', '', seg["text"]).strip()
            if cleaned_text:  # Only add non-empty captions
                seg["text"] = cleaned_text
                captions.append(seg)
        
        logging.info(f"Generated {len(captions)} caption segments")
        return captions
        
    except Exception as e:
        logging.error(f"Caption generation failed: {e}")
        raise VideoGenerationError(f"Caption generation failed: {e}")

def normalize_keywords(keywords) -> List[str]:
    """Normalize and validate keywords input."""
    if isinstance(keywords, str):
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    elif isinstance(keywords, list):
        keyword_list = [str(kw).strip() for kw in keywords if str(kw).strip()]
    else:
        logging.warning(f"Invalid keywords type: {type(keywords)}")
        keyword_list = []
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keyword_list:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_keywords.append(kw)
    
    return unique_keywords[:10]  # Limit to 10 keywords

def fetch_video_urls(keywords, per_keyword: int = Config.MAX_VIDEOS_PER_KEYWORD) -> List[str]:
    """Fetch video URLs from Pexels API with improved error handling."""
    headers = {"Authorization": PEXELS_API_KEY}
    urls = []
    
    keyword_list = normalize_keywords(keywords)
    if not keyword_list:
        logging.warning("No valid keywords provided")
        return []
    
    logging.info(f"🔍 Searching for videos with keywords: {keyword_list}")
    
    for keyword in keyword_list:
        if len(urls) >= Config.MAX_TOTAL_VIDEOS:
            break
            
        retry_count = 0
        while retry_count < Config.MAX_RETRIES:
            try:
                logging.info(f"📦 Fetching videos for: '{keyword}' (attempt {retry_count + 1})")
                
                response = requests.get(
                    "https://api.pexels.com/videos/search",
                    headers=headers,
                    params={
                        "query": keyword,
                        "per_page": 15,
                        "orientation": "landscape",
                        "size": "medium"
                    },
                    timeout=10
                )
                response.raise_for_status()
                
                videos = response.json().get("videos", [])
                logging.info(f"📺 Found {len(videos)} videos for '{keyword}'")
                
                added_count = 0
                for video in videos:
                    if added_count >= per_keyword or len(urls) >= Config.MAX_TOTAL_VIDEOS:
                        break
                        
                    video_files = sorted(
                        video.get("video_files", []), 
                        key=lambda x: x.get("width", 0), 
                        reverse=True
                    )
                    
                    for vf in video_files:
                        width = vf.get("width", 1)
                        height = vf.get("height", 1)
                        
                        if height == 0:
                            continue
                            
                        aspect_ratio = width / height
                        
                        if Config.MIN_ASPECT_RATIO <= aspect_ratio <= Config.MAX_ASPECT_RATIO:
                            video_url = vf.get("link")
                            if video_url and video_url not in urls:
                                logging.debug(f"✅ Selected: {video_url} (ratio={aspect_ratio:.2f})")
                                urls.append(video_url)
                                added_count += 1
                                break
                
                break  # Success, exit retry loop
                
            except requests.RequestException as e:
                retry_count += 1
                logging.warning(f"❌ API error for '{keyword}' (attempt {retry_count}): {e}")
                if retry_count < Config.MAX_RETRIES:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logging.error(f"Failed to fetch videos for '{keyword}' after {Config.MAX_RETRIES} attempts")
            
            except Exception as e:
                logging.error(f"💥 Unexpected error for '{keyword}': {e}")
                break
    
    logging.info(f"🎯 Total video URLs collected: {len(urls)}")
    return urls

def download_video(url: str, filename: str) -> Optional[str]:
    """Download a video with retry logic and validation."""
    if not url or not filename:
        logging.error(f"❌ Invalid download parameters: url={bool(url)}, filename={bool(filename)}")
        return None
        
    filepath = os.path.join(Config.VIDEO_DIR, filename)
    
    for attempt in range(Config.MAX_RETRIES):
        try:
            logging.info(f"⬇️ Downloading: {filename} from {url[:50]}... (attempt {attempt + 1})")
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            logging.info(f"📄 Content type: {content_type}")
            
            if 'video' not in content_type and 'octet-stream' not in content_type:
                logging.warning(f"⚠️ Unexpected content type: {content_type}")
            
            total_size = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Verify file was downloaded and has content
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logging.info(f"📊 Downloaded file size: {file_size / (1024*1024):.2f} MB")
                
                if file_size > 1024:  # At least 1KB
                    # Try to verify it's a valid video file
                    try:
                        # Quick validation using moviepy
                        test_clip = mp.VideoFileClip(filepath, verbose=False)
                        if test_clip.duration > 0:
                            test_clip.close()
                            logging.info(f"✅ Downloaded and verified: {filename}")
                            return filepath
                        else:
                            logging.error(f"❌ Downloaded file has no duration: {filename}")
                            test_clip.close()
                    except Exception as validation_error:
                        logging.error(f"❌ Downloaded file validation failed for {filename}: {validation_error}")
                        # Don't immediately fail - the file might still be usable
                        return filepath
                else:
                    logging.error(f"❌ Downloaded file too small ({file_size} bytes): {filename}")
            else:
                logging.error(f"❌ File not found after download: {filepath}")
                
        except requests.exceptions.Timeout:
            logging.warning(f"⏰ Download timeout for {filename} (attempt {attempt + 1})")
        except requests.exceptions.RequestException as e:
            logging.warning(f"🌐 Request error for {filename} (attempt {attempt + 1}): {e}")
        except Exception as e:
            logging.error(f"💥 Unexpected download error for {filename} (attempt {attempt + 1}): {e}")
        
        # Cleanup failed download
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
            
        if attempt < Config.MAX_RETRIES - 1:
            wait_time = 2 ** attempt
            logging.info(f"⏳ Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    logging.error(f"❌ Failed to download after {Config.MAX_RETRIES} attempts: {filename}")
    return None

def resize_and_pad_clip(clip, target_width: int = Config.TARGET_WIDTH, target_height: int = Config.TARGET_HEIGHT):
    """Resize clip to fit target dimensions while maintaining aspect ratio."""
    try:
        # Calculate scaling to fit within target dimensions
        scale_w = target_width / clip.w
        scale_h = target_height / clip.h
        scale = min(scale_w, scale_h)
        
        # Resize maintaining aspect ratio
        new_width = int(clip.w * scale)
        new_height = int(clip.h * scale)
        
        resized = clip.resize((new_width, new_height))
        
        # Add padding to reach exact target dimensions
        padded = resized.on_color(
            size=(target_width, target_height),
            color=(0, 0, 0),
            pos='center'
        )
        
        return padded
        
    except Exception as e:
        logging.error(f"Failed to resize clip: {e}")
        return clip

def combine_video_audio_captions(video_paths: List[str], captions: List[Dict], audio_path: str, output_path: str) -> str:
    """Combine videos, audio, and captions into final video."""
    if not video_paths:
        raise VideoGenerationError("No video paths provided")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    clips_to_close = []  # Track clips for cleanup
    
    try:
        # Load audio
        audio = mp.AudioFileClip(audio_path)
        clips_to_close.append(audio)
        audio_duration = audio.duration
        
        logging.info(f"🎵 Audio duration: {audio_duration:.2f}s")
        logging.info(f"📹 Processing {len(video_paths)} video paths")
        
        # Load and process video clips
        valid_clips = []
        for i, path in enumerate(video_paths[:4]):  # Limit to prevent memory issues
            logging.info(f"🔍 Processing video {i+1}/{min(len(video_paths), 10)}: {path}")
            
            try:
                if not os.path.exists(path):
                    logging.error(f"❌ Video file not found: {path}")
                    continue
                
                # Check file size
                file_size = os.path.getsize(path)
                if file_size < 1024:  # Less than 1KB
                    logging.error(f"❌ Video file too small ({file_size} bytes): {path}")
                    continue
                
                logging.info(f"📊 Video file size: {file_size / (1024*1024):.2f} MB")
                
                # Try to load the clip with more detailed error handling
                try:
                    clip = mp.VideoFileClip(path,audio=False,verbose=False)
                    clips_to_close.append(clip)
                    
                    # Validate clip properties
                    if not hasattr(clip, 'duration') or clip.duration is None or clip.duration <= 0:
                        logging.error(f"❌ Invalid clip duration: {path}")
                        continue
                    
                    if not hasattr(clip, 'w') or not hasattr(clip, 'h') or clip.w <= 0 or clip.h <= 0:
                        logging.error(f"❌ Invalid clip dimensions: {path}")
                        continue
                    
                    logging.info(f"📐 Clip properties: {clip.w}x{clip.h}, {clip.duration:.2f}s, fps: {clip.fps}")
                    
                    if clip.duration < 0.1:  # Skip very short clips
                        logging.warning(f"⚠️ Skipping very short clip: {path} ({clip.duration:.2f}s)")
                        continue
                    
                    # Test if clip can be read
                    try:
                        test_frame = clip.get_frame(0)
                        if test_frame is None:
                            logging.error(f"❌ Cannot read frames from: {path}")
                            continue
                    except Exception as frame_error:
                        logging.error(f"❌ Frame reading test failed for {path}: {frame_error}")
                        continue
                    
                    # Resize and pad clip
                    try:
                        processed_clip = resize_and_pad_clip(clip)
                        if processed_clip is None:
                            logging.error(f"❌ Failed to process clip: {path}")
                            continue
                        
                        valid_clips.append(processed_clip)
                        logging.info(f"✅ Successfully loaded clip {i+1}: {path} ({clip.duration:.2f}s)")
                        
                    except Exception as resize_error:
                        logging.error(f"❌ Failed to resize clip {path}: {resize_error}")
                        continue
                        
                except Exception as clip_error:
                    logging.error(f"❌ Failed to load video clip {path}: {clip_error}")
                    # Try to provide more specific error information
                    if "codec" in str(clip_error).lower():
                        logging.error(f"💡 Codec issue detected. File may be corrupted or use unsupported codec.")
                    elif "permission" in str(clip_error).lower():
                        logging.error(f"💡 Permission issue detected. Check file permissions.")
                    continue
                    
            except Exception as e:
                logging.error(f"❌ Unexpected error processing {path}: {e}")
                continue
        
        logging.info(f"📊 Valid clips loaded: {len(valid_clips)} out of {len(video_paths)}")
        
        if not valid_clips:
            # Provide detailed error information
            logging.error("❌ No valid video clips could be loaded!")
            logging.error("🔍 Debugging information:")
            logging.error(f"   - Total video paths provided: {len(video_paths)}")
            logging.error(f"   - Video directory exists: {os.path.exists(Config.VIDEO_DIR)}")
            logging.error(f"   - Files in video directory: {os.listdir(Config.VIDEO_DIR) if os.path.exists(Config.VIDEO_DIR) else 'N/A'}")
            
            for i, path in enumerate(video_paths[:5]):  # Show first 5 for debugging
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    logging.error(f"   - File {i+1}: EXISTS, size: {size} bytes")
                else:
                    logging.error(f"   - File {i+1}: MISSING - {path}")
            
            raise VideoGenerationError("No valid video clips could be loaded. Check logs for detailed error information.")
        
        # Create video sequence to match audio duration
        video_sequence = []
        total_duration = 0
        clip_index = 0
        
        while total_duration < audio_duration:
            current_clip = valid_clips[clip_index % len(valid_clips)]
            remaining_time = audio_duration - total_duration
            
            if remaining_time <= 0:
                break
                
            # Use portion of clip that fits remaining time
            clip_duration = min(current_clip.duration, remaining_time)
            
            if clip_duration > 0:
                segment = current_clip.subclip(0, clip_duration)
                video_sequence.append(segment)
                total_duration += clip_duration
                
            clip_index += 1
            
            # Prevent infinite loop
            if clip_index > len(valid_clips) * 20:
                logging.warning("Breaking loop to prevent infinite iteration")
                break
        
        if not video_sequence:
            raise VideoGenerationError("No video sequence created")
        
        # Concatenate video clips
        logging.info("🔗 Concatenating video clips...")
        final_video = mp.concatenate_videoclips(video_sequence, method="compose")
        final_video = final_video.set_audio(audio)
        
        # Add captions
        text_clips = []
        for seg in captions:
            start_time = max(0, seg.get('start', 0))
            end_time = min(seg.get('end', start_time + 1), audio_duration)
            
            if end_time <= start_time:
                continue
                
            try:
                text_clip = mp.TextClip(
                    seg['text'],
                    fontsize=Config.FONT_SIZE,
                    color=Config.FONT_COLOR,
                    stroke_color=Config.STROKE_COLOR,
                    stroke_width=Config.STROKE_WIDTH,
                    font=Config.FONT_FAMILY,
                    size=(Config.TARGET_WIDTH - 100, 120),
                    method='caption'
                ).set_position(('center', Config.TARGET_HEIGHT - 180)).set_start(start_time).set_duration(end_time - start_time)
                
                text_clips.append(text_clip)
                clips_to_close.append(text_clip)
                
            except Exception as e:
                logging.warning(f"Failed to create text clip: {e}")
                continue
        
        # Compose final video
        logging.info("🎬 Creating final composition...")
        if text_clips:
            final_composition = mp.CompositeVideoClip([final_video] + text_clips)
        else:
            final_composition = final_video
            
        final_composition = final_composition.set_duration(audio_duration)
        
        # Write final video
        logging.info(f"💾 Writing video to: {output_path}")
        final_composition.write_videofile(
            output_path,
            codec=Config.VIDEO_CODEC,
            audio_codec=Config.AUDIO_CODEC,
            fps=Config.FPS,
            preset=Config.PRESET,
            bitrate=Config.BITRATE,
            verbose=False,
             # Suppress moviepy logs
        )
        
        logging.info("✅ Video composition completed")
        return output_path
        
    except Exception as e:
        logging.error(f"Video composition failed: {e}")
        raise VideoGenerationError(f"Failed to combine video, audio, and captions: {e}")
        
    finally:
        # Cleanup all clips
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        gc.collect()

def cleanup_files(*file_paths):
    """Clean up specified files and directories."""
    # Clean video directory
    if os.path.exists(Config.VIDEO_DIR):
        for filename in os.listdir(Config.VIDEO_DIR):
            filepath = os.path.join(Config.VIDEO_DIR, filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except OSError as e:
                logging.warning(f"Failed to remove {filepath}: {e}")
    
    # Clean specified files
    for filepath in file_paths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.debug(f"Cleaned up: {filepath}")
            except OSError as e:
                logging.warning(f"Failed to remove {filepath}: {e}")

def upload_to_cloudinary(video_path: str) -> Optional[str]:
    """Upload video to Cloudinary with error handling."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        logging.info("☁️ Uploading to Cloudinary...")
        result = cloudinary.uploader.upload_large(
            video_path,
            resource_type="video",
            folder="ai_videos",
            quality="auto",
            fetch_format="auto"
        )
        
        secure_url = result.get("secure_url")
        if secure_url:
            logging.info(f"✅ Upload successful: {secure_url}")
            return secure_url
        else:
            logging.error("No secure URL returned from Cloudinary")
            return None
            
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return None

def generate_full_video(user_prompt: str, user_script: str, user_keywords) -> Tuple[Optional[str], List[Dict], List[str]]:
    """Main function to generate complete video."""
    audio_path = None
    output_path = None
    
    try:
        # Validate inputs
        if not user_script or not user_script.strip():
            raise ValueError("Script cannot be empty")
        
        # Log inputs
        logging.info(f"📜 Prompt: {user_prompt}")
        logging.info(f"📝 Script length: {len(user_script)} characters")
        logging.info(f"🔑 Raw keywords: {user_keywords}")
        
        # Process keywords
        processed_keywords = normalize_keywords(user_keywords)
        logging.info(f"🔑 Processed keywords: {processed_keywords}")
        
        if not processed_keywords:
            logging.warning("⚠️ No valid keywords provided, using fallback keywords")
            processed_keywords = ["nature", "landscape", "abstract", "motion", "technology"]
        
        # Generate temporary file paths
        with temp_file(Config.AUDIO_SUFFIX) as audio_temp:
            with temp_file(Config.VIDEO_SUFFIX) as video_temp:
                audio_path = audio_temp
                output_path = video_temp
                
                # Step 1: Generate TTS
                logging.info("🔊 Generating text-to-speech...")
                asyncio.run(text_to_speech(user_script, audio_path))
                
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    raise VideoGenerationError("Failed to generate audio file")
                
                logging.info(f"🎵 Audio file created: {os.path.getsize(audio_path) / 1024:.1f} KB")
                
                # Step 2: Generate captions
                logging.info("🔠 Generating captions...")
                captions = generate_captions(audio_path)
                logging.info(f"📝 Generated {len(captions)} caption segments")
                
                # Step 3: Fetch and download videos
                logging.info("📥 Fetching stock videos...")
                video_urls = fetch_video_urls(processed_keywords)
                
                if not video_urls:
                    logging.error("❌ No video URLs found. Trying with broader keywords...")
                    # Fallback to more generic keywords
                    fallback_keywords = ["abstract", "motion", "color", "texture", "pattern"]
                    video_urls = fetch_video_urls(fallback_keywords)
                    
                    if not video_urls:
                        raise VideoGenerationError("No suitable videos found even with fallback keywords")
                
                logging.info(f"🔗 Found {len(video_urls)} video URLs")
                
                # Download videos with progress tracking
                video_paths = []
                successful_downloads = 0
                
                for i, url in enumerate(video_urls):
                    logging.info(f"📥 Downloading video {i+1}/{len(video_urls)}")
                    downloaded_path = download_video(url, f"video_{i}_{int(time.time())}.mp4")
                    
                    if downloaded_path:
                        video_paths.append(downloaded_path)
                        successful_downloads += 1
                        logging.info(f"✅ Download progress: {successful_downloads}/{len(video_urls)}")
                        
                        # Stop if we have enough videos
                        if successful_downloads >= 5:  # We only need a few good videos
                            logging.info("📊 Sufficient videos downloaded, proceeding...")
                            break
                    else:
                        logging.warning(f"⚠️ Failed to download video {i+1}")
                
                if not video_paths:
                    logging.error("❌ No videos were successfully downloaded!")
                    logging.error("🔍 Debugging video download:")
                    logging.error(f"   - Video URLs found: {len(video_urls)}")
                    logging.error(f"   - First few URLs: {video_urls[:3] if video_urls else 'None'}")
                    logging.error(f"   - Video directory: {Config.VIDEO_DIR}")
                    logging.error(f"   - Directory exists: {os.path.exists(Config.VIDEO_DIR)}")
                    
                    raise VideoGenerationError("Failed to download any videos")
                
                logging.info(f"📹 Successfully downloaded {len(video_paths)} videos")
                
                # Debug: List downloaded files
                if os.path.exists(Config.VIDEO_DIR):
                    files = os.listdir(Config.VIDEO_DIR)
                    logging.info(f"📂 Files in video directory: {files}")
                    
                    for file in files:
                        file_path = os.path.join(Config.VIDEO_DIR, file)
                        if os.path.isfile(file_path):
                            size = os.path.getsize(file_path)
                            logging.info(f"   - {file}: {size / (1024*1024):.2f} MB")
                
                # Step 4: Combine everything
                logging.info("🎞️ Starting video composition...")
                final_video_path = combine_video_audio_captions(video_paths, captions, audio_path, output_path)
                
                if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
                    raise VideoGenerationError("Final video file was not created or is empty")
                
                logging.info(f"🎬 Final video created: {os.path.getsize(final_video_path) / (1024*1024):.2f} MB")
                
                # Step 5: Upload to cloud
                cloud_url = upload_to_cloudinary(final_video_path)
                
                if cloud_url:
                    logging.info(f"🎉 Video generation completed successfully!")
                    logging.info(f"📤 Cloud URL: {cloud_url}")
                    return cloud_url, captions, processed_keywords
                else:
                    logging.warning("Video generated locally but cloud upload failed")
                    return final_video_path, captions, processed_keywords
                
    except Exception as e:
        logging.error(f"❌ Video generation failed: {e}")
        logging.error(f"💥 Error type: {type(e).__name__}")
        
        # Add more debugging information
        if "usable clips" in str(e).lower():
            logging.error("🔍 'No usable clips' error debugging:")
            logging.error(f"   - Check if video downloads are completing successfully")
            logging.error(f"   - Verify video files are not corrupted")
            logging.error(f"   - Ensure moviepy can read the downloaded video format")
            logging.error(f"   - Try with different keywords if current ones return poor quality videos")
        
        return None, [], []
        
    finally:
        # Cleanup
        try:
            cleanup_files(audio_path, output_path)
            gc.collect()
            logging.info("🧹 Cleanup completed")
        except Exception as cleanup_error:
            logging.warning(f"Cleanup failed: {cleanup_error}")

# Example usage and diagnostics
def run_diagnostics():
    """Run diagnostic checks to identify potential issues."""
    logging.info("🔧 Running diagnostics...")
    
    # Check directories
    logging.info(f"📁 Video directory exists: {os.path.exists(Config.VIDEO_DIR)}")
    if not os.path.exists(Config.VIDEO_DIR):
        try:
            os.makedirs(Config.VIDEO_DIR, exist_ok=True)
            logging.info("✅ Created video directory")
        except Exception as e:
            logging.error(f"❌ Failed to create video directory: {e}")
    
    # Check API keys
    logging.info(f"🔑 Pexels API key configured: {bool(PEXELS_API_KEY)}")
    logging.info(f"🔑 Gemini API key configured: {bool(GEMINI_API_KEY)}")
    
    # Test a simple API call
    if PEXELS_API_KEY:
        try:
            headers = {"Authorization": PEXELS_API_KEY}
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": "nature", "per_page": 1},
                timeout=10
            )
            logging.info(f"🌐 Pexels API test response: {response.status_code}")
        except Exception as e:
            logging.error(f"❌ Pexels API test failed: {e}")
    
    # Check available disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        logging.info(f"💾 Available disk space: {free_space:.2f} GB")
    except Exception as e:
        logging.warning(f"Could not check disk space: {e}")
    
    # Test moviepy
    try:
        # Create a test clip
        test_clip = mp.ColorClip(size=(640, 480), color=(255, 0, 0), duration=1)
        test_clip.close()
        logging.info("✅ MoviePy working correctly")
    except Exception as e:
        logging.error(f"❌ MoviePy test failed: {e}")
    
    logging.info("🔧 Diagnostics completed")

