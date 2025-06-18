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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

load_dotenv()

# Paths
AUDIO_PATH = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
VIDEO_DIR = "./videos"
OUTPUT_PATH = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
os.makedirs(VIDEO_DIR, exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(AUDIO_PATH)

def generate_captions():
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(AUDIO_PATH, word_timestamps=True, verbose=False)
    for seg in result["segments"]:
        seg["text"] = re.sub(r'[\*\[\]]+', '', seg["text"])
    return result["segments"]

def fetch_video_urls(keywords, per_keyword=2):
    headers = {"Authorization": PEXELS_API_KEY}
    urls = []
    for keyword in keywords:
        try:
            res = requests.get("https://api.pexels.com/videos/search", headers=headers, params={
                "query": keyword,
                "per_page": 10,
                "orientation": "portrait",
                "size": "medium"
            })
            videos = res.json().get("videos", [])
            count = 0
            for video in videos:
                if count >= per_keyword:
                    break
                video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
                for vf in video_files:
                    if 0.55 < vf.get("width", 1)/vf.get("height", 1) < 0.60:
                        urls.append(vf["link"])
                        count += 1
                        break
        except Exception as e:
            logging.warning(f"Error fetching for '{keyword}': {e}")
    return urls

def download_video(url, filename):
    path = os.path.join(VIDEO_DIR, filename)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return path
    except Exception as e:
        logging.warning(f"Download error: {e}")
        return None

def combine_video_audio_captions(video_paths, captions):
    audio = mp.AudioFileClip(AUDIO_PATH)
    clips, looped, txt_clips = [], [], []

    audio_duration = audio.duration
    total, i = 0, 0

    for path in video_paths:
        try:
            clip = mp.VideoFileClip(path).resize(height=1280).crop(x_center=720, width=720)
            clips.append(clip)
        except Exception as e:
            logging.warning(f"Clip load failed: {e}")

    while total < audio_duration:
        clip = clips[i % len(clips)]
        dur = min(clip.duration, audio_duration - total)
        looped.append(clip.subclip(0, dur))
        total += dur
        i += 1

    video = mp.concatenate_videoclips(looped, method="compose").set_audio(audio)

    for seg in captions:
        start, end = seg['start'], min(seg['end'], audio_duration)
        txt = mp.TextClip(
            seg['text'],
            fontsize=40,
            color='white',
            stroke_color='black',
            stroke_width=2,
            font='Arial',
            size=(video.w - 100, 120),
            method='caption'
        ).set_position(('center', video.h - 180)).set_start(start).set_duration(end - start)
        txt_clips.append(txt)

    final = mp.CompositeVideoClip([video] + txt_clips).set_duration(audio_duration)
    final.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast", bitrate="400k")

    # Cleanup clips
    for clip in clips + looped + txt_clips:
        clip.close()
    audio.close()
    final.close()
    gc.collect()

    return OUTPUT_PATH

def cleanup():
    for f in os.listdir(VIDEO_DIR):
        os.remove(os.path.join(VIDEO_DIR, f))
    if os.path.exists(AUDIO_PATH): os.remove(AUDIO_PATH)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)

def generate_full_video(user_prompt, user_script, user_keywords):
    try:
        logging.info(f"📜 Prompt: {user_prompt}")
        logging.info(f"📝 Script: {user_script}")
        logging.info(f"🔑 Keywords: {user_keywords}")

        logging.info("🔊 Generating TTS...")
        asyncio.run(text_to_speech(user_script))

        logging.info("🔠 Generating captions...")
        captions = generate_captions()

        logging.info("📥 Downloading stock videos...")
        urls = fetch_video_urls(user_keywords)
        video_paths = [download_video(url, f"video_{i}.mp4") for i, url in enumerate(urls)]
        video_paths = [v for v in video_paths if v]

        if not video_paths:
            raise Exception("No valid video clips downloaded.")

        logging.info("🎞️ Combining video/audio/captions...")
        output_path = combine_video_audio_captions(video_paths, captions)
        logging.info(f"✅ Video generated locally at: {output_path}")

        logging.info("☁️ Uploading to Cloudinary...")
        cloud_url = cloudinary.uploader.upload_large(output_path, resource_type="video", folder="ai_videos").get("secure_url")

        logging.info(f"📤 Uploaded to: {cloud_url}")
        return cloud_url, captions, user_keywords

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return None, [], []

    finally:
        cleanup()
        gc.collect()
