# video_gen.py

import os
import re
import requests
import asyncio
import whisper
import edge_tts
import moviepy.editor as mp
import cloudinary
import cloudinary.uploader
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

AUDIO_PATH = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
VIDEO_DIR = "./videos"
OUTPUT_PATH = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

os.makedirs("videos", exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

async def text_to_speech(text):
    communicate = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural",
        rate="+0%",
        pitch="+0Hz"
    )
    await communicate.save(AUDIO_PATH)

def generate_captions():
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(AUDIO_PATH, verbose=False, word_timestamps=True)
    for seg in result["segments"]:
        seg["text"] = re.sub(r'[\*\[\]]+', '', seg["text"])
    return result["segments"]

def fetch_video_urls(keywords, per_keyword=2):
    headers = {"Authorization": PEXELS_API_KEY}
    urls = []
    for keyword in keywords:
        params = {
            "query": keyword,
            "per_page": 10,
            "orientation": "portrait",
            "size": "medium"
        }
        try:
            res = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
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
            print(f"⚠️ Error fetching videos for '{keyword}': {e}")
    return urls

def download_video(url, filename):
    path = os.path.join(VIDEO_DIR, filename)
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path
    except Exception as e:
        print(f"⚠️ Download error: {e}")
    return None

def combine_video_audio_captions(video_paths, captions):
    audio = mp.AudioFileClip(AUDIO_PATH)
    audio_duration = audio.duration
    clips = []
    for path in video_paths:
        try:
            clip = mp.VideoFileClip(path).resize(height=1280).crop(x_center=720, width=720)
            if clip.duration >= 1:
                clips.append(clip)
        except Exception as e:
            print(f"⚠️ Video load error: {e}")

    looped = []
    total = 0
    i = 0
    while total < audio_duration:
        clip = clips[i % len(clips)]
        duration = min(clip.duration, audio_duration - total)
        looped.append(clip.subclip(0, duration))
        total += duration
        i += 1

    video = mp.concatenate_videoclips(looped).set_audio(audio)

    txt_clips = []
    for seg in captions:
        start = seg['start']
        end = min(seg['end'], audio_duration)
        text = seg['text']
        txt = mp.TextClip(
            text,
            fontsize=48,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            font='Arial',
            size=(video.w - 100, 160)
        ).set_position(('center', video.h - 200)).set_start(start).set_duration(end - start)
        txt_clips.append(txt)

    final = mp.CompositeVideoClip([video] + txt_clips).set_duration(audio_duration)
    final.write_videofile(
    OUTPUT_PATH,
    codec="libx264",
    audio_codec="aac",
    bitrate="400k",  # lower bitrate
    fps=24,
    preset="ultrafast",
    threads=2
)

    for c in clips + looped + txt_clips:
        c.close()
    audio.close()
    final.close()

    return OUTPUT_PATH

def cleanup():
    for f in os.listdir("videos"):
        os.remove(os.path.join("videos", f))
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)

def generate_full_video(user_prompt, user_script, user_keywords):
    try:
        print("📜 Prompt:", user_prompt)
        print("📝 Script:", user_script)
        print("🔑 Keywords:", user_keywords)

        print("🔊 Generating TTS...")
        asyncio.run(text_to_speech(user_script))

        print("🔠 Generating captions...")
        captions = generate_captions()

        print("📥 Downloading stock videos...")
        urls = fetch_video_urls(user_keywords)
        video_paths = []
        for i, url in enumerate(urls):
            path = download_video(url, f"video_{i}.mp4")
            if path:
                video_paths.append(path)

        if not video_paths:
            raise Exception("❌ No valid video clips found.")

        print("🎞️ Combining video/audio/captions...")
        output_path = combine_video_audio_captions(video_paths, captions)
        print("✅ Video generated locally:", output_path)

        print("☁️ Uploading to Cloudinary...")
        cloudinary_upload = cloudinary.uploader.upload_large(output_path, resource_type="video", folder="ai_videos")
        cloudinary_url = cloudinary_upload.get("secure_url")

        os.remove(output_path)

        print("📤 Uploaded to:", cloudinary_url)
        return cloudinary_url, captions, user_keywords

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, [], []

    finally:
        cleanup()
