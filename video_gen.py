import os
import re
import requests
import asyncio
import whisper
import edge_tts
import tempfile
import subprocess
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables
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

os.makedirs(VIDEO_DIR, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# 🗣️ Text to Speech
async def text_to_speech(text):
    communicate = edge_tts.Communicate(
        text, voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz"
    )
    await communicate.save(AUDIO_PATH)

# 🎯 Generate captions using Whisper
def generate_captions():
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(AUDIO_PATH, verbose=False, word_timestamps=True)
    for seg in result["segments"]:
        seg["text"] = re.sub(r'[\*\[\]]+', '', seg["text"])
    return result["segments"]

# 📝 Write SRT subtitle file
def write_srt_file(captions, srt_path):
    def format_time(seconds):
        hrs, secs = divmod(int(seconds), 3600)
        mins, secs = divmod(secs, 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(captions, 1):
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")

# 📹 Get videos from Pexels
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

# ⬇️ Download video
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

# 🎞️ Combine videos using ffmpeg
def combine_video_audio_captions_ffmpeg(video_paths, captions):
    concat_list_path = os.path.join(VIDEO_DIR, "inputs.txt")
    with open(concat_list_path, "w") as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    temp_concat = os.path.join(VIDEO_DIR, "combined.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy", temp_concat
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with_audio = os.path.join(VIDEO_DIR, "with_audio.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_concat, "-i", AUDIO_PATH,
        "-c:v", "copy", "-c:a", "aac", "-shortest", with_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    srt_path = os.path.join(VIDEO_DIR, "subtitles.srt")
    write_srt_file(captions, srt_path)

    subprocess.run([
        "ffmpeg", "-y", "-i", with_audio, "-vf", f"subtitles={srt_path},scale=720:-2",
        "-preset", "ultrafast", "-b:v", "400k", "-bufsize", "512k",
        OUTPUT_PATH
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return OUTPUT_PATH

# 🧹 Clean up
def cleanup():
    for f in os.listdir(VIDEO_DIR):
        os.remove(os.path.join(VIDEO_DIR, f))
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)

# 🔁 Main function
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
        output_path = combine_video_audio_captions_ffmpeg(video_paths, captions)
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
