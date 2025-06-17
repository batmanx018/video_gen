import os
import asyncio
import requests
import re
from dotenv import load_dotenv
import whisper
import edge_tts
import cloudinary
import cloudinary.uploader
import google.generativeai as genai
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, AudioFileClip,
    TextClip, CompositeVideoClip
)

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
CLOUDINARY_CLOUD = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

AUDIO_PATH = "./audio/output.mp3"
VIDEO_DIR = "./videos"
os.makedirs("audio", exist_ok=True)
os.makedirs("videos", exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def extract_keywords_manually(text):
    return [re.sub(r'^[\d\-\*\.\)\(]+\s*', '', k.strip()) for k in text.strip().splitlines() if k.strip()]

async def text_to_speech(text):
    communicate = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural",
        rate="+0%", pitch="+0Hz"
    )
    await communicate.save(AUDIO_PATH)

def generate_captions(audio_file=AUDIO_PATH):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, verbose=False, word_timestamps=True)
    for seg in result["segments"]:
        seg["text"] = re.sub(r'[\*\[\]]+', '', seg["text"])
    return result["segments"]

def fetch_video_urls(keywords, per_keyword=2):
    headers = {"Authorization": PEXELS_API_KEY}
    urls = []
    for keyword in keywords:
        try:
            res = requests.get("https://api.pexels.com/videos/search", headers=headers, params={
                "query": keyword, "per_page": 10, "orientation": "landscape"
            })
            videos = res.json().get("videos", [])
            count = 0
            for video in videos:
                if count >= per_keyword:
                    break
                files = sorted(video["video_files"], key=lambda x: x["width"], reverse=True)
                for f in files:
                    if 1.75 < (f["width"] / f["height"]) < 1.79:
                        urls.append(f["link"])
                        count += 1
                        break
        except Exception as e:
            print(f"Error fetching for '{keyword}': {e}")
    return urls

def download_video(url, filename):
    path = os.path.join(VIDEO_DIR, filename)
    try:
        res = requests.get(url, stream=True, timeout=30)
        if res.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in res.iter_content(8192):
                    f.write(chunk)
            return path
    except Exception as e:
        print("Download error:", e)
    return None

def combine_videos_audio_captions(video_paths, captions, audio_file=AUDIO_PATH):
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration

    clips = []
    for path in video_paths:
        try:
            clip = VideoFileClip(path).resize(height=720)
            if clip.duration >= 1:
                clips.append(clip)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not clips:
        raise Exception("No valid videos.")

    looped_clips, total, i = [], 0, 0
    while total < audio_duration:
        clip = clips[i % len(clips)]
        dur = min(clip.duration, audio_duration - total)
        looped_clips.append(clip.subclip(0, dur))
        total += dur
        i += 1

    video = concatenate_videoclips(looped_clips).set_duration(audio_duration)
    video = video.set_audio(audio.subclip(0, audio_duration))

    caption_clips = []
    for seg in captions:
        if seg["start"] < audio_duration:
            try:
                txt = TextClip(seg["text"].strip(), fontsize=40, color="white", stroke_color="black", stroke_width=2,
                               font="DejaVu-Sans", method="caption", size=(video.size[0] - 100, 120))
                caption_clips.append(txt.set_position(('center', video.size[1] - 160))
                                     .set_start(seg["start"]).set_duration(min(seg["end"], audio_duration) - seg["start"]))
            except Exception as e:
                print("Caption error:", e)

    final = CompositeVideoClip([video] + caption_clips).set_duration(audio_duration)
    output_file = "final_output.mp4"
    final.write_videofile(output_file, codec="libx264", audio_codec="aac", remove_temp=True)
    return output_file

def generate_full_video(prompt, script, keywords):
    try:
        asyncio.run(text_to_speech(script))
        captions = generate_captions()
        keyword_list = extract_keywords_manually(keywords)
        urls = fetch_video_urls(keyword_list)
        paths = [download_video(url, f"v{i}.mp4") for i, url in enumerate(urls) if download_video(url, f"v{i}.mp4")]
        if not paths:
            raise Exception("No videos downloaded.")
        output_path = combine_videos_audio_captions(paths, captions)

        upload = cloudinary.uploader.upload_large(output_path, resource_type="video", folder="ai_videos")
        return upload["secure_url"], captions, keyword_list

    except Exception as e:
        print("Error:", e)
        return None, [], []
