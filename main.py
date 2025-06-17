from fastapi import FastAPI, Form
from video_gen import generate_full_video
from fastapi.responses import JSONResponse
app = FastAPI()
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/generate-video/")
async def generate_video(
    prompt: str = Form(...),
    script: str = Form(...),
    keywords: str = Form(...)
):
    output_url, captions, keyword_list = generate_full_video(prompt, script, keywords)
    if output_url:
        return JSONResponse(content={
            "status": "success",
            "url": output_url,
            "captions": captions,
            "keywords": keyword_list
        })
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Video generation failed."})
