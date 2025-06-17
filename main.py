from flask import Flask, request, jsonify
from video_gen import generate_full_video

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/generate-video/", methods=["POST"])
def generate_video():
    prompt = request.form.get("prompt")
    script = request.form.get("script")
    keywords = request.form.get("keywords")

    if not prompt or not script or not keywords:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    output_url, captions, keyword_list = generate_full_video(prompt, script, keywords)

    if output_url:
        return jsonify({
            "status": "success",
            "url": output_url,
            "captions": captions,
            "keywords": keyword_list
        })
    else:
        return jsonify({"status": "error", "message": "Video generation failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
