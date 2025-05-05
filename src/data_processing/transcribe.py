import whisper
import json
import os

# Explicitly set the path to ffmpeg
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Load Whisper model
model = whisper.load_model("medium")

# Transcribe audio
result = model.transcribe("data/video.mp4")

# Save transcription as JSON
with open("data/transcripts/transcript.json", "w") as f:
    json.dump(result, f, indent=4)

print("✅ Transcription completed and saved.")