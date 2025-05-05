import cv2
import os

# Video path
video_path = 'data/video.mp4'
frames_dir = 'data/frames/'

# Create frames directory if doesn't exist
os.makedirs(frames_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract a frame every 2 seconds
interval_seconds = 2
interval_frames = int(fps * interval_seconds)

count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % interval_frames == 0:
        frame_time = int(count / fps)
        frame_filename = os.path.join(frames_dir, f"frame_{frame_time}s.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"✅ Extracted and saved {saved_count} frames.")