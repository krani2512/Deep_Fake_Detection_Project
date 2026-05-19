import cv2, os

def extract_frames(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224,224))
        cv2.imwrite(f"{out_dir}/frame_{count}.jpg", frame)
        count += 1
    cap.release()

extract_frames("uploaded_files/new_demo_300.mp4", "static/dataset")
