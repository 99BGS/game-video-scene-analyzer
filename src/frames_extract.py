import cv2
import os

video_path = r"C:\Users\BGS\Desktop\Licenta\uncharted_scene.mp4"
output_folder = r"C:\Users\BGS\Desktop\Licenta\frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
    cv2.imwrite(frame_path, frame)
    
    frame_index += 1

cap.release()
print("Done! Frames extracted:", frame_index)
