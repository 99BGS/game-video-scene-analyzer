import cv2
import os
import numpy as np

frames_folder = r"C:\Users\BGS\Desktop\Licenta\frames"

frame_files = sorted(os.listdir(frames_folder))
differences = []

# calculăm diferențele dintre frame-uri
for i in range(len(frame_files) - 1):
    frame1_path = os.path.join(frames_folder, frame_files[i])
    frame2_path = os.path.join(frames_folder, frame_files[i+1])
    
    f1 = cv2.imread(frame1_path)
    f2 = cv2.imread(frame2_path)
    
    f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(f1_gray, f2_gray)
    diff_score = diff.sum()
    differences.append(diff_score)

# convertim în array și normalizăm
diff = np.array(differences)
diff_norm = diff / diff.max()

# setăm pragul mai mare și ignorăm primele 10 cadre
threshold = 0.5  # prag mai mare
start_ignore = 10  # ignorăm primele 10 frame-uri
candidates = np.where(diff_norm[start_ignore:] > threshold)[0]

if len(candidates) > 0:
    scene_change_frame = candidates[0] + start_ignore
    print("First real scene change at frame:", scene_change_frame)
    fps = 29.97
    scene_time_sec = scene_change_frame / fps
    print(f"Approximate time of scene change: {scene_time_sec:.2f} seconds")
else:
    print("No significant scene change detected.")
