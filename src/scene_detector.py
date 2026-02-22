# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:50:23 2025

@author: BGS
"""

import cv2
import os
import numpy as np

frames_folder = r"C:\Users\BGS\Desktop\Licenta\frames"

differences = []

# numărul total de frame-uri
frame_files = sorted(os.listdir(frames_folder))

for i in range(len(frame_files) - 1):
    frame1_path = os.path.join(frames_folder, frame_files[i])
    frame2_path = os.path.join(frames_folder, frame_files[i+1])
    
    f1 = cv2.imread(frame1_path)
    f2 = cv2.imread(frame2_path)
    
    # convertim în grayscale
    f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    # diferența absolută
    diff = cv2.absdiff(f1_gray, f2_gray)
    diff_score = diff.sum()  # scorul total al diferenței
    
    differences.append(diff_score)

# găsim frame-ul cu cea mai mare diferență (schimbare de scenă)
scene_change_frame = np.argmax(differences)

print("Scene changes at frame:", scene_change_frame)
