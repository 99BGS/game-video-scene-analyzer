# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:53:01 2025

@author: BGS
"""

import cv2

video_path = r"C:\Users\BGS\Desktop\Licenta\rdr2_scene.mp4.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print("FPS =", fps)
