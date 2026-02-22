# Game Video Scene Analyzer

## Overview
Bachelor’s thesis project focused on automatic recognition of scene transitions in recorded gameplay footage.

The system detects transitions between cutscenes and gameplay segments using frame-based video analysis.

## Problem Statement
Manual segmentation of gameplay footage is time-consuming.  
This project implements an automated solution based on frame comparison and pixel-level analysis.

## Technologies Used
- Python
- OpenCV
- NumPy

## Methodology
- Extract frames from video
- Compute pixel difference between consecutive frames
- Apply threshold-based change detection
- Classify scene transitions

## Results
The system successfully detected scene changes with consistent accuracy during testing on sample gameplay footage.

## Project Structure
- `/src` – Python implementation
- `/output_examples` – sample detection results
- `/docs` – academic documentation

## How to Run

1. Install dependencies:
pip install opencv-python numpy

2. Run:
python main.py
