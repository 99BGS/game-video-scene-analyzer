# final_transition_detector.py
# Usage: edit video_path below, then run.
# Requires: opencv, numpy, scikit-image, matplotlib
# Install if needed: conda install -c conda-forge opencv scikit-image matplotlib
# (or pip install opencv-python scikit-image matplotlib)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------- USER PARAMETERS --------------------
video_path = r"C:\Users\BGS\Desktop\Licenta\uncharted_scene.mp4"  # <-- set your file here
output_folder = r"C:\Users\BGS\Desktop\Licenta\results"         # images/plots saved here
top_crop = 40            # pixels to crop from top (ignore letterbox)
bottom_crop = 40         # pixels to crop from bottom
min_frames_before_detect = 30   # how many frames to collect before detection logic starts
prev_window = 15         # number of frames used to compute "previous" average
curr_window = 6          # number of frames used to compute "current" average
motion_delta_thresh = 0.18   # threshold for motion change
entropy_delta_thresh = 0.12  # threshold for entropy change
corner_delta_thresh = 7      # threshold for corners change
consistency_required = 5     # number of consecutive frames that must satisfy the condition
# ---------------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

def calc_entropy(gray_frame):
    hist = cv2.calcHist([gray_frame],[0],None,[256],[0,256]).ravel()
    hist_sum = hist.sum()
    if hist_sum <= 0:
        return 0.0
    hist = hist / hist_sum
    logs = np.log2(hist + 1e-12)
    entropy = -np.sum(hist * logs)
    return float(entropy)

def calc_corners(gray_frame):
    # GoodFeaturesToTrack is fast and indicates texture/HUD presence
    corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=8)
    return 0 if corners is None else len(corners)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"ERROR: Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Video opened. FPS = {fps:.3f}")

prev_gray = None
frame_index = 0

motion_hist = []
entropy_hist = []
corner_hist = []
time_stamps = []

transition_frame = None
consec_counter = 0

# Read frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop top/bottom to ignore letterbox bands
    h = frame.shape[0]
    if top_crop + bottom_crop >= h - 2:
        # avoid invalid crop
        proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        proc_frame = frame[top_crop:h-bottom_crop, :]
        proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

    # compute metrics relative to prev frame
    if prev_gray is not None:
        # optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, proc, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mean_motion = float(np.mean(mag))
        motion_hist.append(mean_motion)

        # entropy
        ent = calc_entropy(proc)
        entropy_hist.append(ent)

        # corners / texture
        corners = calc_corners(proc)
        corner_hist.append(corners)

        time_stamps.append(frame_index / fps)

        # detection logic - only after enough frames collected
        if len(motion_hist) >= prev_window + curr_window and frame_index > min_frames_before_detect:
            # rolling windows
            prev_motion = np.mean(motion_hist[-(prev_window+curr_window):-curr_window])
            curr_motion = np.mean(motion_hist[-curr_window:])

            prev_entropy = np.mean(entropy_hist[-(prev_window+curr_window):-curr_window])
            curr_entropy = np.mean(entropy_hist[-curr_window:])

            prev_corners = np.mean(corner_hist[-(prev_window+curr_window):-curr_window])
            curr_corners = np.mean(corner_hist[-curr_window:])

            # absolute deltas
            dm = abs(curr_motion - prev_motion)
            de = abs(curr_entropy - prev_entropy)
            dc = abs(curr_corners - prev_corners)

            # check thresholds (all three conditions should be reasonably met)
            cond_motion = dm > motion_delta_thresh
            cond_entropy = de > entropy_delta_thresh
            cond_corners = dc > corner_delta_thresh

            if cond_motion and cond_entropy and cond_corners:
                consec_counter += 1
            else:
                consec_counter = 0

            if consec_counter >= consistency_required:
                # transition detected at current frame index
                transition_frame = frame_index
                print(f"Transition detected at frame {transition_frame} (consecutive={consec_counter})")
                break

    prev_gray = proc.copy()
    frame_index += 1

cap.release()

# Output results
if transition_frame is None:
    print("No transition detected by automatic detector.")
else:
    time_sec = transition_frame / fps
    print(f"Detected transition at ~ {time_sec:.2f} seconds (frame {transition_frame})")

    # Re-open video to grab exact frame image to save
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, transition_frame-1))
    ret, frame_img = cap.read()
    cap.release()
    if ret:
        save_path = os.path.join(output_folder, f"transition_frame_{transition_frame}.jpg")
        cv2.imwrite(save_path, frame_img)
        print("Saved transition frame image to:", save_path)
    else:
        print("Could not extract image at detected frame.")

# Plot diagnostics (motion, entropy, corners) with marker
try:
    x = np.array(time_stamps)
    plt.figure(figsize=(11,6))
    if len(motion_hist) > 0:
        plt.plot(x, motion_hist, label='Mean optical flow magnitude')
    if len(entropy_hist) > 0:
        plt.plot(x, entropy_hist, label='Entropy (image)')
    if len(corner_hist) > 0:
        plt.plot(x, corner_hist, label='Corner count (texture/HUD)')
    if transition_frame is not None:
        tx = transition_frame / fps
        plt.axvline(tx, color='k', linestyle='--', label=f'Detected transition ~{tx:.2f}s')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title('Transition detector diagnostics')
    plot_path = os.path.join(output_folder, "transition_diagnostics.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print("Saved diagnostics plot to:", plot_path)
except Exception as e:
    print("Plotting failed:", e)
