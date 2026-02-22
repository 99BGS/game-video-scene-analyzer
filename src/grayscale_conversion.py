import cv2

img = cv2.imread("C:/Users/BGS/Desktop/Licenta/frames/frame_340.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("C:/Users/BGS/Desktop/Licenta/results/gray_frame_340.jpg", gray)
