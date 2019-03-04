import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle


try:
    saved_dist = pickle.load(open('calibrate_camera.p', 'rb'), encoding='latin1')
    mtx = saved_dist['mtx']
    dist = saved_dist['dist']
except (OSError, IOError):  # No progress file yet available
    print("No saved distorsion data. Run camera_calibration.py")

# Collect all example images by name
images = glob.glob("test_images/*.jpg")
print("IM: ",images)

for img_name in images:
    img = mpimg.imread(img_name)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.imshow(img)
    ax1.set_title(img_name)
    ax2.imshow(dst)
    ax2.set_title("Undistorted image")
    plt.show()
