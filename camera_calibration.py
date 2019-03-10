import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Collect all calibration images by name
images = glob.glob("camera_cal/calibration*.jpg")

# Initialize array to store corners
objpoints = []      # Real 3D objectpoints
imgpoints = []    # 2D points in image

for img_name in images:
    img = cv2.imread(img_name)
    # coordinates of this image object points
    if img_name == "camera_cal/calibration1.jpg":
        nx, ny = 9, 5
    elif img_name == "camera_cal/calibration4.jpg":
        nx, ny = 7, 4
    elif img_name == "camera_cal/calibration5.jpg":
        nx, ny = 7, 5
    else:
        nx, ny = 9, 6
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, save object points and corners as image points
    if ret:
        # Save the corners (2D points) and the objp (3D locations) on our arrays
        imgpoints.append(corners)
        objpoints.append(objp)

# Perform camera calibration with data points over all 20 images
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# Save data into file
save_data = {'mtx': mtx, 'dist': dist}
pickle.dump(save_data, open('calibrate_camera.p', 'wb'))

img = cv2.imread("camera_cal/calibration1.jpg")
nx, ny = 9, 5
# Find corners and calibrated images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
dst_all = cv2.undistort(img, mtx, dist, None, mtx)
# Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2.5))
f.tight_layout()
for ax in (ax1, ax2):
    ax.axis('off')
ax1.imshow(img)
ax1.set_title("Original image", fontsize=18)
ax2.imshow(dst_all)
ax2.set_title("Undistorted image", fontsize=18)
plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
plt.show()
f.savefig("chessboard_correct.jpg")

plot = False
if plot:
    idx = 0
    for img_name in images:
        img = cv2.imread(img_name)
        if img_name == "camera_cal/calibration1.jpg":
            nx, ny = 9, 5
        elif img_name == "camera_cal/calibration4.jpg":
            nx, ny = 7, 4
        elif img_name == "camera_cal/calibration5.jpg":
            nx, ny = 7, 5
        else:
            nx, ny = 9, 6
        # Find corners and calibrated images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        ret, mtx_one, dist_one, rvecs, tvecs = cv2.calibrateCamera([objpoints[idx]], [imagepoints[idx]], gray.shape[::-1], None, None)
        dst_one = cv2.undistort(img, mtx_one, dist_one, None, mtx_one)
        dst_all = cv2.undistort(img, mtx, dist, None, mtx)
        # Plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,8), sharex=True)
        ax1.imshow(img)
        ax1.set_title(img_name)
        ax2.imshow(dst_one)
        ax2.set_title("Using corners in this image")
        ax3.imshow(dst_all)
        ax3.set_title("Using all corners")
        plt.show()
        idx += 1
