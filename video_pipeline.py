import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from image_thresholding import *
from plotting_helpers import *
from line_fit import *
from Line import *


# *** PIPELINE ***
def pipeline(img):
    global error_im, skipped_frames

    # 1. Correct distorsion
    # open distorsion matrix
    try:
        saved_dist = pickle.load(open('calibrate_camera.p', 'rb'), encoding='latin1')
        mtx = saved_dist['mtx']
        dist = saved_dist['dist']
    except (OSError, IOError):  # No progress file yet available
        print("No saved distorsion data. Run camera_calibration.py")
    # apply correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # 2. Apply filters to get binary map
    ksize = 3
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(5, 100))
    mag_bin = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(10, 200))
    dir_bin = dir_threshold(undist, sobel_kernel=15, thresh=(0.9, 1.2))
    hls_bin = hls_select(img, thresh=(50, 255))
    white_bin = white_select(img, thresh=195)
    yellow_bin = yellow_select(img)
    # combine filters to a final output
    combined = np.zeros_like(dir_bin)
    combined[((mag_bin == 1) & (dir_bin == 1) & (hls_bin == 1)) |
             ((white_bin == 1) | (yellow_bin == 1))] = 1

    # 3. Define trapezoid points on the road and transform perspective
    X = combined.shape[1]
    Y = combined.shape[0]
    src = np.float32(
            [[205, 720],
             [1075, 720],
             [700, 460],
             [580, 460]])
    dst = np.float32(
            [[300, 720],
             [980, 720],
             [980, 0],
             [300, 0]])
    # get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warp the result of binary thresholds
    warped = cv2.warpPerspective(combined, M, (X,Y), flags=cv2.INTER_LINEAR)

    # 4. Get polinomial fit of lines
    # if > 4 frames skipped (or first frame, as skipped_frames is initialized to 100) do full search
    if skipped_frames > 5:
        fit_method = "Boxes"
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
    else:
        fit_method = "Around fit"
        leftx, lefty, rightx, righty, out_img = find_lane_around_fit(warped, left_lane.fit_x, right_lane.fit_x)

    # fit polynomials and sanity check
    try:
        left_fit, right_fit, left_px, right_px, ploty = fit(leftx, lefty, rightx, righty, warped.shape[0])
        detected, err_msg = sanity_chk(ploty, left_px, right_px)
    except:
        detected, err_msg = False, "Empty data"

    if detected: skipped_frames = 0
    else:        skipped_frames += 1

    # 5. Calculate distance to center, curvature, and update Line objects
    if detected or (fit_method == "Boxes" and err_msg != "Empty data"):
        left_curv, right_curv = find_curv(ploty, left_fit, right_fit)
        left_lane.update(ploty, left_fit, left_px, left_curv)
        right_lane.update(ploty, right_fit, right_px, right_curv)
    lane_w = (right_lane.base_pos - left_lane.base_pos) * 3.7/700
    offset = (((right_lane.base_pos + left_lane.base_pos) - img.shape[1]) / 2) * 3.7/700

    # 6. Plot fitted lanes into original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.fit_x, left_lane.fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.fit_x, right_lane.fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # if error save original img to check closely in image pipeline
    if 1 < skipped_frames < 3:
        mpimg.imsave(err_msg + "_" + str(error_im) + ".jpg", img)
        error_im += 1

    # Add text
    road_curv = (left_lane.curv_avg + right_lane.curv_avg) // 2
    if road_curv > 2000:
        road_curv_text = "Road curvature: straight"
    else:
        road_curv_text = "Road curvature: " + str(road_curv) + "m"
    side = {True: "left", False: "right"}
    offset_txt = "Car is {0:.2f}m {1:s} of center".format(offset, side[offset > 0])

    for i, txt in enumerate([road_curv_text, offset_txt]):
        cv2.putText(result, txt, (75, 75 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Uncomment for debugging messages
    # lane_width_txt = "Lane width: %.2f m" % lane_w
    # for i, obj, txt in [(1, left_lane, "Left"), (2, right_lane, "Right")]:
    #     if obj.curv_avg > 2000:
    #         curv_txt = txt + " curvature: straight"
    #     else:
    #         curv_txt = txt + " curvature: " + str(int(obj.curv_avg)) + "m"
    #     cv2.putText(result,curv_txt, (550, 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    # cv2.putText(result, "Skipped frames: " + str(skipped_frames), (550,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    # cv2.putText(result, fit_method, (550, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    # if err_msg != "":
    #     cv2.putText(result, "Error!: " + err_msg, (550, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    return result


# *** MAIN ***
# define global variables to use in the pipeline
left_lane = Line()
right_lane = Line()
error_im = 1
skipped_frames = 100
# load video
clip_name = "challenge_video"
clip1 = VideoFileClip(clip_name + ".mp4")#.subclip(0, 8)
# run video through the pipeline and save output
out_clip = clip1.fl_image(pipeline)
out_clip.write_videofile("output_videos/" + clip_name + "_output.mp4", audio=False)
