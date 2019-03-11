import matplotlib.pyplot as plt
import cv2


# Plotting functions
def plot_calibration(original, corrected):
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,2.5))
    f.tight_layout()
    for ax in (ax1, ax2):
        ax.axis('off')
    ax1.set_title('Original image', fontsize=18)
    ax1.imshow(original)
    ax2.set_title('Undistorted image', fontsize=18)
    ax2.imshow(corrected)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    plt.show()
    f.savefig("output_images/dist_correct.jpg")
    return


def plot_thresholds(undist, gradx, grady, mag_binary, dir_binary, hls_binary, x_mag, combined, dir_hsl):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12, 8))
    f.tight_layout()
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        ax.axis('off')
    ax1.set_title('Original Image', fontsize=18)
    ax1.imshow(undist)
    ax2.set_title('mag_bin', fontsize=18)
    ax2.imshow(gradx, cmap='gray')
    ax3.set_title('dir_bin', fontsize=18)
    ax3.imshow(grady, cmap='gray')
    ax4.set_title('hls_bin', fontsize=18)
    ax4.imshow(mag_binary, cmap='gray')
    ax5.set_title('white_bin', fontsize=18)
    ax5.imshow(dir_binary, cmap='gray')
    ax6.set_title('yellow_bin', fontsize=18)
    ax6.imshow(hls_binary, cmap='gray')
    ax7.set_title('mag & dir & hls', fontsize=18)
    ax7.imshow(x_mag, cmap='gray')
    ax8.set_title('Combined', fontsize=20, fontweight='bold')
    ax8.imshow(combined, cmap='gray')
    ax9.set_title('white or yellow', fontsize=18)
    ax9.imshow(dir_hsl, cmap='gray')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    plt.show()
    f.savefig("output_images/thresholds.jpg")
    return


def plot_warping(original, warp, src):
    # Uncomment this code to visualize the region of interest
    # for i in range(len(src)):
    #     cv2.line(original, (src[i][0], src[i][1]), (src[(i+1)%4][0], src[(i+1)%4][1]), 1, 2)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))
    f.tight_layout()
    # for ax in (ax1, ax2):
    #     ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.08)
    ax1.set_title('Undistorted Image', fontsize=18)
    ax1.imshow(original)
    ax2.set_title('Warped Image', fontsize=18)
    ax2.imshow(warp)
    plt.show()
    f.savefig("output_images/warp.jpg")
    return


def plot_img(image):
    plt.imshow(image)
    plt.axis('off')
    # plt.tight_layout(pad=0.01, rect=(0,-0.1,1,1.1))
    plt.subplots_adjust(left=0.1, right=0.95, top=1, bottom=0)
    # plt.margins(y=0)
    # plt.savefig("output_images/fit.jpg")
    plt.show()
