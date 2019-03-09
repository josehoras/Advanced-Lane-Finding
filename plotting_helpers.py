import matplotlib.pyplot as plt


# Plotting functions
def plot_thresholds(undist, gradx, grady, mag_binary, dir_binary, hls_binary, combined):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12, 8))
    f.tight_layout()
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        ax.axis('off')
    ax1.set_title('Original Image', fontsize=18)
    ax1.imshow(undist)
    ax2.set_title('Grad. X', fontsize=18)
    ax2.imshow(gradx, cmap='gray')
    ax3.set_title('Grad. Y', fontsize=18)
    ax3.imshow(grady, cmap='gray')
    ax4.set_title('Magnitud', fontsize=18)
    ax4.imshow(mag_binary, cmap='gray')
    ax5.set_title('Direction', fontsize=18)
    ax5.imshow(dir_binary, cmap='gray')
    ax6.set_title('Saturation', fontsize=18)
    ax6.imshow(hls_binary, cmap='gray')
    ax8.set_title('Combined', fontsize=18)
    ax8.imshow(combined, cmap='gray')
    plt.subplots_adjust(left=0.01, right=1, top=0.9, bottom=0.)
    plt.show()
    return


def plot_warping(original, warp):
    # Uncomment this code to visualize the region of interest
    # for i in range(len(src)):
    #     cv2.line(original, (src[i][0], src[i][1]), (src[(i+1)%4][0], src[(i+1)%4][1]), 1, 2)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.imshow(original)
    ax2.imshow(warp)
    plt.show()
    return


def plot_img(image):
    plt.imshow(image)
    plt.show()
