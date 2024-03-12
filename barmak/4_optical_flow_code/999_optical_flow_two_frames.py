'''Based on Daniel's code'''

import os
import glob
import cv2 as cv
import numpy as np
import imutils as im
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rpi = None

    ## Test images (~20 images?)
    # flist = glob.glob("./data/images/zigzag_selection_raw/*.jpg")
    # flist = glob.glob("./data/images/teste_optical_flow/*.jpg")
    # flist = glob.glob("./data/images/optical_flow/rpi2/day-201207/*.jpg")

    flist_a = glob.glob("/Users/barmak/Desktop/obs_videos/output/day-201203_ok/*-0_*.jpg")
    flist_b = glob.glob("/Users/barmak/Desktop/obs_videos/output/day-201203_ok/*-1_*.jpg")
    flist_a.sort()
    flist_b.sort()

    img_idx = 39
    f1 = flist_a[img_idx]
    f2 = flist_b[img_idx]
    print("frame 1: " + f1)
    print("frame 2: " + f2)

    frame1 = cv.imread(f1, cv.IMREAD_GRAYSCALE)
    frame2 = cv.imread(f2, cv.IMREAD_GRAYSCALE)

    frame1_beauty = im.beautify_frame(frame1, 'rpi2')
    frame2_beauty = im.beautify_frame(frame2, 'rpi2')

    ## Resize images (faster?)
    ## You can only resize after the beautify function
    ## it uses the original image size to do some undistortions
    ## Make frame1 and frame2 same size as frame_beauty
    ## which makes some cropping to the image

    frame1 = cv.resize(frame1, (frame1_beauty.shape[1], frame1_beauty.shape[0]))
    frame2 = cv.resize(frame2, (frame2_beauty.shape[1], frame2_beauty.shape[0]))

    ## Now resize all images
    sf = 0.6
    frame1 = cv.resize(frame1, (0, 0), fx=sf, fy=sf)
    frame2 = cv.resize(frame2, (0, 0), fx=sf, fy=sf)
    frame1_beauty = cv.resize(frame1_beauty, (0, 0), fx=sf, fy=sf)
    frame2_beauty = cv.resize(frame2_beauty, (0, 0), fx=sf, fy=sf)

    # --- Display
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(8, 19))
    ax0.imshow(frame1_beauty, cmap='gray')
    ax0.set_title("Frame 1 (%s)" % f1)
    ax0.set_axis_off()

    # difference = cv.subtract(frame1_beauty, frame2_beauty)
    # ret, mask = cv.threshold(difference, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # difference[np.where(mask != 255)] = 255
    #
    #ax1.imshow(difference)
    # gray = frame1_beauty
    # gray[np.where(mask != 255)] = 255
    # ax1.imshow(gray, cmap='gray' )
    # ax1.set_title("Difference between 2 frames")

    # ax1.imshow(frame2_beauty, cmap='gray')
    # ax1.set_title("Frame 2 (%s)" % f2)
    # # ax2.imshow(rgb)
    # # ax2.set_title("Optical flow between 2 frames")

    ########
    flow, of_mag, of_ang = im.compute_dense_optical_flow(frame1_beauty, frame2_beauty)

    ax1.imshow(of_mag)
    ax1.set_title("Magnitude of the optical flow vectors")

    ########
    (cx, cy), biggest_contour, core_area = im.find_biggest_active_area(of_mag)
    print("\tcore area: %.1f" % core_area)

    # mask_cnt = np.zeros(frame1_beauty.shape, np.uint8)
    # cv.drawContours(mask_cnt, [biggest_contour], 0, (36, 255, 12), 2)
    # ax3.imshow( mask_cnt, cmap='gray')
    ax2.imshow( frame1_beauty, cmap='gray' )

    big_cnt_x = biggest_contour[:, 0, 0]
    big_cnt_y = biggest_contour[:, 0, 1]
    ax2.plot(big_cnt_x, big_cnt_y, c='c')
    ax2.scatter(cx, cy, s=40, c='r')

    # ax2.imshow(mask_morph)
    # ax2.imshow(mask_of_cnt)

    ax2.set_title("High optical flow area")


    ########
    im.plot_of_quiver(ax3, flow, 25, 1)
    ax3.set_title("Optical flow vector field")

    ########
    # ax4.imshow(frame1_beauty, cmap='gray')

    cluster_cnt, cluster_area, img_t = im.find_cluster_contour(frame1_beauty, im.load_bg_img('./outputs/2_zigzag/background/rpi2/', scale_factor=sf))
    print("\tcluster area: %.1f" % cluster_area)
    print("\tratio: %.2f" % (core_area/cluster_area))

    # ax4.imshow(img_t, cmap='gray')
    ax4.imshow(frame1_beauty, cmap='gray')
    ax4.plot(cluster_cnt[:, 0, 0], cluster_cnt[:, 0, 1], c='r', lw=3.)
    ax4.plot(big_cnt_x, big_cnt_y, c='c')
    ax4.set_title("cluster (red) and core (cyan) perimeters")


    plt.tight_layout()
    plt.show()
    # plt.savefig("999_test_optical_flow.png", dpi=300)
