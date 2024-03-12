'''
1. Get the masks from ilastik
2. Perform morphological operations to remove noise and fill holes
3. Find the biggest contour
4. Get the centroid of the biggest contour
5. Save the mask with the area of the biggest contour and the centroid on the filename
'''

import glob
import cv2 as cv
import numpy as np


p_in = '/Users/fabianprofeta/Desktop/PDS-IR-imaging/IR-imaging-2024/barmak/5_cluster_contour_ilastik/imgs10/ilastik/'
# where to save masks
p_out = '/Users/fabianprofeta/Desktop/PDS-IR-imaging/IR-imaging-2024/barmak/5_cluster_contour_ilastik/imgs10_masks/'

files = sorted(glob.glob(p_in + '*.png'))

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
# kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))

for i, f in enumerate(files):
    print(i, len(files))

    # Read and invert image
    frame_bw = cv.bitwise_not(cv.imread(files[i]))

    ## Opening is just another name of erosion followed by dilation. It is useful in removing noise
    frame_no_bg = cv.morphologyEx(frame_bw, cv.MORPH_OPEN, kernel, iterations=5)

    frame_no_bg = cv.morphologyEx(frame_no_bg, cv.MORPH_CLOSE, kernel, iterations=5)

    # print('showing image')
    # cv.imshow('frame_bw', frame_bw)
    # cv.imshow('frame_no_bg', frame_no_bg)


    # # Threshold it so it becomes binary
    # ret, thresh = cv.threshold(frame_no_bg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(frame_no_bg[:, :, 0], 4, cv.CV_32S)
    # # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(frame_no_bg, 4, cv.CV_32S)
    # # # Different end domains give different colors
    # # output = np.zeros((frame_no_bg.shape[0], frame_no_bg.shape[1], 3), np.uint8)
    # # for i in range(1, num_labels):
    # #     mask = labels == i
    # #     output[:, :, 0][mask] = np.random.randint(0, 255)
    # #     output[:, :, 1][mask] = np.random.randint(0, 255)
    # #     output[:, :, 2][mask] = np.random.randint(0, 255)
    # # cv.imshow('oginal', output)
    # # cv.waitKey()
    # # cv.destroyAllWindows()


    contours, hierarchy = cv.findContours(frame_no_bg[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours, hierarchy= cv.findContours(frame_no_bg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_areas = np.array([cv.contourArea(cnt) for cnt in contours])

    MIN_BEE_AREA = 50  # areas smaller than MIN_BEE_AREA are excluded
    idx_a = contour_areas > MIN_BEE_AREA
    contours = np.array(contours)
    contours = contours[idx_a]
    contour_areas = contour_areas[idx_a]

    ## Get the centroid of the biggest contour
    idx_big_area = np.argmax(contour_areas)
    # Calculate moments for each contour
    M = cv.moments(contours[idx_big_area])
    # Calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print(cX, cY)

    # cv.circle(frame_no_bg, (cX, cY), 15, (128, 128, 255), -1)
    # cv.imshow('frame_no_bg', frame_no_bg)

    ###
    # cv.drawContours(frame_no_bg, contours, -1, (0,255,0), 2)
    #
    # # Create copy of image
    # height, width = frame_no_bg.shape[:2]
    # img_big_contours = np.zeros((height, width, 1), np.uint8)
    #
    # idx_big_area = np.argmax(contour_areas)
    # cv.fillPoly(img_big_contours, contours[idx_big_area], 128)
    #
    # # cv.fillPoly(img_big_contours, contours, 128)
    # cv.imshow('img_big_contours', img_big_contours)

    # area, in percent, occupied by white pixels (ie occupied by bees)
    img_tot_px = frame_no_bg[:, :, 0].size
    # number of pixels that are white
    idx_white = np.where(frame_no_bg[:, :, 0] == 255)
    img_white_px = len(idx_white[0])

    img_white_area = round(img_white_px / img_tot_px * 100, 1)

    cv.imwrite(p_out + "%s_a%02.1f_x%04dy%04d.png" % (files[i].split('/')[-1][:32], img_white_area, cX, cY), frame_no_bg)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

print("DONE!")
