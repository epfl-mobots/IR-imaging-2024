"""
Script to create contour data from the images.
It stores the datetime, contour of the core (optical flow method) and
the external contour of the winter cluster.

This script create one single file with all data.
So, no need to run it multiple times (only if the raw data changed).
"""

import re
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import imutils as im # file created by Rafael
import datetime as dt

src_dir = "./data/images/optical_flow/rpi2/"
rpi = 'rpi2'
img_sf = 0.6 # resize by 60% (faster than 100%)

days = ['2012%02d' % d for d in range(1, 8)]

print(days)

dates = []
regex = r"\d{6}-\d{6}"

all_dates = []
all_imgs_a = []
all_imgs_b = []
image_size = []
core_center_px = []
core_center_mm = []
core_area_px = []
core_area_mm = []
core_contour = []
cluster_center_px = []
cluster_center_mm = []
cluster_area_px = []
cluster_area_mm = []
cluster_contour = []
area_ratio = []

print("teste")
# for d in days:
# for d in days[:1]:
for d in days[-1:]:
    dir = src_dir + 'day-' + d + '_ok/'
    print(dir)

    imgs_a = sorted(glob.glob("%s*-0_*.jpg" % dir))
    imgs_b = sorted(glob.glob("%s*-1_*.jpg" % dir))

    # Use as dates the date-time of the first frame (interval between frames = 4s)
    dates = [re.search(regex, f.split('/')[-1]).group(0) for f in imgs_a]
    dates = [dt.datetime.strptime(d, '%y%m%d-%H%M%S') for d in dates]

    ## Scan all images in that day
    # for i in range(len(imgs_a[:1])):
    for i in range(len(imgs_a)):
        t_ini = dt.datetime.now()

        f1 = imgs_a[i]
        f2 = imgs_b[i]

        all_dates.append( dates[i] )
        all_imgs_a.append( f1.split('/')[-1] )
        all_imgs_b.append( f2.split('/')[-1] )

        print("frame 1: " + f1.split('/')[-1])
        print("frame 2: " + f2.split('/')[-1])

        frame1 = cv.imread(f1, cv.IMREAD_GRAYSCALE)
        frame2 = cv.imread(f2, cv.IMREAD_GRAYSCALE)

        # Apply transformations to raw images
        frame1_beauty = im.beautify_frame(frame1, rpi)
        frame2_beauty = im.beautify_frame(frame2, rpi)
        print('done - beautyfication (%.1fs)' % (dt.datetime.now()-t_ini).total_seconds())

        frame1_beauty_small = cv.resize(frame1_beauty, (0, 0), fx=img_sf, fy=img_sf)
        frame2_beauty_small = cv.resize(frame2_beauty, (0, 0), fx=img_sf, fy=img_sf)

        image_size = frame1_beauty_small.shape

        # Calculate the optical flow to allow finding the cluster core
        flow, of_mag, of_ang = im.compute_dense_optical_flow(frame1_beauty_small, frame2_beauty_small)
        print('done - optical flow (%.1fs)' % (dt.datetime.now()-t_ini).total_seconds())

        # Find the cluster core
        (core_cx, core_cy), biggest_contour, cr_area = im.find_biggest_active_area(of_mag)

        core_contour_x = biggest_contour[:, 0, 0]
        core_contour_y = biggest_contour[:, 0, 1]

        # Convert core center from pixels to mm
        core_cx_mm = np.round(im.px_to_mm(frame1_beauty_small, core_cx, rpi, _axis='x'),2)
        core_cy_mm = np.round(im.px_to_mm(frame1_beauty_small, core_cy, rpi, _axis='y'),2)
        cr_area_mm = np.round(im.px_to_mm(frame1_beauty_small, cr_area, rpi, _axis='area'),2)

        core_center_px.append((core_cx, core_cy))
        core_center_mm.append((core_cx_mm, core_cy_mm))
        core_area_px.append(cr_area)
        core_area_mm.append(cr_area_mm)
        core_contour.append((core_contour_x, core_contour_y))
        print('done - core contour (%.1fs)' % (dt.datetime.now() - t_ini).total_seconds())

        # Find cluster external edge
        (cluster_cx, cluster_cy), cluster_cnt, cl_area, img_t = im.find_cluster_contour(frame1_beauty_small,
                    im.load_bg_img('./outputs/2_zigzag/background/%s/'%rpi, scale_factor=img_sf))

        cluster_contour_x = cluster_cnt[:, 0, 0]
        cluster_contour_y = cluster_cnt[:, 0, 1]

        # Convert core center from pixels to mm
        cluster_cx_mm = np.round(im.px_to_mm(frame1_beauty_small, cluster_cx, rpi, _axis='x'),2)
        cluster_cy_mm = np.round(im.px_to_mm(frame1_beauty_small, cluster_cy, rpi, _axis='y'),2)
        cl_area_mm = np.round(im.px_to_mm(frame1_beauty_small, cl_area, rpi, _axis='area'),2)

        cluster_center_px.append((cluster_cx, cluster_cy))
        cluster_center_mm.append((cluster_cx_mm, cluster_cy_mm))
        cluster_contour.append((cluster_contour_x, cluster_contour_y))
        cluster_area_px.append(cl_area)
        cluster_area_mm.append(cl_area_mm)
        print('done - cluster contour (%.1fs)' % (dt.datetime.now() - t_ini).total_seconds())

        area_ratio.append( np.round( cr_area/cl_area ,2))

        t_tot = dt.datetime.now() - t_ini
        print("%.1f s" % t_tot.total_seconds())
        print()

# Create the DataFrame
data = {'images_1': all_imgs_a,
        'images_2': all_imgs_b,
        # 'image_dim_resized': image_size, # All fields must be arrays with same size
        # 'image_resize_factor': img_sf,
        'cluster_contour': cluster_contour,
        'cluster_center_px': cluster_center_px,
        'cluster_center_mm': cluster_center_mm,
        'cluster_area_px': cluster_area_px,
        'cluster_area_mm': cluster_area_mm,
        'core_contour': core_contour,
        'core_center_px': core_center_px,
        'core_center_mm': core_center_mm,
        'core_area_px': core_area_px,
        'core_area_mm': core_area_mm,
        'area_ratio': area_ratio }

df_cluster = pd.DataFrame(data, index=all_dates)

# Save to a pickle file
# fn = "1b_cluster_contours_%s_df.pkl" % rpi
fn = "temp.pkl"
df_cluster.to_pickle(fn)

print("Pickl file created (%s)" % fn )

# import matplotlib.pyplot as plt
#
# fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(8, 19))
# ax0.imshow(frame1_beauty, cmap='gray')
# ax1.imshow(frame1_beauty_small, cmap='gray')
# ax2.imshow(of_mag)
#
# ax3.imshow(frame1_beauty_small, cmap='gray')
# ax3.plot(core_contour_x, core_contour_y, c='m')
# ax3.scatter(core_cx, core_cy, s=40, c='r')
#
# ax4.imshow(frame1_beauty_small, cmap='gray')
# ax4.plot(cluster_contour_x, cluster_contour_y, c='c')
# ax4.scatter(cluster_cx, cluster_cy, s=40, c='b')
# plt.show()
# plt.close()