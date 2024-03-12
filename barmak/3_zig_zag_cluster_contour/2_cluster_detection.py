'''
Script to analyse the images of the zig-zag experiment and create a pickle and a CSV file with the cluster's
centroid information.

Pipeline:
1. Undistort and pre-process raw images using the script:
    1_undistort_img.py
2. Create the background image (one for each side of the frame) using:
    2_cluster_generate_background_martin.py
3. Generate files with centroid data of the images (this file):
    2_cluster_detection.py
4. Plot data:
    2_zigzag_centroid.py
'''

# TODO Export small images with cluster line and some data?

import cv2, csv, os, imutils, errno
import glob
import numpy as np
import pandas as pd
from datetime import datetime

rpi = 'rpi4'

## Test images
# f_temp = glob.glob('./data/images/zigzag_selection_undistorted/*.jpg')
## Real deal
f_temp = glob.glob("./data/images/zigzag_undistorted/%s/**/*.jpg" % rpi)

f_temp.sort()

## Background images
bkg_rpi2 = "./outputs/2_zigzag/background/rpi2/final_background_rpi2.jpg"
bkg_rpi4 = "./outputs/2_zigzag/background/rpi4/final_background_rpi4.jpg"

output_folder = "./outputs/2_zigzag/"

scale_percent = 30 # percent of original size
save_csv      = True
save_pandas   = True

# files = [f_temp[1]]
files = f_temp

def create_folder(name):
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def px_to_mm(_img, _val, _rpi, _axis):
    ''' This function is tuned to undistorted images with the specific script (???.py) '''
    px_min = None
    px_max = None
    py_min = None
    py_max = None

    w = _img.shape[1]
    h = _img.shape[0]

    if _rpi == 'rpi2':
        # Pixel number that corresponds to board x=0 mm
        px_min = int(w * 0.0)
        # Pixel number that corresponds to board x=410 mm
        px_max = int(w * 1)
        # Pixel number that corresponds to board y=0 mm
        py_min = int(h * 0.79) # 79% of the height
        # Pixel number that corresponds to board y=180 mm
        py_max = int(h * 0.18) # 22% of the height

    elif _rpi == 'rpi4':
        px_min = int(w * 1)
        px_max = 0 #int(w * 0.0)
        py_min = int(h * 0.77)
        py_max = int(h * 0.18)

    if _axis == 'x':
        x_mm = np.round((_val - px_min) / (px_max - px_min) * (410 - 0) + 0, 1)
        x_norm = np.round((_val - px_min) / (px_max - px_min) * (1 - 0) + 0, 3) # Normalized
        return (x_mm, x_norm)

    elif _axis == 'y':
        # y = (px - 0) / (410 - 0) * (px_max-px_min) + px_min
        y_mm = np.round((_val - py_min) / (py_max - py_min) * (180 - 0) + 0, 1)
        y_norm = np.round((_val - py_min) / (py_max - py_min) * (1 - 0) + 0, 3)
        return (y_mm, y_norm)

    elif _axis == 'area':
        dx = np.abs(px_max - px_min)
        dy = np.abs(py_min - py_max)
        tot_px = dx * dy
        tot_area = 410 * 180
        area_per_px = tot_area / tot_px # [mm^2/px]
        area = _val * area_per_px # [mm^2]
        return area

def draw_grid(_img, rpi):
    x = None
    y = None

    w = _img.shape[1]
    h = _img.shape[0]

    if rpi == 'rpi2':
        x = [int(w * 0), int(w * 0.999)]
        y = [int(h * 0.18), int(h * 0.79)]
    elif rpi == 'rpi4':
        x = [int(w * 0), int(w * 0.999)]
        y = [int(h * 0.18), int(h * 0.77)]

    for i in range(len(x)):
        # Vertical lines
        cv2.line(_img, (x[i], 0), (x[i], h), (255, 255, 0), 1)
        # Horizontal lines
        cv2.line(_img, (0, y[i]), (w, y[i]), (0, 255, 255), 1)


centerX = [] #Center normalized [0, 1]
centerY = []
centerXpx = [] # Center in pixels
centerYpx = []
centerXmm = [] # Center in mm
centerYmm = []
left_edge_norm = []
right_edge_norm = []
left_edge_px = []
right_edge_px = []
left_edge_mm = []
right_edge_mm = []
area_px = []
area_mm = [] # area in mm2 << max area = 410mm x 180mm >>
area_cm = [] # area in cm2 <<max area = [(410mm x 180mm) * 0.01] cm^2>>

i_name = []
i_date = []
i_time = []
i_size = []
i_dts = []

n_files = len(files)

for i, f in enumerate(files):
    ## Load pre-calculated background
    ## (Used 2_cluster_generate_background_martin.py)

    # background_folder = output_folder + "background/%s/" % rpi
    # bg = cv2.imread(background_folder + 'final_background_%s.jpg' % rpi)
    if "rpi2" in f:
        rpi = "rpi2"
        bg = cv2.imread(bkg_rpi2)
    elif "rpi4" in f:
        rpi = "rpi4"
        bg = cv2.imread(bkg_rpi4)

    # bg = cv2.imread( bkg_rpi2 if "rpi2" in f else bkg_rpi4 )

    width = int(bg.shape[1] * scale_percent / 100)
    height = int(bg.shape[0] * scale_percent / 100)
    dim = (width, height)
    bg = cv2.resize(bg, dim)
    bg = cv2.GaussianBlur(bg, (55, 55), 0)
    bg = cv2.equalizeHist(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))

    name = (str(f).split('/')[-1])
    i_name.append(name)
    img = cv2.imread(str(f), 1)
    d = name.split('_%s_' % rpi)[1].split('-')[0]
    t = name.split('_%s_' % rpi)[1].split('-')[1]
    i_date.append(d)
    i_time.append(t)
    i_dts.append(datetime.strptime(d+t, '%y%m%d%H%M%S'))
    print("%d (%d): %s" % (i, n_files, name))

    #resize image
    i_width = int(img.shape[1] * scale_percent / 100)
    i_height = int(img.shape[0] * scale_percent / 100)
    i_dim = (i_width, i_height)
    img2 = cv2.resize(img, i_dim)

    img = cv2.GaussianBlur(img2, (55, 55), 0)
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    thresh = np.zeros_like(img)

    threshold1 = 180
    threshold2 = 70

    img_thresh1 = np.full_like(img, 255)
    img_thresh2 = np.full_like(img, 255)
    img_thresh3 = np.full_like(img, 255)

    img_thresh1[(bg > threshold1)] = img[(bg > threshold1)]
    img_thresh2[(bg > threshold2) & (bg < threshold1)] = img[(bg > threshold2) & (bg < threshold1)]
    img_thresh3[(bg < threshold2)] = img[(bg < threshold2)]

    ret, thresh1 = cv2.threshold(img_thresh1, 175, 255, 0)
    ret, thresh2 = cv2.threshold(img_thresh2, 140, 255, 0)
    ret, thresh3 = cv2.threshold(img_thresh3, 80, 255, 0)

    thresh[(bg > threshold1)] = thresh1[(bg > threshold1)]
    thresh[(bg > threshold2) & (bg < threshold1)] = thresh2[(bg > threshold2) & (bg < threshold1)]
    thresh[(bg < threshold2)] = thresh3[(bg < threshold2)]

    # ret, thresh = cv2.threshold(img, 115, 255, 0)
    #thresh = cv2.GaussianBlur(thresh,(35,35),0)

    # Morphological opening and closing to improve mask
    mask_morph = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )

    mask_morph = cv2.morphologyEx(
            mask_morph,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (155, 155))
    )

    mask_morph[0, :] = 255
    mask_morph[:, 0] = 255
    mask_morph[-1, :] = 255
    mask_morph[:, -1] = 255

    ## Find contours
    contours, hierarchy = cv2.findContours(mask_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((contours, hierarchy))

    # img = cv2.drawContours(img2, contours, -1, (0,255,0), 3)
    # cv2.imshow("C", img2)
    # cv2.waitKey(0)

    idx_c_big = None
    area_big = 0

    if len(cnts) > 0:
        # find the second biggest contour. Exclude the first one which is the whole frame
        for i, c in enumerate(cnts[1:]):
            if cv2.contourArea(c) > area_big:
                area_big = cv2.contourArea(c)
                idx_c_big = i+1 #+1 because we exclude the first one

        ## Just plot the biggest contour
        c = cnts[idx_c_big]

        # compute the centroid of the contour
        M = cv2.moments(c)

        if M["m00"] != 0.0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print(f'{i}: X = {cX}, Y = {cY}')

        cv2.drawContours(img2, [c], -1, (0, 255, 0), 2)
        cv2.circle(img2, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img2, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img2, "x = %d" % cX, (cX - 20, cY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        a = cv2.contourArea(c)
        a_mm = np.round(px_to_mm(img2, a, rpi, _axis='area'),2)
        a_cm = np.round(a_mm * 0.01, 2)

        area_px.append(a)
        area_mm.append( a_mm )
        area_cm.append( a_cm)

        i_size.append((a / img.size))
        cv2.putText(img2, "%.1f cm^2"%(a_cm), (cX - 20, cY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        ## Bounding box
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        cv2.rectangle(img2, (x_box, y_box), (x_box + w_box, y_box + h_box), (127, 127, 0), 2)

        if rpi == "rpi2":
            x_min = x_box
            x_max = x_box + w_box
        elif rpi == "rpi4":
            x_min = x_box + w_box
            x_max = x_box

        left_edge_px.append( x_min )
        left_edge_mm.append(px_to_mm(img2, x_min, rpi, _axis='x')[0])
        left_edge_norm.append(px_to_mm(img2, x_min, rpi, _axis='x')[1])

        right_edge_px.append( x_max )
        right_edge_mm.append(px_to_mm(img2, x_max, rpi, _axis='x')[0])
        right_edge_norm.append(px_to_mm(img2, x_max, rpi, _axis='x')[1])

        # show the image
        # cv2.imwrite(f'{output_folder}/{i_name}', img)
        cenX = px_to_mm(img2, cX, rpi, _axis='x')
        cenY = px_to_mm(img2, cY, rpi, _axis='y')

        centerX.append(cenX[1])
        centerXpx.append(cX)
        centerXmm.append(cenX[0])

        centerY.append(cenY[1])
        centerYpx.append(cY)
        centerYmm.append(cenY[0])

        # print(f'{i}: X = {cX}px, Y = {cY}px')
        # print(f'{i}: X = {cenX[0]}mm, Y = {cenY[0]}mm')

    # print(len(cnts))
    # draw_grid(img2, rpi)
    # cv2.imshow("D", img2)
    # cv2.waitKey(0)

# Save the pandas DataFrame
if save_pandas:
    data = {'names': i_name,
            'center_x': centerX,
            'center_y': centerY,
            'center_x_mm': centerXmm,
            'center_y_mm': centerYmm,
            'left_edge': left_edge_norm,
            'left_edge_mm': left_edge_mm,
            'right_edge': right_edge_norm,
            'right_edge_mm': right_edge_mm,
            'area_cm': area_cm
            }

    df_centroid = pd.DataFrame(data, index=i_dts)
    df_centroid.to_pickle(output_folder + "2_cluster_centroid_df_%s.pkl" % rpi)

# Save a CSV file
if save_csv:
    rows = zip(range(1,len(i_name)), i_name, i_dts,
               centerX, centerY, centerXmm, centerYmm,
               left_edge_norm, left_edge_mm,
               right_edge_norm, right_edge_mm,
               area_cm)

    with open(output_folder + '2_cluster_centroid_%s.csv' % rpi, "w", newline='') as f:
        writer = csv.writer(f)
        # writer.writerow("file,datetime,x_px,y_px,x_mm,y_mm,x,y")
        for row in rows:
            writer.writerow(row)

cv2.destroyAllWindows()