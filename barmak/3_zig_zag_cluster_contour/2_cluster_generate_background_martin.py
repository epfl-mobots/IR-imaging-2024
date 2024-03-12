import cv2, csv, os, imutils, errno, random
import glob
import numpy as np
from pathlib import Path

def generate_background(files, background_folder, rpi):
    mog = cv2.createBackgroundSubtractorMOG2()
    mog.setDetectShadows(False)
    mog.setHistory(50)
    mog.setVarThreshold(16)
    learning_rate = -1

    # Generate one background for each group of 50 images
    for count, x in enumerate(files):
        # read file into open cv and apply to algorithm to generate background model
        img = cv2.imread(str(x), 1)
        img_output = mog.apply(img, learning_rate)
        img_bgmodel = mog.getBackgroundImage()
        bg = background_folder + "background_%s_%04d.jpg" % (rpi, count)

        # Create one background every 50 images
        if count % 50 == 0 and count > 5:
            cv2.imwrite(bg, img_bgmodel)
            print(bg)

    files2 = glob.glob(background_folder + '*.jpg')
    # files2 = searching_all_files(background_folder)
    print(len(files2))

    mog = cv2.createBackgroundSubtractorMOG2()
    mog.setDetectShadows(False)
    mog.setHistory(len(files2))
    mog.setVarThreshold(16)
    learning_rate = -1

    for y in files2:
        # read file into opencv and apply to algorithm to generate background model
        img_new = cv2.imread(str(y), 1)
        img_output_new = mog.apply(img_new, learning_rate);
        img_bgmodel_new = mog.getBackgroundImage();

    cv2.imwrite(f'{str(background_folder)}final_background_{str(rpi)}.jpg', img_bgmodel_new)

    return f'{str(background_folder)}final_background.jpg'


if __name__ == "__main__":
    rpi = 'rpi4'

    # Get all files in all sudirectories
    files = glob.glob("./data/images/zigzag_undistorted/%s/**/*.jpg" % rpi)

    # Where to store the computed backgrounds
    background_folder = "./outputs/2_zigzag/background/" + rpi + "/"

    bg_name = generate_background(files, background_folder, rpi)

    # bg = cv2.imread(bg_name)

