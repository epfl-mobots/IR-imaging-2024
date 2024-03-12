'''Based on Daniel's code'''

import os
import glob
import cv2 as cv
import numpy as np

_cam_matrix = np.array([
    [3.88774532e+03, 0.00000000e+00, 2.01016942e+03],
    [0.00000000e+00, 3.90460835e+03, 1.63295552e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
])

_distortion_coeffs = np.array([[
    -5.04380784e-01,
    3.32158862e-01,
    -4.75798455e-03,
    -3.43884435e-04,
    -1.47282285e-01,
]])

def undistort(img):
    """Remove distortions from the image using calibration data.

    From
    * https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    """
    # Shape of the image
    h, w = img.shape[:2]

    # Refine cam matrix and compute ROI
    newcameramtx, roi = cv.getOptimalNewCameraMatrix( _cam_matrix, _distortion_coeffs, (w, h), 1, (w, h))

    # Undistort
    dst = cv.undistort(
        img,
        _cam_matrix,
        _distortion_coeffs,
        None,
        newcameramtx,
    )

    # Crop the image
    x, y, w, h = roi
    # Ensure even numbers of pixels in rows and cols
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1
    dst = dst[y: y+h, x: x+w]

    return dst


def undistort2(img, rpi):
    """ Remove perspective distortion from Rpi4 (back)
    
        Ref: 
        https://pysource.com/2018/02/14/perspective-transformation-opencv-3-4-with-python-3-tutorial-13/
    """
    
    if rpi == 'rpi2':
        # Points we want to transform
        pa1 = (78, 95)
        pa2 = (3382, 95)
        pa3 = (132, 1950)
        pa4 = (3263, 1950)

        # cv.circle(img, pa1, 5, (0, 0, 255), -1)
        # cv.circle(img, pa2, 5, (0, 0, 255), -1)
        # cv.circle(img, pa3, 5, (0, 0, 255), -1)
        # cv.circle(img, pa4, 5, (0, 0, 255), -1)

        # Target locations for the points to move to
        pb1 = (0, 0)
        pb2 = (3200, 0)
        pb3 = (0, 1950)
        pb4 = (3200, 1950)

        # cv.circle(img, pb1, 15, (0, 200, 255), -1)
        # cv.circle(img, pb2, 15, (0, 200, 255), -1)
        # cv.circle(img, pb3, 15, (0, 200, 255), -1)
        # cv.circle(img, pb4, 15, (0, 200, 255), -1)

        # result = img
        pts_src = np.float32([list(pa1), list(pa2), list(pa3), list(pa4)])
        pts_tgt = np.float32([list(pb1), list(pb2), list(pb3), list(pb4)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        result = cv.warpPerspective(img, matrix, (3200, img.shape[0]))

    elif rpi == 'rpi4':
        # Points we want to transform
        pa1 = (0, 220)
        pa2 = (3200, 250)
        pa3 = (32, 1900)
        pa4 = (3130, 1850)

        # cv.circle(img, pa1, 5, (0, 0, 255), -1)
        # cv.circle(img, pa2, 5, (0, 0, 255), -1)
        # cv.circle(img, pa3, 5, (0, 0, 255), -1)
        # cv.circle(img, pa4, 5, (0, 0, 255), -1)

        # Target locations for the points to move to
        pb1 = (0, 220)
        pb2 = (3200, 220)
        pb3 = (0, 1900)
        pb4 = (3200, 1900)

        # cv.circle(img, pb1, 15, (0, 200, 255), -1)
        # cv.circle(img, pb2, 15, (0, 200, 255), -1)
        # cv.circle(img, pb3, 15, (0, 200, 255), -1)
        # cv.circle(img, pb4, 15, (0, 200, 255), -1)

        pts_src = np.float32([list(pa1), list(pa2), list(pa3), list(pa4)])
        pts_tgt = np.float32([list(pb1), list(pb2), list(pb3), list(pb4)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        result = cv.warpPerspective(img, matrix, (3200, img.shape[0]))

    return result
        
    

def unsharp_mask(
        image,
        kernel_size=(5, 5),
        sigma=1.0,
        amount=1.0,
        threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.

    https://en.wikipedia.org/wiki/Unsharp_masking
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm"""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        # OpenCV4 function copyTo
        np.copyTo(sharpened, image, where=low_contrast_mask)
    return sharpened


def beautify_frame(img, rpi):
    """Undistort, sharpen, hist-equalize and label image."""
    img = undistort(img)
    img = undistort2(img, rpi)
    img = unsharp_mask(img, amount=1.5)

    # Histogram equalization
    img = cv.equalizeHist(img)
    # img = cv.GaussianBlur(img, (3, 3), 0)

    return img


if __name__ ==  "__main__":
    rpi = None

    ## Test images (~20 images?)
    flist = glob.glob("./data/images/zigzag_selection_raw/*.jpg")
    outdir = "./data/images/zigzag_selection_undistorted/"

    ## Real images (many many images)
    # outdir = "./month-2021-02_undist/"
    # flist = glob.glob("./month-2021-02/*.jpg")
    # flist.sort()
       
    
    for i, f in enumerate(flist):
        # Detect which RPi the file belongs to
        if "rpi2" in f:
            rpi = 'rpi2'
        elif "rpi4" in f:
            rpi = 'rpi4'

        print("%d of %d" % (i, len(flist)))
        img = cv.imread(f, cv.IMREAD_GRAYSCALE)
        img = beautify_frame(img, rpi)

        cv.imwrite(outdir + "undist" + f.split("/")[-1][3:], img)
