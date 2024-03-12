import cv2 as cv
import numpy as np
import glob


def draw_h_lines(_img, y):
    line_thick = 3
    img_with_lines = None
    _h, _w = _img.shape[:2]

    # if y < 0 or y > _h:
    #     y = _h
    #     print('ERR: y value is outside the image.')

    if isinstance(y, list):
        for a in y:
            start_point = (0, a)
            end_point = (_w, a)
            img_with_lines = cv.line(_img, start_point, end_point, (0, 255, 0), line_thick)
    else:
        start_point = (0, y)
        end_point = (_w, y)
        img_with_lines = cv.line(_img, start_point, end_point, (0, 255, 0), line_thick)

    return img_with_lines


def draw_v_lines(_img, x):
    line_thick = 3
    img_with_lines = None
    _h, _w = _img.shape[:2]

    # if y < 0 or y > _h:
    #     y = _h
    #     print('ERR: y value is outside the image.')

    if isinstance(x, list):
        for a in x:
            start_point = (a, 0)
            end_point = (a, _h)
            img_with_lines = cv.line(_img, start_point, end_point, (0, 255, 0), line_thick)
    else:
        start_point = (x, 0)
        end_point = (x, _h)
        img_with_lines = cv.line(_img, start_point, end_point, (0, 255, 0), line_thick)

    return img_with_lines


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def undistort(img, pos):
    if pos == 'back':
        # Rectify image
        # Points we want to transform -> pa
        pa_tl = (0, 160 ); pa_tr = (3200, 190)
        pa_ll = (0, 1950); pa_lr = (3150, 1890)

        # Destination points -> pb
        pb_tl = (0, 160 ); pb_tr = (3200, 160)
        pb_ll = (0, 1950); pb_lr = (3200, 1950)

        pts_src = np.float32([list(pa_ll), list(pa_tl), list(pa_tr), list(pa_lr)])
        pts_tgt = np.float32([list(pb_ll), list(pb_tl), list(pb_tr), list(pb_lr)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        img_trans = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

        return crop_img(img_trans, [0, 170, 3200, 1950])  # [x, y, w, h]

    elif pos == 'front':
        # Rectify image
        # Points we want to transform -> pa
        pa_tl = (45 , 210 ); pa_tr = (3400, 250)
        pa_ll = (105, 1995); pa_lr = (3300, 2020)

        # Destination points -> pb
        pb_tl = (45, 210 ); pb_tr = (3400, 210)
        pb_ll = (45, 2020); pb_lr = (3400, 2020)

        pts_src = np.float32([list(pa_ll), list(pa_tl), list(pa_tr), list(pa_lr)])
        pts_tgt = np.float32([list(pb_ll), list(pb_tl), list(pb_tr), list(pb_lr)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        img_trans = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

        return crop_img(img_trans, [55, 225, 3375, 2010])  # [x, y, w, h]


def crop_img(img: np.ndarray, cropping) -> np.ndarray:
    """Crop image with given `cropping=[x, y, w, h]`."""
    x, y, w, h = cropping
    return img[y:h, x:w]


pos = 'front'  # 'back' or 'front'
p_imgs = '/Users/barmak/Desktop/sample_median/%s/' % pos
f_imgs = sorted(glob.glob(p_imgs + '*.jpg'))

pcb_ratio = 420 / 220

# The function cv2.imread() is used to read an image.
img = cv.imread(f_imgs[400])

img_trans_crop = undistort(img, pos)

# cv.imshow('image', img)
cv.imshow('image', img_trans_crop)

cv.waitKey(0)
cv.destroyAllWindows()

# print(i, len(f_img), fn)
# cv.imwrite(p_out + "u_" + pos + fn[3:], img_reg)
