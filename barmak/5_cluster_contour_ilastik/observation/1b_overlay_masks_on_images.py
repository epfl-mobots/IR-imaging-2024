import glob
import imutils
import cv2 as cv
import numpy as np

# Resize parameter
percentage = 50  # specify your percentage here

color = 'yellow'  # 'red' or 'yellow'
p_imgs  = '/Users/barmak/Desktop/data_mess/imgs_obs_video_S1/imgs10/'
p_fills = '/Users/barmak/Desktop/data_mess/imgs_obs_video_S1/imgs10_masks/'
p_out   = '/Users/barmak/Desktop/data_mess/imgs_obs_video_S1/imgs10_masks_overlay_%s/' % color

imgs = sorted(glob.glob(p_imgs + '*.jpg'))
masks = sorted(glob.glob(p_fills + '*.png'))

c = None
if color == 'yellow':
    c = [0, 210, 255]
elif color == 'red':
    c = [0, 0, 255]

for i in range(len(imgs)):
    print(i, len(imgs))

    img = cv.imread(imgs[i])
    mask = cv.imread(masks[i])

    # Change color of mask to red
    mask[(mask == 255).all(-1)] = c

    dst = cv.addWeighted(img, 1.0, mask, 0.2, 0)

    resized_img = cv.resize(dst, (int(dst.shape[1] * percentage / 100), int(dst.shape[0] * percentage / 100)))

    cv.imwrite(p_out + masks[i].split('/')[-1][:38] + '_fill.jpg', resized_img)

print('DONE')
