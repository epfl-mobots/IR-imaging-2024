import glob
import cv2 as cv

fps = 30
scale_percent = 45  # percent of original size

p_in = '/Users/barmak/Desktop/data_mess/imgs_obs_video_S1/imgs10_masks_overlay_yellow/'
p_out = '/Users/barmak/Desktop/data_mess/imgs_obs_video_S1/'
fn = '1c_obs_video_%dfps_yellow.mp4' % fps

files = sorted(glob.glob(p_in+'*.jpg'))

img = cv.imread(files[0])
height, width, layers = img.shape
size = (width, height)
print('Original Dimensions : ', img.shape)

## resize image
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
# print('Resized Dimensions : ', resized.shape)

# out = cv.VideoWriter(fn, cv.VideoWriter_fourcc('a','v','c','1'), fps, size)
out = cv.VideoWriter(p_out + fn, cv.VideoWriter_fourcc('a','v','c','1'), fps, dim)

for i, f in enumerate(files):
    print("%d (%d)" % (i, len(files)))
    img = cv.imread(f)
    # img_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    out.write(img)

out.release()
print("Video saved!")
