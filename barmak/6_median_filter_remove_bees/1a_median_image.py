import glob
import cv2 as cv
import numpy as np
import dask.array as da
from dask import delayed
from skimage.io import imread
import timeit
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Define a function to read an image
@delayed
def read_image(path):
    return imread(path)

pos = 'bot'

p_top = '/Users/barmak/Desktop/sample_h4r4_200729--201009/imgs/imgs_h4r4/'
p_bot = '/Users/barmak/Desktop/sample_h4r2_200729--201009/imgs/imgs_h4r2/'

p = p_top if pos == 'top' else p_bot
files = sorted(glob.glob(p + '*.jpg'))

## Find datetimes of images
rpi = 4 if pos == 'top' else 2

# regex pattern to match the time in the filename
pattern = re.compile(r'vid_hive4_rpi%d_(\d{6})-(\d{6})-utc.jpg' % rpi)

# list to store the datetime objects
datetimes = []

# iterate over all files in the directory
for filename in files:
    filename = filename.split('/')[-1]
    match = pattern.match(filename)
    if match:
        # extract the date and time from the filename
        date_str = match.group(1)
        time_str = match.group(2)
        # create a datetime object
        dt = datetime.strptime(date_str + time_str, '%y%m%d%H%M%S')
        # add the datetime object to the list
        datetimes.append(dt)



start_time = timeit.default_timer()

print('1')
i = 4624
n = 300
# List of image paths
image_paths = files[i:i+n]

print('2')
# Use Dask's delayed function to create a lazy evaluation of the function
delayed_images = [da.from_delayed(read_image(path), shape=(cv.imread(image_paths[0]).shape), dtype=np.uint8) for path in image_paths]

print('3')
# Stack all the images into a Dask array
images = da.stack(delayed_images, axis=0)
first_image = np.array(images[0].compute(), dtype=np.uint8)

# # Read the images into a list
# images = [cv.imread(img) for img in image_paths]
#
# # Stack all images along a new dimension
# stacked_images = np.stack(images)

print('4')
# Calculate the median along the new dimension
median_image = np.median(images, axis=0)

print('5')
# convert to numpy array so it can be displayed by CV2
median_image = median_image.compute()
median_image = np.array(median_image, dtype=np.uint8)

end_time = timeit.default_timer()

print(f"The code took {round(end_time - start_time, 1)} seconds to execute.")
# # The median image is floating point, convert it back to 8-bit
# median_image = median_image.astype(np.uint8)
#
# # Save the median image
# print('showing median image...')
# cv.imshow('median_image', median_image)
# cv.waitKey(0)
# cv.destroyAllWindows()


# Create a subplot
plt.figure(figsize=(7, 10))

plt.subplot(2, 1, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(first_image)
plt.title('First Image')
plt.axis('off')

plt.subplot(2, 1, 2)  # 1 row, 2 columns, 2nd subplot
plt.imshow(median_image)
plt.title('Median image of %d images' % n)
plt.axis('off')

# plt.show()
plt.savefig('1a_median_image_i%04d_n%d.jpg' % (i, n))

#
# # cv.imwrite('median_image.jpg', median_image)

print('DONE!')
