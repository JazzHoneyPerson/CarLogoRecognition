# generateFeature.py
# Yu Fu

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage import io, transform, morphology
from skimage.feature import hog
from skimage import data, color, exposure

training_sample_paths = []
y = []
brand = ["audi", "benz", "bmw", "chevrolet", "honda", "lexus", "toyota", "volkswagon"]

for brand_num in range(len(brand)):
#for brand_num in range(1):
    for root, dirs, files in os.walk("./TrainingSet/" + brand[brand_num]):
        for filename in files:
            if filename.endswith(".jpg"):
                training_sample_paths.append(os.path.join(root, filename))
                y.append(brand_num + 1)

# hog test
# image = io.imread(training_sample_paths[100])
# image = transform.resize(image, (400, 400))
# fd, hog_image = hog(image, orientations=8,
#                     pixels_per_cell=(20, 20), cells_per_block=(1, 1),
#                     visualise=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_adjustable('box-forced')
#
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# ax1.set_adjustable('box-forced')
# plt.show()
# End of hog test

# Feature Extraction for NN
y = np.repeat(y, 30)
features = np.empty([len(y), 3200])

row_count = 0
for image_path in training_sample_paths:
    image = io.imread(image_path)
    image = transform.resize(image, (400, 400))
    for i in range(10):
        image_new = image * np.random.normal(1.0, 0.2) + np.random.normal(0, 0.2)
        image_new = np.maximum(np.minimum(image, 1.0), 0.0)

        image_new_dilation = morphology.dilation(image, morphology.square(2))
        image_new_erosion = morphology.erosion(image, morphology.square(2))

        features[row_count, :] = hog(image, orientations=8,
                                     pixels_per_cell=(20, 20), cells_per_block=(1, 1))
        features[row_count+1, :] = hog(image_new_dilation, orientations=8,
                                       pixels_per_cell=(20, 20), cells_per_block=(1, 1))
        features[row_count+2, :] = hog(image_new_erosion, orientations=8,
                                       pixels_per_cell=(20, 20), cells_per_block=(1, 1))
        row_count += 3
    print("Extracting feature from {} out of {} images".format(row_count // 30, len(training_sample_paths)))

np.savez("data", features, y)

# Feature Extraction Small

# features = np.empty([len(y), 3200])
#
# row_count = 0
# for image_path in training_sample_paths:
#     image = io.imread(image_path)
#     image = transform.resize(image, (400, 400))
#
#     features[row_count, :] = hog(image, orientations=8,
#                                  pixels_per_cell=(20, 20), cells_per_block=(1, 1))
#
#     row_count += 1
#     print("Extracting feature from {} out of {} images".format(row_count, len(training_sample_paths)))
#
# np.savez("data_small", features, y)

