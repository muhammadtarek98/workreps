import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2 as cv


# Let's load a simple images with 3 black squares
image = cv.imread(r'D:\graval detection project\resyris\resyris\test_data_realworld\gt\000.png')
cv.waitKey(0)

# Grayscale

#gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
# Find Canny edges
edged = cv.Canny(image, 30, 200)
cv.waitKey(0)

# Finding Contours
# Use a copy of the images e.g. edged.copy()
# since findContours alters the images
contours, hierarchy = cv.findContours(edged,
	cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.imshow('Canny Edges After Contouring', edged)
cv.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv.drawContours(image, contours, -1, (0, 255, 0), 3)

cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()

"""
# generate some lowpass-filtered noise as a test images
gen = np.random.RandomState(0)
img = gen.poisson(2, size=(512, 512))
img = ndimage.gaussian_filter(img.astype(np.double), (30, 30))
img -= img.min()
img /= img.max()

# use a boolean condition to find where pixel values are > 0.75
blobs = img > 0.75

# label connected regions that satisfy this condition
labels, nlabels = ndimage.label(blobs)

# find their centres of mass. in this case I'm weighting by the pixel values in
# `img`, but you could also pass the boolean values in `blobs` to compute the
# unweighted centroids.
r, c = np.vstack(ndimage.center_of_mass(img, labels, np.arange(nlabels) + 1)).T

# find their distances from the top-left corner
d = np.sqrt(r*r + c*c)

# plot
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
ax[0].imshow(img)
ax[1].imshow(np.ma.masked_array(labels, ~blobs), cmap=plt.cm.rainbow)
for ri, ci, di in zip(r, c, d):
    ax[1].annotate('', xy=(0, 0), xytext=(ci, ri),
                   arrowprops={'arrowstyle':'<-', 'shrinkA':0})
    ax[1].annotate('d=%.1f' % di, xy=(ci, ri),  xytext=(0, -5),
                   textcoords='offset points', ha='center', va='top',
                   fontsize='x-large')
for aa in ax.flat:
    aa.set_axis_off()
fig.tight_layout()
plt.show()"""