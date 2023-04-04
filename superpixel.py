import numpy as np
import skimage
from skimage import color, io
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from segment import Process

# Default value of segments to partition via SLIC
SEGMENTS = 200

# Opens image in formats for skimage SLIC and homebrew SLIC
rgb = io.imread("Brandeis.jpg")
sk_img = img_as_float(rgb)
lab = color.rgb2lab(rgb)
Home_SLIC = Process(lab, SEGMENTS)
#Home_SLIC.slic_compute()

# default SciKitLearn SLIC with segments set to 200
sk_segments200 = slic(sk_img, n_segments=SEGMENTS)
plt.imshow(mark_boundaries(sk_img, sk_segments200))
plt.show()
