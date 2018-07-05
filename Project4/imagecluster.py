import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot



# read data from disk
directory = 'C:/Users/redee/Documents/Projects/PyCharmProjects/machine_learning_proj/data/faces_4/'
imgfiles = []
imgfilenames = []
imglist = []

for root, dir, files in os.walk(directory):
    imgfiles.append(files)

imgfilenames = imgfiles[1:21]

for i in range(len(imgfilenames)):
    for j in range(len(imgfilenames[i])):
        # read image
        tmp_name = imgfilenames[i][j].split('_')
        picPath = os.path.join(directory+str(tmp_name[0]), imgfilenames[i][j])
        # tmp_image = Image.open(directory+str(tmp_name[0])+'/'+imgfilenames[i][j])
        tmp_image = Image.open(picPath)
        resize_img = tmp_image.resize((28, 28))
        image_array = np.array(resize_img)
        reshape_img_array = image_array.reshape((1, 784))
        imglist.append(reshape_img_array)

re_img = []
for item in imglist:
    re_img.append(item.reshape(784))

img_array = np.array(re_img)

# decomposition using PCA
pca = PCA(n_components=2)
image_pca = pca.fit_transform(img_array)

# set bandwidth1
bandwidth = estimate_bandwidth(image_pca, quantile=0.15)
# set mean shift function
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# train
ms.fit(image_pca)

# number of clusters
labels = np.unique(ms.labels_)
num_clusters = len(labels)
print("Number of clusters: %d"%num_clusters)
centers = ms.cluster_centers_
# plot
pyplot.scatter(image_pca[:, 0], image_pca[:, 1], c=ms.labels_, s=30, cmap='viridis')
pyplot.scatter(centers[:, 0], centers[:, 1], c='black', s=150, alpha=0.7)
pyplot.show()
