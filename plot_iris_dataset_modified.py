import logging
import time

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

__doc__ = \
    """
    =========================================================
    The Iris Dataset
    =========================================================
    This data sets consists of 3 different types of irises'
    (Setosa, Versicolour, and Virginica) petal and sepal
    length, stored in a 150x4 numpy.ndarray
    
    The rows being the samples and the columns being:
    Sepal Length, Sepal Width, Petal Length	and Petal Width.
    
    The below plot uses the first two features.
    See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
    information on this dataset.
    """
logger.debug(__doc__)

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the training points
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# ax1.xlabel('Sepal length')
# ax1.ylabel('Sepal width')

# ax1.xlim(x_min, x_max)
# ax1.ylim(y_min, y_max)
# ax1.xticks(())
# ax1.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
# ax2 = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')  # , s=40)
ax2.set_title("First three PCA directions")
ax2.set_xlabel("1st eigenvector")
# ax2.w_xaxis.set_ticklabels([])
ax2.set_ylabel("2nd eigenvector")
# ax2.w_yaxis.set_ticklabels([])
# ax2.set_zlabel("3rd eigenvector")
# ax2.w_zaxis.set_ticklabels([])

output_file = './output/pca_iris_plot.png'
logger.debug('saving plots to %s' % output_file)
plt.savefig(output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
