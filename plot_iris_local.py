from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets


def make_meshgrid(arg_x, arg_y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    arg_x: data to base x-axis meshgrid on
    arg_y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    result_xx, result_yy : ndarray
    """
    x_min, x_max = arg_x.min() - 1, arg_x.max() + 1
    y_min, y_max = arg_y.min() - 1, arg_y.max() + 1
    result_xx, result_yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return result_xx, result_yy


def plot_contours(arg_axis, clf, arg_xx, arg_yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    arg_axis: matplotlib axes object
    clf: a classifier
    arg_xx: meshgrid ndarray
    arg_yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    local_z = clf.predict(np.c_[arg_xx.ravel(), arg_yy.ravel()])
    local_z = local_z.reshape(arg_xx.shape)
    result = arg_axis.contourf(arg_xx, arg_yy, local_z, **params)
    return result


# quit early if the output folder doesn't exist
output_folder = './output/'
if not isdir(output_folder):
    print('Output folder ' + output_folder + ' does not exist. Quitting.')
    quit()
else:
    print('output folder is %s' % output_folder)

# import some data to play with
iris = datasets.load_iris()

g = pd.read_csv('./data/synthetic_iris_with_labels.csv')
iris_values = g[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                 'petal width (cm)']].values.astype(np.float32).tolist()
data = np.array(iris_values)
y = g[['target']].values.astype(int).ravel()

run_data = {
    'sepal': {
        'data': data[:, 0:2],
        'x_label': 'Sepal length',
        'y_label': 'Sepal width',
        'output_file': 'iris_sepal_svc_plots_local.png'
    },
    'petal': {
        'data': data[:, 2:4],
        'x_label': 'Petal length',
        'y_label': 'Petal width',
        'output_file': 'iris_petal_svc_plots_local.png'
    }
}
random_state = 1
for feature in ['sepal', 'petal']:
    X = run_data[feature]['data']
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    models = [svm.SVC(kernel='linear', C=C, random_state=random_state),
              svm.LinearSVC(C=C, random_state=random_state),
              svm.SVC(kernel='rbf', gamma=0.7, C=C, random_state=random_state),
              svm.SVC(kernel='poly', degree=3, C=C, random_state=random_state)]
    fitted = [clf.fit(X, y) for clf in models]

    scores = [clf.score(X, y) for clf in models]

    # title for the plots
    titles = ('SVC with linear kernel {0:.2f}'.format(scores[0]),
              'LinearSVC (linear kernel) {0:.2f}'.format(scores[1]),
              'SVC with RBF kernel {0:.2f}'.format(scores[2]),
              'SVC with cubic kernel {0:.2f}'.format(scores[3]))

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    cmap = plt.get_cmap('coolwarm')
    for clf, title, ax in zip(fitted, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, cmap=cmap, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=cmap, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(run_data[feature]['x_label'])
        ax.set_ylabel(run_data[feature]['y_label'])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    output_file = run_data[feature]['output_file']
    long_output_filename = output_folder + output_file
    plt.savefig(long_output_filename)
