from io import BytesIO, StringIO

from flask import render_template, send_file
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from app import app
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt


def make_dbscan():
    cluster1 = 0
    cluster2 = 0
    centers = [[1, 1], [-1, -1], [1, -1], [1, 0], [0, 1], [-1, 1]]
    # x, labels_true = make_blobs(n_samples=1500, centers=centers, cluster_std=0.3,
    #                             random_state=0)
    x = pd.read_csv("C:\\Users\\Patrick\\Documents\\Clusterlab\\app\\static\\assets\\files\\test-data-clustering.csv", sep=';')
    x = StandardScaler().fit_transform(x)

    db = DBSCAN(eps=0.3, min_samples=10).fit(x)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = x[class_member_mask & core_samples_mask]
        cluster1 = plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=14)

        xy = x[class_member_mask & ~core_samples_mask]
        cluster2 = plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=6)

    return cluster1, cluster2

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()


@app.route('/index/cluster/sampleimage')
def fig():
    cluster1, cluster2 = make_dbscan()

    img = BytesIO()
    plt.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route('/')
@app.route('/index')
def index():
    return render_template('wizard-book-room.html')
