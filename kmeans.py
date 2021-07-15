import numpy as np
import random
import copy
import pandas as pd
from scipy.spatial.distance import cdist as dist


def centroid_choice(X, k):

    # Indicies tracker
    random_indicies = []

    # Iterate through potential indicies until 'k' unique in indicies found.
    while len(random_indicies) < k:

        index = random.randint(0, len(X)-1)

        if index not in random_indicies:
            random_indicies.append(index)

    return random_indicies


def kmeans_plus(X, k):

    # Indicies and centroid trackers
    indicies = []
    centroids = []

    # First centroid index chosen at random, and kept track of
    index = random.randint(0, len(X))
    indicies.append(index)

    # Use list of indicies to "Slice" data
    centroids.append(index)
    centroids = X[index, :]
    centroids = np.array(centroids).reshape(1, -1)

    for i in range(k-1):

        # Centroids in terms of X, and distance from each point in X to all other points
        centroids = X[indicies, :]
        distances = dist(X, centroids, 'euclidean')**2

        # Stack horizontally to get "rows" of distances from every centroid, to each point
        row_matrix = np.hstack(distances)
        row_matrix = row_matrix.reshape(-1, len(distances))

        # Put into Dataframe, take minimum of each column to find closest centroid to that point
        df = pd.DataFrame(row_matrix)
        min_dist = df.min(axis=0)
        min_dist = np.array(min_dist)

        # Create pobability distribution from minimum distances
        prob_of_choice = min_dist/np.sum(min_dist)

        # Randomly choose value from min_dist using probability distribution above as a parameter
        result = np.random.choice(min_dist, p=prob_of_choice)

        # Get index of result in min_dist and add to indicie tracker
        ind = np.where(min_dist == result)
        indicies.append(ind[0][0])

    return indicies


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):

    # 1) Select k unique points from X as initial centroids (call those initial centroid indicies 'm')

    if centroids == 'kmeans++':
        # Using k++ implementaion
        m = kmeans_plus(X, k)
    else:
        # random centroid implementaion
        m = centroid_choice(X, k)

    # 2) For every point x in 'X':
    #    - Find distance and then compute the argmin of distance(x, m)  <-- which finds the closest centroid to x.
    #    - After finding closest centroid to x, add x to cluster associated with that centroid

    centroids = X[m, :]
    distances = dist(X, centroids, 'euclidean')
    labels = np.array([np.argmin(i) for i in distances])

    # 3) Iterate through, and calculate new label, distances, and centroids
    for ind in range(max_iter):
        centroids = []

        for i in range(k):
            # Take mean of clusters, and use as new centroid points
            centroid_mean = X[labels == i].mean(axis=0)
            centroids.append(centroid_mean)

        # New centroids using the means of each cluster
        centroids = np.vstack(centroids)

        # Check the mean of the difference between current centroid and previous against a tolerance
        if ind >= 1 and abs(np.mean(centroids - last_centroids)) <= tolerance:
            # Finished if both parameters met
            iterations = ind
            return last_centroids, labels, iterations

        # New distances and labels calculated for centroids
        distances = dist(X, centroids, 'euclidean')
        labels = np.array([np.argmin(i) for i in distances])

        # Keep track of previous centroids. Used for comparison in next iteration
        last_centroids = copy.deepcopy(centroids)
        iterations = 0

    return last_centroids, labels, iterations
