# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:25:46 2019

@author: DMa
"""
from time import time
import csv
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_covtype
#from sklearn.preprocessing import scale

np.random.seed(42)

sample_size = 2000
trainSize=0.8

  
X= fetch_covtype().data
y = fetch_covtype().target

n_digits = len(np.unique(y))


total=len(X)

trainScores=[]
testScores=[]
numIts=[]
#domain=np.linspace(0, 0.8, 5)

# percentage of training set vs whole data set


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size = (1-trainSize), random_state = 3)
# Normalize feature data
#scaler = MinMaxScaler()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

labels=y_train


dim = [2, 3, 4, 5]

km = KM(random_state=42)
gmm = GMM(random_state=42)

Score = defaultdict(list)
adjMI = defaultdict(list)
S_homog = defaultdict(list)
S_adjMI = defaultdict(list)
S_vm = defaultdict(list)


for i in dim:
    reduced_X = PCA(n_components=i, random_state=42).fit_transform(X_train_scaled)
    k=30
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(reduced_X)
    gmm.fit(reduced_X)
    S_homog['km'].append(metrics.homogeneity_score(labels, km.predict(reduced_X)))
    S_homog['gmm'].append(metrics.homogeneity_score(labels, gmm.predict(reduced_X)))   
    S_adjMI['km'].append(metrics.adjusted_mutual_info_score(labels, km.predict(reduced_X)))
    S_adjMI['gmm'].append(metrics.adjusted_mutual_info_score(labels, gmm.predict(reduced_X)) )
    S_vm['km'].append(metrics.v_measure_score(labels, km.predict(reduced_X)))
    S_vm['gmm'].append(metrics.v_measure_score(labels, gmm.predict(reduced_X)))


#plt.legend(['Train', 'Test'], loc='lower right')

    

for i in ['km', 'gmm']:
    plt.figure(figsize=(8, 6))
    plt.xlabel('Number of dimentions')
    plt.ylabel('Score value')
    if i=='km':
        plt.title('Score vs. dimensionality for K-means with PCA (coverType)')
    else:
        plt.title('Score vs. dimensionality for Gaussian Mixture with PCA (coverType)')
    plt.grid(True)
    plt.plot(dim, S_homog[i], label= i+' homogeneity score', linewidth=2)
    plt.plot(dim, S_adjMI[i], label= i+' adjusted mutual info score', linewidth=2)
    plt.plot(dim, S_vm[i], label= i+' v measure score', linewidth=2)
    plt.legend()
    plt.savefig(i+"_PCA_Scores_coverType.png")
    
#
## #############################################################################
## Visualize the results on PCA-reduced data
#reduced_data = PCA(n_components=15).fit_transform(X_scaled)
##kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
##kmeans.fit(reduced_data)
#kmeans=bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#              name="PCA", data=reduced_data)

def draw_2d():
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()