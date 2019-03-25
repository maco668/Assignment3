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
readFileName = 'Frogs_MFCCs.csv'
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


#for i in data[1:]: # skip title row
#  X = [float(x) for x in i[0:-4]]
## Normalize feature data
#scaler = MinMaxScaler()
#
#X_scaled = scaler.fit_transform(X)


clusters =  [2,4,8,16,32,48]

km = KM(random_state=42)
gmm = GMM(random_state=42)

Score = defaultdict(list)
adjMI = defaultdict(list)
S_homog = defaultdict(list)
S_adjMI = defaultdict(list)
S_vm = defaultdict(list)

for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(X_train_scaled)
    gmm.fit(X_train_scaled)
    S_homog['km'].append(metrics.homogeneity_score(y_test, km.predict(X_test_scaled)))
    S_homog['gmm'].append(metrics.homogeneity_score(y_test, gmm.predict(X_test_scaled)))   
    S_adjMI['km'].append(metrics.adjusted_mutual_info_score(y_test, km.predict(X_test_scaled)))
    S_adjMI['gmm'].append(metrics.adjusted_mutual_info_score(y_test, gmm.predict(X_test_scaled)) )
    S_vm['km'].append(metrics.v_measure_score(y_test, km.predict(X_test_scaled)))
    S_vm['gmm'].append(metrics.v_measure_score(y_test, gmm.predict(X_test_scaled)))

    
#

for i in ['km', 'gmm']:
    plt.figure(figsize=(8, 6))
    plt.xlabel('Number of clusters')
    plt.ylabel('Score value')
    if i=='km':
        plt.title('Score vs. Cluster number for K-means with NMF (coverType)')
    else:
        plt.title('Score vs. Cluster number for Gaussian Mixture with NMF (coverType)')
    plt.grid(True)
    plt.plot(clusters, S_homog[i], label= i+' homogeneity score', linewidth=2)
    plt.plot(clusters, S_adjMI[i], label= i+' adjusted mutual info score', linewidth=2)
    plt.plot(clusters, S_vm[i], label= i+' v measure score', linewidth=2)
    plt.legend()
    plt.savefig(i+"_Scores_coverType.png")


## #############################################################################
## Visualize the results on PCA-reduced data
#reduced_data = PCA(n_components=15).fit_transform(X_scaled)
##kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
##kmeans.fit(reduced_data)
#kmeans=bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#              name="PCA", data=reduced_data)
