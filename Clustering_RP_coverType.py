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
from sklearn.random_projection import SparseRandomProjection
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
trainSize=0.5

  
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
    reduced_X = SparseRandomProjection(n_components=i, random_state=42).fit_transform(X_train_scaled)
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
        plt.title('Score vs. dimensionality for K-means with randomized projection (coverType)')
    else:
        plt.title('Score vs. dimensionality for Gaussian Mixture with randomized projection (coverType)')
    plt.grid(True)
    plt.plot(dim, S_homog[i], label= i+' homogeneity score', linewidth=2)
    plt.plot(dim, S_adjMI[i], label= i+' adjusted mutual info score', linewidth=2)
    plt.plot(dim, S_vm[i], label= i+' v measure score', linewidth=2)
    plt.legend()
    plt.savefig(i+"_RP_Scores_coverType.png")
    
#
