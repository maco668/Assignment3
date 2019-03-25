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
#from sklearn.preprocessing import scale

np.random.seed(42)

sample_size = 2000
readFileName = 'Frogs_MFCCs.csv'


with open(readFileName) as df:
  df_iter = csv.reader(df, delimiter=',', quotechar='"')
  data=[x for x in df_iter]
  data = np.array(data)
  
# skip 1st row and last 4 columns
X = data[1:,0:-4].astype(np.float)

labels = data[1:,-2]
n_digits = len(np.unique(labels))


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
#X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_hot = one_hot.fit_transform(labels.reshape(-1, 1)).todense()
#y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

#for i in data[1:]: # skip title row
#  X = [float(x) for x in i[0:-4]]
## Normalize feature data
#scaler = MinMaxScaler()
#
#X_scaled = scaler.fit_transform(X)




print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    return estimator


clusters =  [2,5,10,15,20,25,30,35]

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
    km.fit(X_scaled)
    gmm.fit(X_scaled)
    Score['km'].append( km.score(X_scaled))
    Score['gmm'].append( gmm.score(X_scaled))
    S_homog['km'].append(metrics.homogeneity_score(labels, km.predict(X_scaled)))
    S_homog['gmm'].append(metrics.homogeneity_score(labels, gmm.predict(X_scaled)))   
    S_adjMI['km'].append(metrics.adjusted_mutual_info_score(labels, km.predict(X_scaled)))
    S_adjMI['gmm'].append(metrics.adjusted_mutual_info_score(labels, gmm.predict(X_scaled)) )
    S_vm['km'].append(metrics.v_measure_score(labels, km.predict(X_scaled)))
    S_vm['gmm'].append(metrics.v_measure_score(labels, gmm.predict(X_scaled)))

plt.figure(figsize=(9.6, 7.2))
plt.xlabel('Number of clusters')
plt.ylabel('Score value')
plt.title('Score vs. Cluster number for K-mean and Gaussian Mixture (species)')
plt.grid(True)
#plt.legend(['Train', 'Test'], loc='lower right')

    

for i in ['km', 'gmm']:
    plt.plot(clusters, S_homog[i], label= i+' homogeneity score', linewidth=2)
    plt.plot(clusters, S_adjMI[i], label= i+' adjusted mutual info score', linewidth=2)
    plt.plot(clusters, S_vm[i], label= i+' v measure score', linewidth=2)
    
plt.legend()
plt.savefig("KM_GMM_Scores_species.png")
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