# Implementation of K Means Clustering for Customer Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the outputs and end the program

## Program:

Program to implement the K Means Clustering for Customer Segmentation.

Developed by : KEERTHIVASAN S

RegisterNumber : 212223220046

```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data = pd.read_csv("/Mall_Customers.csv")
data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
colors = ['r', 'g', 'b', 'c', 'm']

for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()
```

## Output:

![ex10op1](https://github.com/user-attachments/assets/262542e9-798d-4603-9653-1ce5f9f400fc)

![ex10op2](https://github.com/user-attachments/assets/047073e1-7b05-4df0-a44b-405059962d32)

![ex10op3](https://github.com/user-attachments/assets/fdee9b18-d025-4b83-8e85-e9f2d3989bdc)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
