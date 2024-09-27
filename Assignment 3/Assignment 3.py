
# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram

# Create a path for the data
path = "Assignment 3/movies_metadata.csv"
 
# Read the CSV file with appropriate encoding
data = pd.read_csv(path, encoding='latin-1')

# Creating the data sample
sample_movie = data.iloc[:, :][['title', 'budget', 'vote_average', 'vote_count']] #We choose the columns we want to form clusters with
sample_movie['budget'] = pd.to_numeric(sample_movie['budget'], errors='coerce')   #We tranform the budget column from an object to an integer
sample_movie = sample_movie[(sample_movie['vote_count'] > 6000)]                  #We remove all observations with less than 6000 vote counts and no information about budget
sample_movie = sample_movie[(sample_movie['budget'] != 0)]                        #We remove all observations with no information about budget
sample_movie

sample_movie[['budget', 'vote_average', 'vote_count']].values # we look into the newly created sample


clustering = AgglomerativeClustering().fit(sample_movie[['budget', 'vote_average', 'vote_count']].values)
clustering.labels_

# Calculate the linkage matrix using Average Linkage
linkage_matrix = linkage(sample_movie[['budget', 'vote_average', 'vote_count']], method='average') 

# Create the dendrogram
dendrogram(linkage_matrix, labels=sample_movie['title'].tolist(), orientation='right')


# Implementing agglomerative clustering

# First, we assign each point to a cluster.
sample_movie.shape
sample_movie['cluster'] = sample_movie.index
sample_movie.head()


clustering = AgglomerativeClustering().fit(sample_movie[['budget', 'vote_average', 'vote_count']].values)
sample_movie['cluster'] = clustering.labels_

# Determine the number of clusters
num_clusters = len(sample_movie['cluster'].unique())

# Assign each cluster a color
cluster_colors = plt.cm.get_cmap('tab10', num_clusters)  # Use a colormap for distinct colors

# Assign each cluster a color
cluster_labels = [f'cluster {i}' for i in range(num_clusters)]

# Create figure and axis with additional space for legend
plt.figure(figsize=(8,6))

# Plot each point with corresponding color based on the cluster
# Use iterrows() to iterate over rows and access cluster ID based on index
for index, row in sample_movie.iterrows():
    cluster_id = row['cluster']
    plt.scatter(row['vote_average'], row['budget'], c=cluster_colors(cluster_id), label=row['title'])
    plt.text(row['vote_average'] + 0.1, row['budget'], row['title'], fontsize=9)

# Title and labels
plt.title("Movie's average rating vs budget")
plt.xlabel("vote_average")
plt.ylabel("budget")

# Create a legend to show the cluster colors outside the plot
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors(i), markersize=8) for i in range(num_clusters)]
plt.legend(handles, cluster_labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Clusters')

# Display the plot
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to fit the legend
plt.show()

# After the hierarchical clustering is done, we can try to do the K-means clustering.

X = sample_movie[['vote_average', 'budget']].values
movie_titles = sample_movie['title'].values

k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Using the elbow method for find ing the optimal K.

wcss = []

k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(k_values, wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Creating the cluster plot

def kmeans_with_visualization(X, k, max_iters=10):
    # Step 1: Initialization - Randomly initialize k centroids from your defined X
    random_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[random_indices]

    iteration_visualizations = []

    for i in range(max_iters):
        # Step 2: Assignment - Assign each point to the nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Capture current state for visualization
        iteration_visualizations.append((centroids.copy(), labels.copy()))

        # Step 3: Update Centroids - Recalculate centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return iteration_visualizations

# Run k-means algorithm and capture the output at each iteration using your defined X
iteration_visualizations = kmeans_with_visualization(X, k=3)

# Plot the clusters and centroids at each iteration
fig, axes = plt.subplots(1, len(iteration_visualizations), figsize=(20, 4))
for i, (centroids, labels) in enumerate(iteration_visualizations):
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
    axes[i].set_title(f"Iteration {i + 1}")

plt.tight_layout()
plt.show()

# Standardize the data
scaler = StandardScaler()
movie_scaled = scaler.fit_transform(sample_movie[['vote_average', 'budget']])

# Create the distance matrix
distance_matrix = cosine_distances(movie_scaled[:, [0, 1]])
distance_matrix.shape

np.fill_diagonal(distance_matrix, 1)

# Show the first few rows of the scaled data
distance_df = pd.DataFrame(distance_matrix, index=sample_movie['title'], columns=sample_movie['title'])
distance_df

############################

from sklearn.preprocessing import LabelEncoder

le_movie = LabelEncoder()
le_movie.fit(sample_movie['title'])

movie_index = 0  # Change to index instead of titles (0 = Star Wars)

#find similar movies using np.argsort
similar_movies = np.argsort(distance_matrix[movie_index,:])[:5]

# Inverse transformation to get the movie titles
recommended_movies = le_movie.inverse_transform(similar_movies)

# Output the recommended movies
print("Recommended Movies:", recommended_movies)