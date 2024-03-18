# Data Mining - K-means and Spectral Clustering

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Development Tools](#2-development-tools)
3. [Visual Journey](#3-visual-journey)
4. [Implementation Highlights](#4-implementation-highlights)
5. [Demo](#5-demo)
6. [Authors](#6-authors)

## 1. Project Overview

This project was developed for submission in **Methods of Data Mining** course at **[Aalto University](https://www.aalto.fi/en/)**.

The main goal was to explore the effectiveness of K-means and Spectral Clustering algorithms with varying values of k (ranging from 2 to 4) in partitioning datasets and to optimize clustering quality using a combination of internal validation indices: Silhouette Index (SI) and Calinski-Harabasz (CH), along with the external validation index Normalized Mutual Information (NMI). Additionally, the project seeks to understand the inherent biases of internal clustering validation indices and their implications on clustering quality assessment.

### 1.1 Datasets Involved

1. **Balls Dataset**: A collection of 2D points (x, y) with class labels, featuring convex-shaped clusters resembling balls. Optimal cluster number: 3.

<div align="center">

<img src="Visuals/Balls/balls_dataset.png" alt="Balls Dataset" style="width:50%;height:50%;">

</div>

2. **Spirals Dataset**: A collection of 2D points (x, y) with class labels, featuring non-convex-shaped clusters resembling spirals. Optimal cluster number: 3.

<div align="center">

<img src="Visuals/Spirals/spirals_dataset.png" alt="Spirals Dataset" style="width:50%;height:50%;">

</div>

These datasets serve as ideal test cases to evaluate the performance of the clustering algorithms and validation indices across different cluster shapes and densities.

### 1.2 Tasks

1. Apply K-means and Spectral Clustering algorithms on both 'balls.txt' and 'spirals.txt' datasets using 2-dimensional spectral embedding, normalized Laplacian matrix and Gaussian kernel.
2. Evaluate clustering results using SI, CH, and NMI indices.

### 1.3 Conclusion

- While internal indices like SI and CH are valuable for assessing clustering quality, they may exhibit biases, especially when clusters are non-convex. They often fail to capture changes in cluster density and shape accurately.
- In contrast, the external index NMI utilizes true labels, providing a more robust evaluation of clustering quality by minimizing bias.

## 2. Development Tools

- **Language**: Python 3.10.13
- **IDE**: PyCharm 2023.2.5
- **Libraries**:
    - **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis library, providing versatile data structures and tools.
    - **[Numpy](https://numpy.org/)**: Support for large, multi-dimensional arrays and matrices, fundamental for scientific computing.
    - **[Matplotlib](https://matplotlib.org/)**: A comprehensive plotting library used for creating high-quality visualizations.

## 3. Visual Journey

This section presents the clustering results of K-means and Spectral Clustering, followed by the evaluation indices SI, CH, and NMI.

<div align="center">

### Balls Dataset

#### Clustering without Spectral Embedding

| <div align="center">2-means<br>(Suboptimal)</div> | <div align="center">3-means<br>(Optimal)</div> | <div align="center">4-means<br>(Suboptimal)</div> |
|------------------------------------|---------------------------------------|---------------------------------------|
| ![2-means without Embedding](Visuals/Balls/WithoutEmbedding/balls_2_means_without_embedding.PNG) | ![3-means without Embedding](Visuals/Balls/WithoutEmbedding/balls_3_means_without_embedding.PNG) | ![4-means without Embedding](Visuals/Balls/WithoutEmbedding/balls_4_means_without_embedding.PNG) |

#### Evaluation of Clustering without Spectral Embedding

![k-means evaluation](Visuals/Balls/WithoutEmbedding/balls_k_means_performance_without_embedding.PNG)

**Note**: The indices accurately reflect the optimal number of clusters (3).

#### Clustering with Spectral Embedding (Original and Embedded Space)

| <div align="center">2-means<br>(Suboptimal)</div> | <div align="center">3-means<br>(Optimal)</div> | <div align="center">4-means<br>(Suboptimal)</div> |
|------------------------------------|---------------------------------------|---------------------------------------|
| ![2-means with Embedding (Original Space)](Visuals/Balls/WithEmbedding/balls_2_means_with_embedding_original.PNG) | ![3-means with Embedding (Original Space)](Visuals/Balls/WithEmbedding/balls_3_means_with_embedding_original.PNG) | ![4-means with Embedding (Original Space)](Visuals/Balls/WithEmbedding/balls_4_means_with_embedding_original.PNG) |
| ![2-means with Embedding (Embedded Space)](Visuals/Balls/WithEmbedding/balls_2_means_with_embedding_embedded.PNG) | ![3-means with Embedding (Embedded Space)](Visuals/Balls/WithEmbedding/balls_3_means_with_embedding_embedded.PNG) | ![4-means with Embedding (Embedded Space)](Visuals/Balls/WithEmbedding/balls_4_means_with_embedding_embedded.PNG) |

#### Evaluation of Clustering with Spectral Embedding

![k-means evaluation](Visuals/Balls/WithEmbedding/balls_k_means_performance_with_embedding.PNG)

**Note**: The indices accurately reflect the optimal number of clusters (3) in the embedded space as well.

### Spirals Dataset

#### Clustering without Spectral Embedding

| <div align="center">2-means<br>(Suboptimal)</div> | <div align="center">3-means<br>(Suboptimal)</div> | <div align="center">4-means<br>(Suboptimal)</div> |
|------------------------------------|---------------------------------------|---------------------------------------|
| ![2-means without Embedding](Visuals/Spirals/WithoutEmbedding/spirals_2_means_without_embedding.PNG) | ![3-means without Embedding](Visuals/Spirals/WithoutEmbedding/spirals_3_means_without_embedding.PNG) | ![4-means without Embedding](Visuals/Spirals/WithoutEmbedding/spirals_4_means_without_embedding.PNG) |

#### Evaluation of Clustering without Spectral Embedding

![k-means evaluation](Visuals/Spirals/WithoutEmbedding/spirals_k_means_performance_without_embedding.PNG)

**Note**: The indices do not reflect the optimal number of clusters (3) anymore due to the non-convex shape of clusters.

#### Clustering with Spectral Embedding (Original and Embedded Space)

| <div align="center">2-means<br>(Suboptimal)</div> | <div align="center">3-means<br>(Optimal)</div> | <div align="center">4-means<br>(Suboptimal)</div> |
|------------------------------------|---------------------------------------|---------------------------------------|
| ![2-means with Embedding (Original Space)](Visuals/Spirals/WithEmbedding/spirals_2_means_with_embedding_original.PNG) | ![3-means with Embedding (Original Space)](Visuals/Spirals/WithEmbedding/spirals_3_means_with_embedding_original.PNG) | ![4-means with Embedding (Original Space)](Visuals/Spirals/WithEmbedding/spirals_4_means_with_embedding_original.PNG) |
| ![2-means with Embedding (Embedded Space)](Visuals/Spirals/WithEmbedding/spirals_2_means_with_embedding_embedded.PNG) | ![3-means with Embedding (Embedded Space)](Visuals/Spirals/WithEmbedding/spirals_3_means_with_embedding_embedded.PNG) | ![4-means with Embedding (Embedded Space)](Visuals/Spirals/WithEmbedding/spirals_4_means_with_embedding_embedded.PNG) |

#### Evaluation of Clustering with Spectral Embedding

![k-means evaluation](Visuals/Spirals/WithEmbedding/spirals_k_means_performance_with_embedding.PNG)

**Note**:  The indices now accurately reflect the optimal number of clusters (3) thanks to the transformation facilitated by spectral embedding.

</div>

## 4. Implementation Highlights

**Highlight 1/3**: The K-means algorithm implemented here utilizes the K-means++ initialization method for centroid initialization and iteratively updates centroids and cluster assignments until convergence or a maximum number of iterations is reached.

```python
def k_means(data, k, dist_func, max_iterations=100):
    # Initialize centroids using K-means++ initialization
    centroids = initialize_centroids(data, k)
    clusters = [[] for _ in centroids]

    # Repeat clustering and updating centroids until convergence or maximum iterations
    for _ in range(max_iterations):
        clusters = [[] for _ in centroids]
        for i, row in enumerate(data.values):
            distances = [dist_func(row, centroid) for centroid in centroids]
            best_k = np.argmin(distances)
            clusters[best_k].append(i)
            
        # Update centroids with the mean of each cluster's data points
        # Break loop if centroids do not change significantly
        new_centroids = [np.mean(data.iloc[indices], axis=0).tolist() for indices in clusters]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return clusters, centroids
```

**Highlight 2/3**: The spectral embedding function utilizes a Laplacian matrix computed with a Gaussian kernel to project high-dimensional data into a lower-dimensional space. It selects the 'dim' number of eigenvectors associated with the smallest eigenvalues to reduce dimensionality while preserving the important data structure.

```python
def spectral_embedding(data, sigma=1.0, dim=1, normalized_laplacian=True):
    # Calculate Laplacian matrix (normalized by default) with Gaussian kernel
    laplacian_matrix = calc_laplacian_matrix(data, sigma, normalized_laplacian)

    # Calculate eigenvalues and eigenvectors of Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Find the index of the first non-zero eigenvalue
    non_zero_index = np.argmax(eigenvalues > 0.0001)

    # Check if dim exceeds the number of available eigenvectors
    if dim > eigenvectors.shape[1] - non_zero_index:
        raise ValueError("Dimensionality 'dim' exceeds the number of available eigenvectors.")

    # Perform the embedding: get the first "dim" number of sorted eigenvectors
    embedding = eigenvectors[:, non_zero_index: non_zero_index + dim]
    columns = [f"Dimension_{i + 1}" for i in range(dim)]
    return pd.DataFrame(embedding, columns=columns)
```

**Highlight 3/3**: The Silhouette Index function computes silhouette scores for each data point based on its intra-cluster and nearest inter-cluster distances, followed by averaging, providing a single metric to evaluate the overall clustering quality.

```python
def si(data, labels, dist_func):
    silhouette_scores = np.zeros(len(data))

    # Calculate the silhouette score for every data point
    for i in range(len(data)):
        a = mean_intra_cluster_distance(i, data, labels, dist_func)
        b = min_mean_inter_cluster_distance(i, data, labels, dist_func)
        max_ab = np.maximum(a, b)
        if max_ab != 0:
            silhouette_scores[i] = (b - a) / max_ab

    # Calculate the average silhouette score to get the Silhouette Index
    return np.mean(silhouette_scores)
```

## 5. Demo

The project is not uploaded, and access to its contents is provided upon request.

## 6. Authors
- Anlin Sun
- Yaojia Wang
- Ferenc Szendrei

[Back to Top](#data-mining---k-means-and-spectral-clustering)
