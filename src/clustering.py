import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def perform_elbow_method(X_scaled, k_range=range(1, 11)):
    """Calculates WCSS for a range of K values."""
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss

def run_kmeans(X_scaled, n_clusters=4):
    """Fits K-Means and returns cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans

def run_tsne(X_scaled, perplexity=20):
    """Performs t-SNE dimensionality reduction."""
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne

def run_clustering_pipeline(df, features=['xG_per_game', 'xGA_per_game', 'Attendance']):
    """Orchestrates the clustering workflow."""
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    optimal_k = 4 # Based on notebook analysis
    labels, model = run_kmeans(X_scaled, n_clusters=optimal_k)

    # Assign back to df
    df.loc[X.index, 'Cluster_ID'] = labels

    # t-SNE
    X_tsne = run_tsne(X_scaled)
    tsne_df = pd.DataFrame(X_tsne, columns=['TSNE_Dim1', 'TSNE_Dim2'], index=X.index)

    return df, X_scaled, labels, tsne_df, optimal_k
