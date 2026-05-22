import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.paths import RESULTS_FIGURES

def setup_style():
    """Sets the global aesthetic style for plots."""
    sns.set_theme(style="whitegrid")

def plot_elbow_curve(k_range, wcss):
    """Plots the Elbow method curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    plt.title('The Elbow Method: Evaluating Optimal Cluster Count (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig(RESULTS_FIGURES / 'elbow_curve.png')
    plt.close()

def plot_tactical_clusters(df, optimal_k):
    """Plots tactical dispersal: xG vs xGA."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x='xG_per_game', y='xGA_per_game',
        hue='Cluster_ID', palette='viridis', s=100, alpha=0.8, edgecolor='black'
    )
    plt.gca().invert_yaxis()
    plt.title(f'Tactical Cluster Dispersal (k={optimal_k})')
    plt.xlabel('xG Per Game')
    plt.ylabel('xGA Per Game')
    plt.savefig(RESULTS_FIGURES / 'tactical_clusters.png')
    plt.close()

def plot_tsne(tsne_df):
    """Plots t-SNE manifold projection."""
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=tsne_df, x='TSNE_Dim1', y='TSNE_Dim2',
        hue='Cluster_ID', palette='viridis', s=110, alpha=0.85, edgecolor='black'
    )
    plt.title('t-SNE Low-Dimensional Projection')
    plt.savefig(RESULTS_FIGURES / 'tsne_projection.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, model_name, target_names):
    """Plots a heatmap for the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(RESULTS_FIGURES / f'cm_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def plot_country_distribution(df):
    """Plots team counts per country."""
    plt.figure(figsize=(10, 6))
    df['Country'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Teams across European Leagues')
    plt.xlabel('Country')
    plt.ylabel('Number of Teams')
    plt.xticks(rotation=45)
    plt.savefig(RESULTS_FIGURES / 'country_distribution.png')
    plt.close()
