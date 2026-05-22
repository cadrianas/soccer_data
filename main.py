import pandas as pd
from src.preprocessing import run_preprocessing_pipeline
from src.clustering import run_clustering_pipeline, perform_elbow_method
from src.classification import run_classification_pipeline
from src.visualisation import (
    setup_style, plot_country_distribution, plot_elbow_curve,
    plot_tactical_clusters, plot_tsne, plot_confusion_matrix
)
from src.utils.paths import RESULTS_LOGS

def main():
    # 0. Setup
    setup_style()
    print("--- Starting European Football Analytics Pipeline ---")

    # 1. Preprocessing
    print("\n[1/4] Running Preprocessing...")
    df = run_preprocessing_pipeline()
    plot_country_distribution(df)

    # 2. Clustering
    print("\n[2/4] Running Unsupervised Clustering...")
    cluster_features = ['xG_per_game', 'xGA_per_game', 'Attendance']

    # Optional: Elbow Method data generation
    from sklearn.preprocessing import StandardScaler
    X_scaled_temp = StandardScaler().fit_transform(df[cluster_features].dropna())
    wcss = perform_elbow_method(X_scaled_temp)
    plot_elbow_curve(range(1, 11), wcss)

    df, X_scaled, labels, tsne_df, optimal_k = run_clustering_pipeline(df, cluster_features)

    plot_tactical_clusters(df, optimal_k)
    tsne_df['Cluster_ID'] = df.loc[tsne_df.index, 'Cluster_ID']
    plot_tsne(tsne_df)

    # 3. Classification
    print("\n[3/4] Running Supervised Classification...")
    classification_results = run_classification_pipeline(df, cluster_features)

    # Generate Confusion Matrices
    plot_confusion_matrix(classification_results['Binary']['conf_matrix'], 'Logistic Regression Binary', ['Non-Elite', 'Elite'])

    multi_class_models = ['Logistic Regression', 'Random Forest', 'SVM', 'GBM', 'KNN']
    multi_target_names = ['Elite (1)', 'Mid-Tier (2)', 'Bottom (3)']

    for model_name in multi_class_models:
        plot_confusion_matrix(classification_results[model_name]['conf_matrix'], model_name, multi_target_names)

    # 4. Export & Reporting
    print("\n[4/4] Generating Reports...")
    with open(RESULTS_LOGS / "classification_report.txt", "w") as f:
        for model, res in classification_results.items():
            f.write(f"=== {model} ===\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(str(pd.DataFrame(res['class_report']).transpose()))
            f.write("\n\n")

    print(f"\n--- Pipeline Complete! Results saved in 'data/processed/' and 'results/' ---")

if __name__ == "__main__":
    main()
