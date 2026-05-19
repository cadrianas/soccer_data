# European Football Analytics: Predictive Modeling & Operational Profiling

This project presents a data science framework for analyzing the competitive structure of elite European football teams. By integrating performance and economic data from the top five domestic leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), the framework identifies the primary drivers of competitive success beyond standard league table outcomes.

## Technical Rigor & Data Engineering

To ensure model reliability and mitigate data bias, the following standards were implemented:

* **Matchday Rate Normalization:** Raw seasonal totals (Goals, xG) were converted to per-game averages to account for differences between 18-team and 20-team leagues.
* **Leakage Prevention:** Features associated with final league rankings (such as final points or goal differentials) were excluded from training inputs, requiring models to rely exclusively on underlying metrics such as `xG_per_game`, `xGA_per_game`, and `Attendance`.
* **Dynamic Tier Labeling:** Instead of fixed rank numbers, target labels were generated using intra-league quantile thresholds. This approach ensures that each team’s status is evaluated relative to its specific league environment.

## Modeling Approach

### 1. Unsupervised Profiling
The Elbow Method was used to determine optimal cluster counts, and K-Means clustering identified four distinct club archetypes, ranging from high-revenue "Super-Elite" teams to "Low-Block" relegation contenders. 



t-SNE dimensionality reduction was applied to map these clusters, confirming clear structural separation in the underlying data.



### 2. Supervised Classification
To forecast performance tiers, five distinct models were trained, each implemented within a standard `scikit-learn` pipeline using `StandardScaler` for feature scaling:

* **Multinomial Logistic Regression:** Employed as the baseline parametric model.
* **Random Forest Classifier:** A parallelized ensemble of 100 trees, configured with `class_weight` set to 'balanced' to prevent majority-class bias.
* **Support Vector Machine (SVM):** Utilized an RBF kernel to address non-linear decision boundaries between performance tiers.
* **Gradient Boosting Machine (GBM):** A sequential model that iteratively refines predictions using residual errors.
* **K-Nearest Neighbors (KNN):** A proximity-based algorithm that classifies teams according to the performance signatures of their five most similar peers.

## Project Organization & Reproducibility

* **Modular Architecture:** The project is organized into dedicated `src/` modules, separating data preprocessing from the model training logic.
* **Validation Strategy:** Datasets were partitioned using stratified splitting to ensure that high-performing "Elite" teams were proportionally represented in both training and testing sets.
* **Evaluation:** Each model run outputs a performance dashboard, specifically tracking Precision, Recall, and F1-scores to ensure that high accuracy is not solely due to majority-class predictions.

---
*Note: Grammarly was used to refine the English grammar and phrasing of this documentation.*