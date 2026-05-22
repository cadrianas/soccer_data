import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_classification_data(df, features, target_col):
    """Prepares stratified train-test splits for classification."""
    clean_df = df.dropna(subset=features + [target_col]).copy()
    X = clean_df[features]
    y = clean_df[target_col].astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_eval_model(pipeline, X_train, X_test, y_train, y_test, target_names):
    """Trains a pipeline and returns evaluation metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'class_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    }
    return metrics, y_pred

def run_classification_pipeline(df, features=['xG_per_game', 'xGA_per_game', 'Attendance']):
    """Executes all classification models and returns results."""

    # Binary Classification (Logistic Regression)
    X_tr_bin, X_te_bin, y_tr_bin, y_te_bin = get_classification_data(df, features, 'Binary_Label')
    log_reg_bin = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    bin_results, _ = train_eval_model(log_reg_bin, X_tr_bin, X_te_bin, y_tr_bin, y_te_bin, ['Non-Elite', 'Elite'])

    # Multi-Class Classification
    X_tr, X_te, y_tr, y_te = get_classification_data(df, features, 'MultiClass_Label')
    target_names = ['Elite (1)', 'Mid-Tier (2)', 'Bottom (3)']

    models = {
        'Logistic Regression': LogisticRegression(solver='lbfgs', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42, probability=True),
        'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    results = {'Binary': bin_results}

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        metrics, _ = train_eval_model(pipe, X_tr, X_te, y_tr, y_te, target_names)
        results[name] = metrics

    return results
