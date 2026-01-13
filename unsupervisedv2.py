"""
Telecom Fraud Detection - Unsupervised Anomaly Detection System
================================================================
This module implements unsupervised learning models for detecting fraudulent
activities in IPDR and CDR datasets using multiple anomaly detection techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, silhouette_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def extract_time_features(df, timestamp_col='timestamp'):
    """
    Extract temporal features from timestamp column.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    df['hour'] = df[timestamp_col].dt.hour
    df['weekday'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    return df


def encode_categorical_features(df, cat_columns, max_categories=50):
    """
    Encode categorical features using label encoding.
    
    Args:
        df: Input DataFrame
        cat_columns: List of categorical column names
        max_categories: Maximum unique values to encode (prevents memory issues)
    
    Returns:
        DataFrame with encoded features, dictionary of encoders
    """
    df = df.copy()
    encoders = {}
    
    for col in cat_columns:
        if col in df.columns:
            # Handle high cardinality by grouping rare categories
            value_counts = df[col].value_counts()
            if len(value_counts) > max_categories:
                top_categories = value_counts.head(max_categories).index
                df[col] = df[col].apply(
                    lambda x: x if x in top_categories else 'OTHER'
                )
            
            # Encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def preprocess_ipdr(df, drop_cols=None):
    """
    Preprocess IPDR dataset for unsupervised learning.
    
    Args:
        df: Raw IPDR DataFrame
        drop_cols: Additional columns to drop
    
    Returns:
        Preprocessed features, labels, feature names, preprocessor objects
    """
    df = df.copy()
    
    # Extract time features
    if 'timestamp' in df.columns:
        df = extract_time_features(df, 'timestamp')
    
    # Store labels if available
    labels = df['is_fraud'].values if 'is_fraud' in df.columns else None
    
    # Identify columns to drop (identifiers and non-feature columns)
    default_drop = ['event_id', 'imei', 'imsi', 'timestamp', 'is_fraud', 
                    'user_id', 'ip_src', 'ip_dst', 'anomaly_type', 'runbook_id',
                    'session_id', 'session_start', 'session_end']
    if drop_cols:
        default_drop.extend(drop_cols)
    drop_cols_final = [col for col in default_drop if col in df.columns]
    
    # Separate categorical and numerical columns
    cat_cols = ['domain', 'protocol', 'cell_id', 'domain_topk']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df, cat_cols)
    
    # Drop identifier columns
    df = df.drop(columns=drop_cols_final)
    
    # Identify remaining non-numeric columns and drop them
    non_numeric_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if non_numeric_cols:
        print(f"  ⚠️  Dropping additional non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    
    # Handle missing values - use median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Scale numerical features
    scaler = StandardScaler()
    feature_names = df.columns.tolist()
    X_scaled = scaler.fit_transform(df)
    
    preprocessors = {
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names
    }
    
    return X_scaled, labels, feature_names, preprocessors


def preprocess_cdr(df, drop_cols=None):
    """
    Preprocess CDR dataset for unsupervised learning.
    
    Args:
        df: Raw CDR DataFrame
        drop_cols: Additional columns to drop
    
    Returns:
        Preprocessed features, labels, feature names, preprocessor objects
    """
    df = df.copy()
    
    # Extract time features if timestamp column exists (not already extracted)
    if 'call_start_time' in df.columns:
        df = extract_time_features(df, 'call_start_time')
    elif 'timestamp' in df.columns and 'hour' not in df.columns:
        df = extract_time_features(df, 'timestamp')
    
    # Feature engineering for CDR (only if not already present)
    if 'duration' in df.columns and 'duration_z' not in df.columns:
        df['duration_z'] = (df['duration'] - df['duration'].mean()) / df['duration'].std()
    
    if 'bytes_sent' in df.columns and 'bytes_sent_z' not in df.columns:
        df['bytes_sent_z'] = (df['bytes_sent'] - df['bytes_sent'].mean()) / df['bytes_sent'].std()
    
    # Count unique IMEIs per IMSI (device shift detection)
    if 'imsi' in df.columns and 'imei' in df.columns:
        imei_counts = df.groupby('imsi')['imei'].nunique().to_dict()
        df['imei_shift_count'] = df['imsi'].map(imei_counts)
    
    # Cell tower change patterns
    if 'cell_id' in df.columns and 'imsi' in df.columns:
        cell_changes = df.groupby('imsi')['cell_id'].nunique().to_dict()
        df['cell_tower_changes'] = df['imsi'].map(cell_changes)
    
    # Store labels if available
    labels = df['is_fraud'].values if 'is_fraud' in df.columns else None
    
    # Identify columns to drop (identifiers and non-feature columns)
    default_drop = ['call_id', 'event_id', 'imei', 'imsi', 'call_start_time', 
                    'timestamp', 'is_fraud', 'user_id', 'ip_src', 'ip_dst',
                    'anomaly_type', 'runbook_id', 'session_id', 'session_start', 
                    'session_end']
    if drop_cols:
        default_drop.extend(drop_cols)
    drop_cols_final = [col for col in default_drop if col in df.columns]
    
    # Separate categorical and numerical columns
    cat_cols = ['cell_id', 'call_type', 'domain', 'protocol', 'domain_topk']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df, cat_cols)
    
    # Drop identifier columns
    df = df.drop(columns=drop_cols_final)
    
    # Identify remaining non-numeric columns and drop them
    non_numeric_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if non_numeric_cols:
        print(f"  ⚠️  Dropping additional non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    
    # Handle missing values - use median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Scale numerical features
    scaler = StandardScaler()
    feature_names = df.columns.tolist()
    X_scaled = scaler.fit_transform(df)
    
    preprocessors = {
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names
    }
    
    return X_scaled, labels, feature_names, preprocessors


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Train Isolation Forest model.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of outliers
        random_state: Random seed
    
    Returns:
        Trained model, anomaly scores
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)
    
    # Convert to anomaly scores (higher = more anomalous)
    anomaly_scores = -scores
    
    return model, anomaly_scores


def train_lof(X, n_neighbors=20, contamination=0.1):
    """
    Train Local Outlier Factor model.
    
    Args:
        X: Feature matrix
        n_neighbors: Number of neighbors
        contamination: Expected proportion of outliers
    
    Returns:
        Trained model, anomaly scores
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_
    
    # Convert to anomaly scores (higher = more anomalous)
    anomaly_scores = -scores
    
    return model, anomaly_scores


def train_pca_reconstruction(X, n_components=0.95, random_state=42):
    """
    Train PCA and compute reconstruction error.
    
    Args:
        X: Feature matrix
        n_components: Number of components or variance ratio
        random_state: Random seed
    
    Returns:
        Trained PCA model, reconstruction errors
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Compute reconstruction error
    reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    return pca, reconstruction_errors


def train_kmeans_anomaly(X, n_clusters=8, random_state=42):
    """
    Train KMeans and compute distance to nearest cluster center.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Trained KMeans model, anomaly scores
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    model.fit(X)
    
    # Compute distance to nearest cluster center
    distances = model.transform(X)
    anomaly_scores = np.min(distances, axis=1)
    
    return model, anomaly_scores


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(y_true, anomaly_scores, threshold_percentile=90):
    """
    Evaluate anomaly detection model.
    
    Args:
        y_true: True labels
        anomaly_scores: Anomaly scores from model
        threshold_percentile: Percentile threshold for classification
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert scores to binary predictions
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    y_pred = (anomaly_scores >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'auc': roc_auc_score(y_true, anomaly_scores),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def print_evaluation_results(model_name, metrics):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"AUC Score:        {metrics['auc']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_correlation_heatmap(df, feature_names, title="Feature Correlation Heatmap"):
    """Plot correlation heatmap of features."""
    plt.figure(figsize=(14, 10))
    
    # Create DataFrame for correlation
    df_corr = pd.DataFrame(df, columns=feature_names)
    corr = df_corr.corr()
    
    # Plot heatmap
    sns.heatmap(corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(df, feature_names, n_features=6):
    """Plot histograms of important features."""
    df_plot = pd.DataFrame(df, columns=feature_names)
    
    # Select features with highest variance
    variances = df_plot.var().sort_values(ascending=False)
    top_features = variances.head(n_features).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        axes[idx].hist(df_plot[feature], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{feature}', fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_anomaly_scatter(df, labels, anomaly_scores, feature_names):
    """Plot scatter plot of anomalies."""
    df_plot = pd.DataFrame(df, columns=feature_names)
    
    # Find duration and bytes columns (or use first two features)
    duration_col = next((col for col in feature_names if 'duration' in col.lower()), feature_names[0])
    bytes_col = next((col for col in feature_names if 'byte' in col.lower()), feature_names[1])
    
    plt.figure(figsize=(12, 7))
    
    # Normalize scores for coloring
    norm_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    scatter = plt.scatter(df_plot[duration_col], df_plot[bytes_col], 
                         c=norm_scores, cmap='RdYlBu_r', 
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Highlight true frauds if labels available
    if labels is not None:
        fraud_mask = labels == 1
        plt.scatter(df_plot.loc[fraud_mask, duration_col], 
                   df_plot.loc[fraud_mask, bytes_col],
                   c='red', s=100, alpha=0.8, marker='x', 
                   linewidths=2, label='True Fraud')
    
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel(duration_col, fontsize=12, fontweight='bold')
    plt.ylabel(bytes_col, fontsize=12, fontweight='bold')
    plt.title('Anomaly Detection Scatter Plot', fontsize=14, fontweight='bold')
    if labels is not None:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('anomaly_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, anomaly_scores, model_name):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    auc = roc_auc_score(y_true, anomaly_scores)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pca_projection(X, pca_model, labels, anomaly_scores):
    """Plot PCA 2D projection."""
    X_pca = pca_model.transform(X)
    
    plt.figure(figsize=(12, 7))
    
    # Normalize scores for coloring
    norm_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=norm_scores, cmap='RdYlBu_r',
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Highlight true frauds if labels available
    if labels is not None:
        fraud_mask = labels == 1
        plt.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1],
                   c='red', s=100, alpha=0.8, marker='x',
                   linewidths=2, label='True Fraud')
    
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)', 
               fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)', 
               fontsize=12, fontweight='bold')
    plt.title('PCA 2D Projection with Anomaly Scores', fontsize=14, fontweight='bold')
    if labels is not None:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_projection.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_anomaly_distribution(anomaly_scores_dict, labels):
    """Plot distribution of anomaly scores for different models."""
    n_models = len(anomaly_scores_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, scores) in enumerate(anomaly_scores_dict.items()):
        # Separate scores by fraud/normal
        if labels is not None:
            normal_scores = scores[labels == 0]
            fraud_scores = scores[labels == 1]
            
            axes[idx].hist(normal_scores, bins=50, alpha=0.7, 
                          label='Normal', color='steelblue', edgecolor='black')
            axes[idx].hist(fraud_scores, bins=50, alpha=0.7, 
                          label='Fraud', color='orangered', edgecolor='black')
            axes[idx].legend()
        else:
            axes[idx].hist(scores, bins=50, alpha=0.7, 
                          color='steelblue', edgecolor='black')
        
        axes[idx].set_title(f'{model_name}', fontweight='bold')
        axes[idx].set_xlabel('Anomaly Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def run_ipdr_pipeline(df, output_prefix='ipdr_unsupervised'):
    """
    Complete pipeline for IPDR unsupervised anomaly detection.
    
    Args:
        df: IPDR DataFrame
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary containing models, scores, and metrics
    """
    print("\n" + "="*80)
    print("IPDR UNSUPERVISED ANOMALY DETECTION PIPELINE")
    print("="*80)
    
    # Preprocessing
    print("\n[1/5] Preprocessing IPDR data...")
    X, labels, feature_names, preprocessors = preprocess_ipdr(df)
    print(f"✓ Processed {X.shape[0]} samples with {X.shape[1]} features")
    
    # Train models
    print("\n[2/5] Training anomaly detection models...")
    
    print("  → Training Isolation Forest...")
    if_model, if_scores = train_isolation_forest(X)
    
    print("  → Training Local Outlier Factor...")
    lof_model, lof_scores = train_lof(X)
    
    print("  → Training PCA Reconstruction...")
    pca_model, pca_scores = train_pca_reconstruction(X)
    
    print("✓ All models trained successfully")
    
    # Store results
    models = {
        'isolation_forest': if_model,
        'lof': lof_model,
        'pca': pca_model
    }
    
    scores = {
        'Isolation Forest': if_scores,
        'LOF': lof_scores,
        'PCA Reconstruction': pca_scores
    }
    
    # Evaluation
    print("\n[3/5] Evaluating models...")
    metrics = {}
    
    if labels is not None:
        for model_name, anomaly_scores in scores.items():
            model_metrics = evaluate_model(labels, anomaly_scores)
            metrics[model_name] = model_metrics
            print_evaluation_results(model_name, model_metrics)
    
    # Save scores
    print("\n[4/5] Saving results...")
    scores_df = pd.DataFrame(scores)
    if labels is not None:
        scores_df['true_label'] = labels
    scores_df.to_csv(f'{output_prefix}_scores.csv', index=False)
    print(f"✓ Scores saved to {output_prefix}_scores.csv")
    
    # Save models
    with open(f'{output_prefix}_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'preprocessors': preprocessors
        }, f)
    print(f"✓ Models saved to {output_prefix}_models.pkl")
    
    # Visualizations
    print("\n[5/5] Generating visualizations...")
    
    plot_correlation_heatmap(X, feature_names, "IPDR Feature Correlation")
    plot_feature_distributions(X, feature_names)
    plot_anomaly_scatter(X, labels, if_scores, feature_names)
    
    if labels is not None:
        plot_roc_curve(labels, if_scores, 'Isolation Forest')
        plot_roc_curve(labels, lof_scores, 'LOF')
        plot_roc_curve(labels, pca_scores, 'PCA Reconstruction')
    
    plot_pca_projection(X, pca_model, labels, if_scores)
    plot_anomaly_distribution(scores, labels)
    
    print("✓ All visualizations saved")
    
    print("\n" + "="*80)
    print("IPDR PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return {
        'models': models,
        'scores': scores,
        'metrics': metrics,
        'preprocessors': preprocessors,
        'feature_names': feature_names
    }


def run_cdr_pipeline(df, output_prefix='cdr_unsupervised'):
    """
    Complete pipeline for CDR unsupervised anomaly detection.
    
    Args:
        df: CDR DataFrame
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary containing models, scores, and metrics
    """
    print("\n" + "="*80)
    print("CDR UNSUPERVISED ANOMALY DETECTION PIPELINE")
    print("="*80)
    
    # Preprocessing
    print("\n[1/5] Preprocessing CDR data...")
    X, labels, feature_names, preprocessors = preprocess_cdr(df)
    print(f"✓ Processed {X.shape[0]} samples with {X.shape[1]} features")
    
    # Train models
    print("\n[2/5] Training anomaly detection models...")
    
    print("  → Training Isolation Forest...")
    if_model, if_scores = train_isolation_forest(X)
    
    print("  → Training Local Outlier Factor...")
    lof_model, lof_scores = train_lof(X)
    
    print("  → Training KMeans Anomaly Detection...")
    kmeans_model, kmeans_scores = train_kmeans_anomaly(X)
    
    print("✓ All models trained successfully")
    
    # Store results
    models = {
        'isolation_forest': if_model,
        'lof': lof_model,
        'kmeans': kmeans_model
    }
    
    scores = {
        'Isolation Forest': if_scores,
        'LOF': lof_scores,
        'KMeans Distance': kmeans_scores
    }
    
    # Evaluation
    print("\n[3/5] Evaluating models...")
    metrics = {}
    
    if labels is not None:
        for model_name, anomaly_scores in scores.items():
            model_metrics = evaluate_model(labels, anomaly_scores)
            metrics[model_name] = model_metrics
            print_evaluation_results(model_name, model_metrics)
        
        # Compute silhouette score for KMeans
        silhouette = silhouette_score(X, kmeans_model.labels_)
        print(f"\nKMeans Silhouette Score: {silhouette:.4f}")
    
    # Save scores
    print("\n[4/5] Saving results...")
    scores_df = pd.DataFrame(scores)
    if labels is not None:
        scores_df['true_label'] = labels
    scores_df.to_csv(f'{output_prefix}_scores.csv', index=False)
    print(f"✓ Scores saved to {output_prefix}_scores.csv")
    
    # Save models
    with open(f'{output_prefix}_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'preprocessors': preprocessors
        }, f)
    print(f"✓ Models saved to {output_prefix}_models.pkl")
    
    # Visualizations
    print("\n[5/5] Generating visualizations...")
    
    plot_correlation_heatmap(X, feature_names, "CDR Feature Correlation")
    plot_feature_distributions(X, feature_names)
    plot_anomaly_scatter(X, labels, if_scores, feature_names)
    
    if labels is not None:
        plot_roc_curve(labels, if_scores, 'Isolation Forest')
        plot_roc_curve(labels, lof_scores, 'LOF')
        plot_roc_curve(labels, kmeans_scores, 'KMeans')
    
    # PCA for visualization
    pca_viz = PCA(n_components=2, random_state=42)
    plot_pca_projection(X, pca_viz.fit(X), labels, if_scores)
    
    plot_anomaly_distribution(scores, labels)
    
    print("✓ All visualizations saved")
    
    print("\n" + "="*80)
    print("CDR PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return {
        'models': models,
        'scores': scores,
        'metrics': metrics,
        'preprocessors': preprocessors,
        'feature_names': feature_names
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    print("="*80)
    print("TELECOM FRAUD DETECTION - UNSUPERVISED LEARNING")
    print("="*80)
    print("\nThis system performs unsupervised anomaly detection on IPDR and CDR datasets")
    print("="*80)
    
    # Configuration
    IPDR_FILE = 'ipdr.csv'  # Change to your IPDR file path
    CDR_FILE = 'cdrv2.csv'    # Change to your CDR file path
    
    # Check if files exist
    ipdr_exists = os.path.exists(IPDR_FILE)
    cdr_exists = os.path.exists(CDR_FILE)
    
    if not ipdr_exists and not cdr_exists:
        print("\n⚠️  WARNING: No data files found!")
        print(f"   Looking for: {IPDR_FILE} and/or {CDR_FILE}")
        print("\n" + "="*80)
        print("DEMO MODE: Creating synthetic datasets for demonstration")
        print("="*80)
        
        # Generate synthetic IPDR data
        print("\n📊 Generating synthetic IPDR dataset...")
        np.random.seed(42)
        n_samples = 5000
        n_fraud = int(n_samples * 0.05)  # 5% fraud
        
        ipdr_data = {
            'event_id': [f'EVT{i:06d}' for i in range(n_samples)],
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
            'imei': [f'IMEI{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
            'imsi': [f'IMSI{np.random.randint(100, 999)}' for _ in range(n_samples)],
            'domain': np.random.choice(['google.com', 'facebook.com', 'twitter.com', 
                                       'malicious.com', 'phishing.net'], n_samples),
            'protocol': np.random.choice(['HTTP', 'HTTPS', 'FTP', 'SMTP'], n_samples),
            'bytes_sent': np.random.exponential(5000, n_samples),
            'bytes_received': np.random.exponential(50000, n_samples),
            'cell_id': np.random.choice([f'CELL{i}' for i in range(1, 21)], n_samples),
            'is_fraud': [0] * n_samples
        }
        
        # Inject fraud patterns
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        for idx in fraud_indices:
            ipdr_data['is_fraud'][idx] = 1
            ipdr_data['bytes_sent'][idx] *= np.random.uniform(5, 15)  # Unusual traffic
            ipdr_data['bytes_received'][idx] *= np.random.uniform(5, 15)
            if np.random.rand() > 0.5:
                ipdr_data['domain'][idx] = 'malicious.com'
        
        ipdr_df = pd.DataFrame(ipdr_data)
        ipdr_df.to_csv('synthetic_ipdr_data.csv', index=False)
        print(f"✓ Created synthetic IPDR dataset: {n_samples} samples, {n_fraud} frauds")
        
        # Generate synthetic CDR data
        print("\n📊 Generating synthetic CDR dataset...")
        cdr_data = {
            'call_id': [f'CALL{i:06d}' for i in range(n_samples)],
            'call_start_time': pd.date_range('2024-01-01', periods=n_samples, freq='3min'),
            'imei': [f'IMEI{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
            'imsi': [f'IMSI{np.random.randint(100, 999)}' for _ in range(n_samples)],
            'duration': np.random.exponential(300, n_samples),  # seconds
            'bytes_sent': np.random.exponential(1000, n_samples),
            'bytes_received': np.random.exponential(5000, n_samples),
            'cell_id': np.random.choice([f'CELL{i}' for i in range(1, 21)], n_samples),
            'call_type': np.random.choice(['voice', 'data', 'sms'], n_samples),
            'is_fraud': [0] * n_samples
        }
        
        # Inject fraud patterns in CDR
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        for idx in fraud_indices:
            cdr_data['is_fraud'][idx] = 1
            cdr_data['duration'][idx] *= np.random.uniform(10, 30)  # Unusually long calls
            cdr_data['bytes_sent'][idx] *= np.random.uniform(20, 50)
            cdr_data['bytes_received'][idx] *= np.random.uniform(20, 50)
        
        cdr_df = pd.DataFrame(cdr_data)
        cdr_df.to_csv('synthetic_cdr_data.csv', index=False)
        print(f"✓ Created synthetic CDR dataset: {n_samples} samples, {n_fraud} frauds")
        
        IPDR_FILE = 'synthetic_ipdr_data.csv'
        CDR_FILE = 'synthetic_cdr_data.csv'
    
    # Process IPDR data
    if os.path.exists(IPDR_FILE):
        print("\n" + "="*80)
        print("📡 PROCESSING IPDR DATA")
        print("="*80)
        
        try:
            ipdr_df = pd.read_csv(IPDR_FILE)
            print(f"✓ Loaded IPDR data: {ipdr_df.shape[0]} rows, {ipdr_df.shape[1]} columns")
            
            # Display sample data
            print("\n📋 IPDR Data Preview:")
            print(ipdr_df.head())
            
            # Check for required columns
            print("\n📊 IPDR Columns:", ipdr_df.columns.tolist())
            
            if 'is_fraud' in ipdr_df.columns:
                fraud_rate = ipdr_df['is_fraud'].mean()
                print(f"\n⚠️  Fraud Rate: {fraud_rate:.2%} ({ipdr_df['is_fraud'].sum()} fraudulent records)")
            
            # Run IPDR pipeline
            print("\n🚀 Starting IPDR anomaly detection pipeline...")
            ipdr_results = run_ipdr_pipeline(ipdr_df, output_prefix='ipdr_unsupervised')
            
            # Summary of results
            print("\n" + "="*80)
            print("📊 IPDR RESULTS SUMMARY")
            print("="*80)
            
            if ipdr_results['metrics']:
                print("\n🎯 Model Performance Comparison:")
                print(f"{'Model':<25} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
                print("-" * 70)
                for model_name, metrics in ipdr_results['metrics'].items():
                    print(f"{model_name:<25} {metrics['auc']:<10.4f} "
                          f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1']:<10.4f}")
            
            print(f"\n✓ IPDR pipeline completed successfully")
            print(f"✓ Models saved to: ipdr_unsupervised_models.pkl")
            print(f"✓ Scores saved to: ipdr_unsupervised_scores.csv")
            
        except Exception as e:
            print(f"\n❌ Error processing IPDR data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Process CDR data
    if os.path.exists(CDR_FILE):
        print("\n" + "="*80)
        print("📞 PROCESSING CDR DATA")
        print("="*80)
        
        try:
            cdr_df = pd.read_csv(CDR_FILE)
            print(f"✓ Loaded CDR data: {cdr_df.shape[0]} rows, {cdr_df.shape[1]} columns")
            
            # Display sample data
            print("\n📋 CDR Data Preview:")
            print(cdr_df.head())
            
            # Check for required columns
            print("\n📊 CDR Columns:", cdr_df.columns.tolist())
            
            if 'is_fraud' in cdr_df.columns:
                fraud_rate = cdr_df['is_fraud'].mean()
                print(f"\n⚠️  Fraud Rate: {fraud_rate:.2%} ({cdr_df['is_fraud'].sum()} fraudulent records)")
            
            # Run CDR pipeline
            print("\n🚀 Starting CDR anomaly detection pipeline...")
            cdr_results = run_cdr_pipeline(cdr_df, output_prefix='cdr_unsupervised')
            
            # Summary of results
            print("\n" + "="*80)
            print("📊 CDR RESULTS SUMMARY")
            print("="*80)
            
            if cdr_results['metrics']:
                print("\n🎯 Model Performance Comparison:")
                print(f"{'Model':<25} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
                print("-" * 70)
                for model_name, metrics in cdr_results['metrics'].items():
                    print(f"{model_name:<25} {metrics['auc']:<10.4f} "
                          f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1']:<10.4f}")
            
            print(f"\n✓ CDR pipeline completed successfully")
            print(f"✓ Models saved to: cdr_unsupervised_models.pkl")
            print(f"✓ Scores saved to: cdr_unsupervised_scores.csv")
            
        except Exception as e:
            print(f"\n❌ Error processing CDR data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL PIPELINES COMPLETED")
    print("="*80)
    
    print("\n📦 Generated Files:")
    output_files = [
        'ipdr_unsupervised_scores.csv',
        'ipdr_unsupervised_models.pkl',
        'cdr_unsupervised_scores.csv',
        'cdr_unsupervised_models.pkl'
    ]
    
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✓ {file:<35} ({size:.2f} KB)")
    
    print("\n📊 Generated Visualizations:")
    viz_files = [
        'ipdr_feature_correlation.png',
        'cdr_feature_correlation.png',
        'feature_distributions.png',
        'anomaly_scatter.png',
        'roc_curve_isolation_forest.png',
        'roc_curve_lof.png',
        'pca_projection.png',
        'anomaly_score_distributions.png'
    ]
    
    for file in viz_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
    
    print("\n" + "="*80)
    print("🎓 USAGE TIPS")
    print("="*80)
    print("""
    1. Load saved models for inference:
       with open('ipdr_unsupervised_models.pkl', 'rb') as f:
           saved = pickle.load(f)
           models = saved['models']
           preprocessors = saved['preprocessors']
    
    2. Score new data:
       # Preprocess new data using saved preprocessors
       X_new_scaled = preprocessors['scaler'].transform(X_new)
       # Get anomaly scores
       scores = models['isolation_forest'].score_samples(X_new_scaled)
    
    3. Adjust contamination parameter:
       # In the pipeline functions, modify:
       train_isolation_forest(X, contamination=0.05)  # 5% expected fraud
    
    4. Customize visualizations:
       # Modify plot functions or add your own
       
    5. Feature importance:
       # Check PCA explained variance for feature reduction
       pca_model = models['pca']
       print(pca_model.explained_variance_ratio_)
    """)
    
    print("\n" + "="*80)
    print("Thank you for using the Telecom Fraud Detection System! 🚀")
    print("="*80 + "\n")