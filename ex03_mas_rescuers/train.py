import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Constants
N_SPLITS = 5
RANDOM_STATE = 42


def load_vitals(file_path):
    """Load data from the txt file."""
    data = pd.read_csv(file_path, delimiter=",", header=None)
    data.columns = [
        "Id",
        "pSist",
        "pDiast",
        "qPA",
        "pulso",
        "freq_resp",
        "gravidade",
        "classe",
    ]
    return data


def load_victims(file_path):
    """Load data from the txt file."""
    data = pd.read_csv(file_path, delimiter=",", header=None)
    data.columns = [
        "x",
        "y",
    ]
    data["Id"] = data.index
    return data


def preprocess_data(data):
    """Preprocess the data for training."""
    # Features for both classifier and regressor
    # Using qPA, pulso, frequência respiratória as features
    X = data[["qPA", "pulso", "freq_resp"]]

    # Target for classifier (classes 1-4)
    y_class = data["classe"].astype("category")

    print("Unique classes in y_class:", y_class.unique())  # Check unique values

    # Target for regressor (gravidade)
    y_reg = data["gravidade"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_class, y_reg, scaler


def evaluate_models(classifier, regressor, X_test, y_class_test, y_reg_test):
    """Evaluate models on test data."""
    # Classifier evaluation
    y_class_pred = classifier.predict(X_test)
    class_accuracy = accuracy_score(y_class_test, y_class_pred)
    class_report = classification_report(y_class_test, y_class_pred)

    # Regressor evaluation
    y_reg_pred = regressor.predict(X_test)
    reg_mse = mean_squared_error(y_reg_test, y_reg_pred)

    print("\nTest Set Evaluation:")
    print(f"Classifier Accuracy: {class_accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print(f"Regressor MSE: {reg_mse:.4f}")


def train_and_evaluate_models():
    """Train and evaluate different models using cross-validation."""
    # Load training data
    data = load_vitals("datasets/data_300v_90x90/env_vital_signals.txt")
    X, y_class, y_reg, scaler = preprocess_data(data)

    # Split the data into training (75%) and testing (25%)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_class, y_reg, test_size=0.25, random_state=RANDOM_STATE)
    )

    y_class_train = y_class_train.astype("category")
    y_class_test = y_class_test.astype("category")

    # Initialize K-Fold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Model configurations
    cart_class_configs = [{"max_depth": 5}, {"max_depth": 10}, {"max_depth": None}]

    nn_class_configs = [
        {"hidden_layer_sizes": (50,), "max_iter": 1000},
        {"hidden_layer_sizes": (100, 50), "max_iter": 1000},
        {"hidden_layer_sizes": (100, 100, 50), "max_iter": 1000},
    ]

    cart_reg_configs = [{"max_depth": 5}, {"max_depth": 10}, {"max_depth": None}]

    nn_reg_configs = [
        {"hidden_layer_sizes": (50,), "max_iter": 1000},
        {"hidden_layer_sizes": (100, 50), "max_iter": 1000},
        {"hidden_layer_sizes": (100, 100, 50), "max_iter": 1000},
    ]

    # Results storage
    best_class_model = None
    best_class_score = 0
    best_reg_model = None
    best_reg_score = float("inf")

    # Train and evaluate classifiers
    print("\nTraining Classifiers:")

    # CART Classifier
    for config in cart_class_configs:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            clf = DecisionTreeClassifier(**config, random_state=RANDOM_STATE)
            clf.fit(X_train[train_idx], y_class_train.iloc[train_idx])
            score = accuracy_score(
                y_class_train.iloc[val_idx], clf.predict(X_train[val_idx])
            )
            scores.append(score)
        avg_score = np.mean(scores)
        print(f"CART Config {config}: Accuracy = {avg_score:.4f}")

        if avg_score > best_class_score:
            best_class_score = avg_score
            best_class_model = DecisionTreeClassifier(
                **config, random_state=RANDOM_STATE
            )

    # Neural Network Classifier
    for config in nn_class_configs:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            clf = MLPClassifier(**config, random_state=RANDOM_STATE)
            clf.fit(X_train[train_idx], y_class_train.iloc[train_idx])
            score = accuracy_score(
                y_class_train.iloc[val_idx], clf.predict(X_train[val_idx])
            )
            scores.append(score)
        avg_score = np.mean(scores)
        print(f"NN Config {config}: Accuracy = {avg_score:.4f}")

        if avg_score > best_class_score:
            best_class_score = avg_score
            best_class_model = MLPClassifier(**config, random_state=RANDOM_STATE)

    # Train and evaluate regressors
    print("\nTraining Regressors:")

    # CART Regressor
    for config in cart_reg_configs:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            reg = DecisionTreeRegressor(**config, random_state=RANDOM_STATE)
            reg.fit(X_train[train_idx], y_reg_train.iloc[train_idx])
            score = mean_squared_error(
                y_reg_train.iloc[val_idx], reg.predict(X_train[val_idx])
            )
            scores.append(score)
        avg_score = np.mean(scores)
        print(f"CART Config {config}: MSE = {avg_score:.4f}")

        if avg_score < best_reg_score:
            best_reg_score = avg_score
            best_reg_model = DecisionTreeRegressor(**config, random_state=RANDOM_STATE)

    # Neural Network Regressor
    for config in nn_reg_configs:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            reg = MLPRegressor(**config, random_state=RANDOM_STATE)
            reg.fit(X_train[train_idx], y_reg_train.iloc[train_idx])
            score = mean_squared_error(
                y_reg_train.iloc[val_idx], reg.predict(X_train[val_idx])
            )
            scores.append(score)
        avg_score = np.mean(scores)
        print(f"NN Config {config}: MSE = {avg_score:.4f}")

        if avg_score < best_reg_score:
            best_reg_score = avg_score
            best_reg_model = MLPRegressor(**config, random_state=RANDOM_STATE)

    # Train final models on full training data
    best_class_model.fit(X_train, y_class_train)
    best_reg_model.fit(X_train, y_reg_train)

    # Evaluate models on test set
    evaluate_models(best_class_model, best_reg_model, X_test, y_class_test, y_reg_test)

    # Save models and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_class_model, "models/best_classifier.joblib")
    joblib.dump(best_reg_model, "models/best_regressor.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    return best_class_model, best_reg_model, scaler


def plot_clusters(df, save_path="clusters/cluster_visualization.png"):
    """Plot the clusters with victims colored by cluster and markers by severity class."""
    plt.figure(figsize=(12, 8))

    # Create color palette for clusters
    cluster_colors = sns.color_palette("husl", n_colors=df["cluster"].nunique())

    # Create marker styles for different classes
    markers = {
        1: "X",
        2: "s",
        3: "^",
        4: "o",
    }  # X for critical, square for unstable, triangle for pot. stable, circle for stable
    marker_sizes = {
        1: 150,
        2: 100,
        3: 80,
        4: 60,
    }  # Larger markers for more critical victims

    # Plot each class separately for proper legend
    for classe in sorted(df["classe"].unique()):
        for cluster in sorted(df["cluster"].unique()):
            mask = (df["classe"] == classe) & (df["cluster"] == cluster)
            plt.scatter(
                df[mask]["x"],
                df[mask]["y"],
                c=[cluster_colors[cluster - 1]],  # -1 because clusters are 1-based
                marker=markers[classe],
                s=marker_sizes[classe],
                alpha=0.7,
                label=(
                    f"Class {classe} - Cluster {cluster}" if classe == 1 else None
                ),  # Only add cluster to legend once
            )

    # Add legend
    class_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[classe],
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label=f"Class {classe}",
        )
        for classe in sorted(df["classe"].unique())
    ]

    cluster_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=f"Cluster {i+1}",
        )
        for i, color in enumerate(cluster_colors)
    ]

    # Create two legends
    plt.legend(handles=cluster_legend_elements, loc="upper left", title="Clusters")
    plt.legend(
        handles=class_legend_elements, loc="upper right", title="Severity Classes"
    )

    plt.title("Victim Clusters and Severity Classes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_clusters(env_victims, env_vitals, n_clusters=4):
    """Create cluster files based on victim locations, gravity, and class."""
    # Merge victim locations with vital signs
    victims_df = pd.DataFrame(env_victims)
    vitals_df = pd.DataFrame(env_vitals)
    df = pd.merge(victims_df, vitals_df, on="Id")

    # Load models
    classifier = joblib.load("models/best_classifier.joblib")
    regressor = joblib.load("models/best_regressor.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Prepare features for prediction
    features = scaler.transform(df[["qPA", "pulso", "freq_resp"]])

    # Make predictions
    df["grav"] = regressor.predict(features)
    df["classe"] = classifier.predict(features)

    # Create clusters directory if it doesn't exist
    os.makedirs("clusters", exist_ok=True)

    # Prepare data for clustering
    # Scale features for clustering
    feature_scaler = MinMaxScaler()

    # Scale spatial features (x, y)
    spatial_features = feature_scaler.fit_transform(df[["x", "y"]])

    # Scale gravity (already predicted)
    gravity_scaled = feature_scaler.fit_transform(df[["grav"]])

    # Convert class to numeric importance (inverse of class number since class 1 is most critical)
    class_importance = 5 - df["classe"].values.reshape(
        -1, 1
    )  # Class 1 (critical) becomes 4, Class 4 (stable) becomes 1
    class_scaled = feature_scaler.fit_transform(class_importance)

    # Combine features with weights
    # Higher weights for gravity and class to prioritize severity
    # Format: [x, y, gravity * weight, class * weight]
    GRAVITY_WEIGHT = 1.5
    CLASS_WEIGHT = 1.2

    clustering_features = np.hstack(
        [
            spatial_features,  # x, y coordinates (weight 1.0)
            gravity_scaled * GRAVITY_WEIGHT,  # gravity with higher weight
            class_scaled * CLASS_WEIGHT,  # class importance with higher weight
        ]
    )

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(clustering_features)

    # Calculate cluster priorities based on average gravity and class
    cluster_priorities = []
    for i in range(n_clusters):
        cluster_data = df[df["cluster"] == i]
        avg_gravity = cluster_data["grav"].mean()
        avg_class = cluster_data["classe"].mean()
        # Priority score: higher gravity and lower class number (more severe) means higher priority
        priority = avg_gravity * (5 - avg_class)
        cluster_priorities.append((i, priority))

    # Sort clusters by priority
    cluster_priorities.sort(key=lambda x: x[1], reverse=True)

    # Reassign cluster numbers based on priority (highest priority = cluster 1)
    cluster_map = {
        old_cluster: new_cluster + 1
        for new_cluster, (old_cluster, _) in enumerate(cluster_priorities)
    }
    df["cluster"] = df["cluster"].map(cluster_map)

    # Print cluster statistics
    print("\nCluster Statistics:")
    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster"] == i]
        print(f"\nCluster {i}:")
        print(f"Number of victims: {len(cluster_data)}")
        print(f"Average gravity: {cluster_data['grav'].mean():.2f}")
        print(f"Average class: {cluster_data['classe'].mean():.2f}")
        print("Class distribution:")
        print(cluster_data["classe"].value_counts().sort_index())

    # Save cluster files
    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster"] == i]
        cluster_data[["Id", "x", "y", "grav", "classe"]].to_csv(
            f"clusters/cluster{i}.txt", index=False, header=False
        )

    # Plot the clusters
    plot_clusters(df)

    return df  # Return the dataframe with cluster assignments for potential further analysis


if __name__ == "__main__":
    # Train models
    # best_classifier, best_regressor, scaler = train_and_evaluate_models()
    # print("\nModels have been trained and saved in the 'models' directory.")

    # Create clusters
    create_clusters(
        load_victims("datasets/data_300v_90x90/env_victims.txt"),
        load_vitals("datasets/data_300v_90x90/env_vital_signals.txt"),
        n_clusters=4,
    )
