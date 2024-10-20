"""
This module performs K-Means clustering on Amazon product reviews and visualizes
the clusters using PCA and scatter plots. It also generates an elbow diagram to
identify the optimal number of clusters. 
"""

# Import Libraries
import os
import warnings
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
import matplotlib.pyplot as plt

# Environment Configuration
os.environ["WANDB_DISABLED"] = "true"

# Warnings Configuration
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

# Set device for torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
df = pd.read_csv(r"dataset/1429_1.csv")
df = df.dropna(subset=["reviews.rating"])

# Define function to label sentiment based on rating
def label_sentiment(rating):
    """
    Classifies sentiment based on the rating.
    1-2: Negative, 3: Neutral, 4-5: Positive.
    """
    if rating in [1, 2]:
        return 0  # Negative
    if rating == 3:
        return 1  # Neutral
    return 2  # Positive

# Apply sentiment labeling to the dataset
df["label"] = df["reviews.rating"].apply(label_sentiment)

# Drop rows with missing 'reviews.text'
df = df.dropna(subset=["reviews.text"])

# Drop unused columns
df = df.drop(
    [
        "reviews.dateAdded",
        "reviews.didPurchase",
        "reviews.id",
        "reviews.userCity",
        "reviews.userProvince",
    ],
    axis=1,
)

# Combine 'name' and 'categories' fields for feature extraction
df["combined"] = df["name"] + " " + df["categories"]

# Use TfidfVectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["combined"])

# Apply K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df["category_label"] = kmeans.fit_predict(X)

# Display value counts for clusters
print(df["category_label"].value_counts())

# Save categorized dataset to a CSV file
df.to_csv("categorized_dataset_k5.csv", index=False)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())  # Convert sparse matrix to dense array for PCA

# Define cluster names
cluster_names = {
    0: "Fire Tablet 7",
    1: "Kindle",
    2: "Speakers/Streaming",
    3: "Fire HD 8",
    4: "Fire KIDS",
}

# Add cluster names to DataFrame
df["category_name"] = df["category_label"].map(cluster_names)

# Scatter plot for clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=df["category_label"], cmap="viridis", alpha=0.6
)
plt.legend(handles=scatter.legend_elements()[0], labels=list(cluster_names.values()))
plt.title("K-Means Clustering of Product Categories")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Save the DataFrame with new category names to CSV
df.to_csv("categorized_dataset_k5_with_names.csv", index=False)

# Elbow diagram to find the optimal number of clusters
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Diagram
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, "bo-", color="b")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Diagram")
plt.show()

