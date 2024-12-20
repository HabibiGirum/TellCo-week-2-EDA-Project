import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class UserEngagementAnalysis:
    def __init__(self, data_path):
        """
        Initialize the class with the dataset path.
        """
        self.data = pd.read_csv(data_path)
        self.aggregated_data = None
        self.normalized_data = None
        self.cluster_stats = None
        self.kmeans = None

    def aggregate_metrics(self):
        """
        Aggregate engagement metrics per customer.
        """
        self.aggregated_data = self.data.groupby("MSISDN/Number").agg({
            "Dur. (ms)": "sum",
            "Total UL (Bytes)": "sum",
            "Total DL (Bytes)": "sum",
            "Bearer Id": "count"
        }).rename(columns={
            "Dur. (ms)": "Total Duration (ms)",
            "Total DL (Bytes)": "Total Download Traffic (Bytes)",
            "Total UL (Bytes)": "Total Upload Traffic (Bytes)",
            "Bearer Id": "Session Frequency"
        })

        # # Convert duration from ms to seconds
        # self.aggregated_data["Total Duration (ms)"] = self.aggregated_data["Total Duration (ms)"] / 1000
        # self.aggregated_data.drop(columns=["Total Duration (ms)"], inplace=True)

    def top_customers(self, metric, top_n=10):
        """
        Get top N customers based on a specified metric.
        """
        return self.aggregated_data.nlargest(top_n, metric)

    def normalize_and_cluster(self, k=3):
        """
        Normalize data and perform K-means clustering.
        """
        scaler = MinMaxScaler()
        self.normalized_data = scaler.fit_transform(self.aggregated_data)
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        self.aggregated_data["Cluster"] = self.kmeans.fit_predict(self.normalized_data)

    def compute_cluster_statistics(self):
        """
        Compute statistics for each cluster.
        """
        self.cluster_stats = self.aggregated_data.groupby("Cluster").agg({
            "Total Duration (ms)": ["min", "max", "mean", "sum"],
            "Total Download Traffic (Bytes)": ["min", "max", "mean", "sum"],
            "Total Upload Traffic (Bytes)": ["min", "max", "mean", "sum"],
            "Session Frequency": ["min", "max", "mean", "sum"]
        })
        return self.cluster_stats

    def determine_optimal_k(self, max_k=10):
        """
        Determine the optimal number of clusters using the elbow method.
        """
        inertia = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.normalized_data)
            inertia.append(kmeans.inertia_)

        # Plot elbow method
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), inertia, marker='o')
        plt.title("Elbow Method for Optimal k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.savefig('determine_optimal_k.png')
        plt.show()

    def analyze_application_usage(self):
        """
        Analyze application usage metrics per user.
        """
        app_columns = [
            "Social Media DL (Bytes)", "Social Media UL (Bytes)",
            "Youtube DL (Bytes)", "Youtube UL (Bytes)",
            "Netflix DL (Bytes)", "Netflix UL (Bytes)",
            "Google DL (Bytes)", "Google UL (Bytes)",
            "Email DL (Bytes)", "Email UL (Bytes)",
            "Gaming DL (Bytes)", "Gaming UL (Bytes)"
        ]
        self.data["Total Traffic"] = self.data[app_columns].sum(axis=1)
        user_app_traffic = self.data.groupby("MSISDN/Number")[app_columns].sum()
        return user_app_traffic

    def plot_top_applications(self, user_app_traffic, top_n=3):
        """
        Plot the top  applications based on total user engagement.
        """
        top_apps = user_app_traffic.sum().nlargest(top_n).index
        top_apps_data = user_app_traffic[top_apps]

        top_apps_data.sum().plot(kind="bar", figsize=(8, 6), color=["blue", "green", "orange"])
        plt.title("Top Applications by User Engagement")
        plt.xlabel("Applications")
        plt.ylabel("Total Traffic (Bytes)")
        plt.xticks(rotation=45)
        plt.savefig('top_application.png')
        plt.show()

