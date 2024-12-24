import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ExperienceAnalytics:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return pd.read_csv(self.dataset_path)

    # Task 3.1 - Aggregate customer information
    def aggregate_per_customer(self):
        df = self.dataset.copy()
        
        # Handle missing values with mean for numeric columns and mode for categorical
        df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
        df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
        df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
        df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
        
        # Remove outliers (using IQR method)
        for col in ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
        
        agg_data = df.groupby('IMSI').agg(
            avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
            avg_rtt=('Avg RTT DL (ms)', 'mean'),
            handset_type=('Handset Type', 'first'),
            avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
        ).reset_index()
        
        return agg_data
    
    # Task 3.2 - Top, Bottom, and Frequent TCP, RTT, Throughput values
    def top_bottom_frequent_values(self):
        results = {}
        for col in ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']:
            results[col] = {
                'Top 10': self.dataset[col].nlargest(10).values,
                'Bottom 10': self.dataset[col].nsmallest(10).values,
                'Most Frequent': self.dataset[col].mode().values
            }
        return results
    
    # Task 3.3 - Distribution per Handset Type
    def distribution_per_handset(self):
        df = self.dataset.copy()
        throughput_by_handset = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False)
        tcp_by_handset = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)
        
        # Plot Throughput
        throughput_by_handset.head(10).plot(kind='bar', title='Top 10 Handsets by Average Throughput')
        plt.show()
        
        # Plot TCP Retransmission
        tcp_by_handset.head(10).plot(kind='bar', title='Top 10 Handsets by Average TCP Retransmission')
        plt.show()
        
        return throughput_by_handset, tcp_by_handset
    
    



    def kmeans_clustering(self, k=3):
        df = self.aggregate_per_customer()
        
        # Ensure numeric columns
        numeric_features = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
        for feature in numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Handle missing values (e.g., fill with mean)
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_features])
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # Map cluster numbers to meaningful names
        cluster_names = {
            0: 'Low Engagement',
            1: 'Moderate Engagement',
            2: 'High Engagement'
        }
        df['Cluster_Name'] = df['Cluster'].map(cluster_names)
        
        # Visualize Clusters
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=df['avg_throughput'],
            y=df['avg_rtt'],
            hue=df['Cluster_Name'],
            palette='viridis',
            s=50,
            alpha=0.6
        )
        plt.title('K-Means Clustering of User Experience')
        plt.xlabel('avg_throughput')
        plt.ylabel('avg_rtt')
        plt.legend(title='Cluster')
        plt.show()
        
        # Cluster summary with meaningful names
        cluster_summary = df.groupby('Cluster_Name')[numeric_features].mean().reset_index()
        return cluster_summary


