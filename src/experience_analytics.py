import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px


# -------------------------------
# ExperienceAnalytics Class
# -------------------------------
class ExperienceAnalytics:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return pd.read_csv(self.dataset_path)

    def aggregate_per_customer(self):
        df = self.dataset.copy()
        
        df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
        df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
        df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
        df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
        
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
    
    def top_bottom_frequent_values(self):
        results = {}
        for col in ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']:
            results[col] = {
                'Top 10': self.dataset[col].nlargest(10).values,
                'Bottom 10': self.dataset[col].nsmallest(10).values,
                'Most Frequent': self.dataset[col].mode().values
            }
        return results
    
    def distribution_per_handset(self):
        df = self.dataset.copy()
        throughput_by_handset = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False)
        tcp_by_handset = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)
        
        throughput_by_handset.head(10).plot(kind='bar', title='Top 10 Handsets by Average Throughput')
        plt.show()
        
        tcp_by_handset.head(10).plot(kind='bar', title='Top 10 Handsets by Average TCP Retransmission')
        plt.show()
        
        return throughput_by_handset, tcp_by_handset
    
    def kmeans_clustering(self, k=3):
        df = self.aggregate_per_customer()
        numeric_features = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_features])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        cluster_names = {0: 'Low Engagement', 1: 'Moderate Engagement', 2: 'High Engagement'}
        df['Cluster_Name'] = df['Cluster'].map(cluster_names)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=df['avg_throughput'], y=df['avg_rtt'], hue=df['Cluster_Name'],
            palette='viridis', s=50, alpha=0.6
        )
        plt.title('K-Means Clustering of User Experience')
        plt.xlabel('avg_throughput')
        plt.ylabel('avg_rtt')
        plt.legend(title='Cluster')
        plt.show()
        
        cluster_summary = df.groupby('Cluster_Name')[numeric_features].mean().reset_index()
        return cluster_summary


# -------------------------------
# Streamlit Experience Analysis
# -------------------------------
def experience_analysis_page():
    st.title("📊 Experience Analysis")
    
    # Initialize Analysis Class
    analysis = ExperienceAnalytics('Data/Copy of Week2_challenge_data_source(CSV).csv')
    df = analysis.dataset
    
    # Plot 1: Average Download Speed by Region
    st.subheader("Average Download Speed by Region")
    if 'Region' in df.columns and 'Avg Download Speed' in df.columns:
        fig = px.bar(df, x='Region', y='Avg Download Speed', color='Region')
        st.plotly_chart(fig)
    else:
        st.warning("Required columns for this plot are missing.")
    
    # Plot 2: Latency Distribution
    st.subheader("Latency Distribution")
    if 'Latency' in df.columns:
        fig = px.histogram(df, x='Latency')
        st.plotly_chart(fig)
    else:
        st.warning("Required column for this plot is missing.")
    
    # Aggregated Data
    st.subheader("Aggregated Customer Data")
    agg_data = analysis.aggregate_per_customer()
    st.write(agg_data.head())
    
    # Clustering
    st.subheader("K-Means Clustering")
    cluster_summary = analysis.kmeans_clustering()
    st.write(cluster_summary)
    
    # Top 10 Handsets by Average Throughput
    st.subheader("Top 10 Handsets by Average Throughput")
    throughput, _ = analysis.distribution_per_handset()
    st.bar_chart(throughput.head(10))


# -------------------------------
# Streamlit App Entry Point
# -------------------------------
if __name__ == "__main__":
    experience_analysis_page()
