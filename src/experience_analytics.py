import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ExperienceAnalytics:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return pd.read_csv(self.dataset_path)

    def aggregate_per_customer(self):
        df = self.dataset.copy()
        
        # Handle missing values
        df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
        df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
        df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
        df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
        
        # Handle outliers
        for col in ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
        
        # Aggregate per customer
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
        
        return df, kmeans

# Streamlit App Interface
def experience_analytics_page():
    st.title("📈 Experience Analytics")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file is not None:
        analysis = ExperienceAnalytics(uploaded_file)

        # Sidebar Navigation
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis option",
            [
                "Aggregate Per Customer",
                "Top/Bottom/Frequent Values",
                "Distribution Per Handset",
                "K-Means Clustering"
            ]
        )

        if analysis_option == "Aggregate Per Customer":
            agg_data = analysis.aggregate_per_customer()
            st.write("### Aggregate Data Per Customer")
            st.write(agg_data.head())

        elif analysis_option == "Top/Bottom/Frequent Values":
            st.write("### Top, Bottom, and Most Frequent Values")
            results = analysis.top_bottom_frequent_values()
            for col, data in results.items():
                st.write(f"#### {col}")
                st.write("Top 10 Values:", data['Top 10'])
                st.write("Bottom 10 Values:", data['Bottom 10'])
                st.write("Most Frequent Values:", data['Most Frequent'])

        elif analysis_option == "Distribution Per Handset":
            st.write("### Distribution Per Handset")
            throughput_by_handset, tcp_by_handset = analysis.distribution_per_handset()

            st.write("#### Top 10 Handsets by Average Throughput")
            fig1 = px.bar(
                x=throughput_by_handset.head(10).values,
                y=throughput_by_handset.head(10).index,
                orientation='h',
                title="Top 10 Handsets by Average Throughput",
                labels={"x": "Throughput", "y": "Handset Type"}
            )
            st.plotly_chart(fig1)

            st.write("#### Top 10 Handsets by Average TCP Retransmission")
            fig2 = px.bar(
                x=tcp_by_handset.head(10).values,
                y=tcp_by_handset.head(10).index,
                orientation='h',
                title="Top 10 Handsets by Average TCP Retransmission",
                labels={"x": "TCP Retransmission", "y": "Handset Type"}
            )
            st.plotly_chart(fig2)

        elif analysis_option == "K-Means Clustering":
            st.write("### K-Means Clustering")
            clusters, kmeans = analysis.kmeans_clustering()

            st.write("#### Cluster Summary")
            cluster_summary = clusters.groupby('Cluster_Name')[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']].mean()
            st.write(cluster_summary)

            st.write("#### Cluster Visualization")
            fig = px.scatter(
                clusters, x='avg_throughput', y='avg_rtt', color='Cluster_Name',
                title="K-Means Clustering of User Experience",
                labels={"avg_throughput": "Average Throughput", "avg_rtt": "Average RTT"},
                hover_data=['avg_tcp_retransmission']
            )
            st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == '__main__':
    experience_analytics_page()
