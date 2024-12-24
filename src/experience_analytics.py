import streamlit as st
import plotly.express as px
from scripts.experience_analytics import ExperienceAnalytics
def experience_analytics_page():
    st.title("📈 Experience Analytics")
    st.write("After uploading your dataset, use the sidebar to explore detailed insights about user experience, including various analytical.")

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
