import streamlit as st
import pandas as pd
from scripts.user_engagement_analysis import UserEngagementAnalysis

# Streamlit App Interface
def user_engagement_page():
    st.title("📊 User Engagement Analysis")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file is not None:
        # Initialize the analysis
        analysis = UserEngagementAnalysis(uploaded_file)

        # Display Dataset
        st.write("### Dataset Preview")
        st.dataframe(analysis.data.head())  # Display first 5 rows of the dataset

        # Automatically call the analysis steps in sequence without needing a dropdown
        # 1. Aggregate Engagement Metrics
        st.write("### Aggregated Data Per Customer")
        analysis.aggregate_metrics()
        st.write(analysis.aggregated_data.head())

        # 2. Top Customers
        st.write("### Top Customers Based on Metric")
        if analysis.aggregated_data is None:
            st.error("Please call 'Aggregate Engagement Metrics' first!")
        else:
            metric = st.selectbox("Select Metric", ["Total Duration (ms)", "Total Download Traffic (Bytes)", "Total Upload Traffic (Bytes)", "Session Frequency"])
            top_n = st.slider("Top N Customers", 1, 20, 10)
            try:
                top_customers = analysis.top_customers(metric, top_n)
                st.write(top_customers)
            except ValueError as e:
                st.error(str(e))

        # 3. K-Means Clustering
        st.write("### K-Means Clustering")
        try:
            analysis.normalize_and_cluster(k=3)
            st.write(analysis.aggregated_data.head())
        except ValueError as e:
            st.error(str(e))

        # 4. Cluster Statistics
        st.write("### Cluster Statistics")
        try:
            cluster_stats = analysis.compute_cluster_statistics()
            st.write(cluster_stats)
        except ValueError as e:
            st.error(str(e))

        # 5. Determine Optimal k
        st.write("### Determine Optimal k using Elbow Method")
        try:
            fig = analysis.determine_optimal_k()
            st.pyplot(fig)  # Display the Elbow plot
        except ValueError as e:
            st.error(str(e))

        # 6. Application Usage
        st.write("### Application Usage Per User")
        app_usage = analysis.analyze_application_usage()
        st.write(app_usage.head())

        # Plot top applications
        st.write("### Top Applications by User Engagement")
        try:
            fig = analysis.plot_top_applications(app_usage)
            st.pyplot(fig)  # Display the top applications bar plot
        except ValueError as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    user_engagement_page()
