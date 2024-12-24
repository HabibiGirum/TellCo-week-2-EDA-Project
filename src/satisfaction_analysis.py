import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from urllib.parse import quote_plus
from sklearn.metrics import mean_squared_error  

# Assuming the ExperienceAnalytics class is implemented as mentioned earlier
from scripts.experience_analytics import ExperienceAnalytics

class SatisfactionAnalysis:
    def __init__(self, dataset_path):
        self.analytics = ExperienceAnalytics(dataset_path)

    def compute_satisfaction_scores(self):
        df = self.analytics.aggregate_per_customer()
        numeric_features = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_features])

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        cluster_centroids = kmeans.cluster_centers_

        least_engaged_centroid = cluster_centroids[0]
        worst_experience_centroid = cluster_centroids[0]

        df['Engagement_Score'] = df[numeric_features].apply(
            lambda row: euclidean(row, least_engaged_centroid), axis=1
        )
        df['Experience_Score'] = df[numeric_features].apply(
            lambda row: euclidean(row, worst_experience_centroid), axis=1
        )

        return df

    def top_satisfied_customers(self):
        df = self.compute_satisfaction_scores()
        df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2
        return df.nlargest(10, 'Satisfaction_Score')[['IMSI', 'Engagement_Score', 'Experience_Score', 'Satisfaction_Score']]

    def predict_satisfaction_score(self):
        df = self.compute_satisfaction_scores()
        df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

        # Define features and target variable
        X = df[['Engagement_Score', 'Experience_Score']]
        y = df['Satisfaction_Score']

        model = LinearRegression()
        model.fit(X, y)
        mse = mean_squared_error(y, model.predict(X))

        return model, mse

    def export_to_postgresql(self, db_name, table_name, user, password, host='localhost', port=5432):
        try:
            df = self.compute_satisfaction_scores()
            df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

            password = quote_plus(password)
            connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'
            engine = create_engine(connection_string)

            df[['IMSI', 'Engagement_Score', 'Experience_Score', 'Satisfaction_Score']].to_sql(
                table_name, con=engine, if_exists='replace', index=False
            )
            st.success("Data successfully exported to PostgreSQL.")
        except OperationalError as e:
            st.error(f"OperationalError: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Streamlit App Interface
def satisfaction_analysis_page():
    st.title("Satisfaction Analysis")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file is not None:
        analysis = SatisfactionAnalysis(uploaded_file)

        # Sidebar Navigation
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis option",
            [
                "Top Satisfied Customers",
                "Predict Satisfaction Score",
                "Export to PostgreSQL"
            ]
        )

        if analysis_option == "Top Satisfied Customers":
            st.write("### Top 10 Satisfied Customers")
            top_customers = analysis.top_satisfied_customers()
            st.write(top_customers)

        elif analysis_option == "Predict Satisfaction Score":
            st.write("### Predict Satisfaction Score")
            model, mse = analysis.predict_satisfaction_score()
            st.write(f"Mean Squared Error: {mse}")
            st.write("Model Coefficients: ", model.coef_)

        elif analysis_option == "Export to PostgreSQL":
            st.write("### Export Satisfaction Analysis Results to PostgreSQL")
            db_name = st.text_input("Enter Database Name")
            table_name = st.text_input("Enter Table Name")
            user = st.text_input("Enter Username")
            password = st.text_input("Enter Password", type="password")
            host = st.text_input("Enter Host", value="localhost")
            port = st.text_input("Enter Port", value="5432")

            if st.button("Export Data"):
                analysis.export_to_postgresql(db_name, table_name, user, password, host, port)

# Run the Streamlit app
if __name__ == '__main__':
    satisfaction_analysis_page()
