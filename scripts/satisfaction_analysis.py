from scripts.experience_analytics import ExperienceAnalytics
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


class SatisfactionAnalysis:
    def __init__(self, dataset_path):
        self.analytics = ExperienceAnalytics(dataset_path)

    def compute_satisfaction_scores(self):
        df = self.analytics.aggregate_per_customer()
        numeric_features = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_features])
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        cluster_centroids = kmeans.cluster_centers_
        
        # Define centroids for least engagement and worst experience clusters
        least_engaged_centroid = cluster_centroids[0]
        worst_experience_centroid = cluster_centroids[0]
        
        # Compute Engagement and Experience Scores
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
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        mse = mean_squared_error(y, model.predict(X))
        
        return model, mse

    


    def export_to_postgresql(self, db_name, table_name, user, password, host='localhost', port=5432):
        """
        Export satisfaction analysis results to a PostgreSQL database.
        """
        try:
            df = self.compute_satisfaction_scores()
            df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

            # Ensure special characters in the password are URL-encoded
            from urllib.parse import quote_plus
            password = quote_plus(password)

            # Correct PostgreSQL connection string
            connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'
            engine = create_engine(connection_string)

            # Export data to PostgreSQL
            df[['IMSI', 'Engagement_Score', 'Experience_Score', 'Satisfaction_Score']].to_sql(
                table_name, con=engine, if_exists='replace', index=False
            )
            print("✅ Data successfully exported to PostgreSQL.")

        except OperationalError as e:
            print(f"❌ OperationalError: {e}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")


