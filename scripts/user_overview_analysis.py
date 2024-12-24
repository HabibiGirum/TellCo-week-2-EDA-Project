import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore

class UserOverviewAnalysis:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        # Load the dataset from the provided path
        # Return the loaded dataset 
        dataset = pd.read_csv(self.dataset_path)
        return dataset
    def data_info(self):
        # Display basic information about the dataset
        # Number of rows, columns, data types, and memory usage
        self.dataset.info()
    def check_dataset_missing(self):
        missing_percentage = (self.dataset.isnull().sum() / len(self.dataset)) * 100
        print(missing_percentage[missing_percentage > 0])
    def column(self):
        print(self.dataset.columns)

    
    def top_10_Handset(self):
        # Top 10 handsets type
        top_10_handsets = self.dataset['Handset Type'].value_counts().head(10)

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette="viridis")
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Count')
        plt.ylabel('Handset Type')
        
        plt.show()

    
    def top_3_manufacturers(self):
        # 3. Identifying the Top 3 Manufacturers
        manufacturer_counts = self.dataset['Handset Manufacturer'].value_counts().head(3)
        print("\nTop 3 Manufacturers:")
        print(manufacturer_counts)
        # Plotting Top 3 Manufacturers
        plt.figure(figsize=(10, 6))
        manufacturer_counts.plot(kind='bar', color='lightgreen')
        plt.title('Top 3 Manufacturers with the Most Handsets')
        plt.xlabel('Manufacturer')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        top_manufacturers = manufacturer_counts.index
        return top_manufacturers
    
    def top_5_handset_top_3_manufacturers(self):
        top_manufactures= self.top_3_manufacturers()
        for manufacturer in top_manufactures:
            top_handsets = self.dataset[self.dataset['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            print(f"\nTop 5 Handsets for {manufacturer}:")
            print(top_handsets)
            # Plotting Top 5 Handsets for each manufacturer
            plt.figure(figsize=(12, 8))
            top_handsets.plot(kind='bar', color='orange')
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handset')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        print("all in one graph ")
        # Top 3 manufacturers
        top_3 = top_manufactures

        # Top 5 handsets per manufacturer
        top_5_per_manufacturer = {}
        for manufacturer in top_3:
            top_5_per_manufacturer[manufacturer] = self.dataset[self.dataset['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)

        # Visualization
        plt.figure(figsize=(12, 8))
        for manufacturer, handsets in top_5_per_manufacturer.items():
            sns.barplot(x=handsets.values, y=handsets.index, label=manufacturer)

        plt.title('Top 5 Handsets per Top 3 Manufacturers')
        plt.xlabel('Count')
        plt.ylabel('Handset Type')
        plt.legend(title="Manufacturer")
        plt.show()



    def aggregate_per_user(self):
        # Ensure dataset is loaded
        if self.dataset is None:
            raise ValueError("Dataset is not loaded properly.")
        
        agg_data = self.dataset.groupby('IMSI').agg(
            number_of_xdr_sessions=('Bearer Id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_download=('Total DL (Bytes)', 'sum'),
            total_upload=('Total UL (Bytes)', 'sum'),
        ).reset_index()

        # Add the total_data_volume column
        agg_data['total_data_volume'] = agg_data['total_download'] + agg_data['total_upload']
        
        print("\nAggregated Data Sample:")
        print(agg_data.head())  # Display sample
        return agg_data

    def handle_missing_values(self, strategy='mean'):
        # Aggregate data first
        agg_data = self.aggregate_per_user()
        
        # Handle missing values
        print("\nBefore Handling Missing Values:")
        print(agg_data.isnull().sum())  # Show missing value count
        
        if strategy == 'mean':
            agg_data.fillna(agg_data.mean(), inplace=True)
        elif strategy == 'median':
            agg_data.fillna(agg_data.median(), inplace=True)
        elif strategy == 'mode':
            agg_data.fillna(agg_data.mode().iloc[0], inplace=True)
        else:
            raise ValueError("Invalid strategy for handling missing values. Choose from: mean, median, mode")
        
        print("\nAfter Handling Missing Values:")
        print(agg_data.isnull().sum())  # Verify missing values are handled
        return agg_data
    def decile_classification(self,data):
        decile_labels = ['Decile 1', 'Decile 2', 'Decile 3', 'Decile 4', 'Decile 5']
        data['duration_decile'] = pd.qcut(data['total_duration'], q=5, labels=decile_labels)
        decile_summary = data.groupby('duration_decile')[['total_data_volume']].sum()
        print("\nDecile Summary:")
        print(decile_summary)
        # Basic Metrics
        print("\nBasic Metrics:")
        print(data[['total_duration', 'total_download', 'total_upload', 'total_data_volume']].describe())
        
        # Correlation Analysis
        corr_matrix = data[['total_duration', 'total_download', 'total_upload', 'total_data_volume']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

        # PCA for Dimensionality Reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data[['total_duration', 'total_download', 'total_upload', 'total_data_volume']])
        print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, color='purple')
        plt.title('PCA: Dimensionality Reduction')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()
        plt.show()