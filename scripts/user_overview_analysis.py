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
        print(f'Number of rows: {self.dataset.shape[0]}')
        print(f'Number of columns: {self.dataset.shape[1]}')
        print(f'Data types: {self.dataset.dtypes}')
        print(f'Memory usage: {self.dataset.memory_usage().sum()} bytes')
    
    def top_10_Handset(self):
        # 2. Identifying the Top 10 Handsets
        handset_counts = self.dataset['Handset Type'].value_counts().head(10)
        print("\nTop 10 Handsets:")
        print(handset_counts)
        # Plotting Top 10 Handsets
        plt.figure(figsize=(10, 6))
        handset_counts.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Handset')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_10_handsets.png')
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
        plt.savefig('top_3_manufacturers.png')
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
            plt.figure(figsize=(10, 6))
            top_handsets.plot(kind='bar', color='orange')
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handset')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'top_5_handsets_{manufacturer}.png')
            plt.show()
    def aggregate_per_user(self):
        agg_data = self.dataset.groupby('IMSI').agg(
            number_of_xdr_sessions=('Bearer Id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_download=('Total DL (Bytes)', 'sum'),
            total_upload=('Total UL (Bytes)', 'sum'),
        ).reset_index()
        # Add the total_data_volume column
        agg_data['total_data_volume'] = agg_data['total_download'] + agg_data['total_upload']       

        print("\nAggregated Data Sample:")
        # agg_data.head()
        return agg_data
    
    def handle_missing_values(self, strategy='mean', ):
        # Handle missing values in the dataset based on the specified strategy
        # Replace missing values with the mean, median, or mode of the respective columns
        agg_data = self.aggregate_per_user()
        print(agg_data)
        if strategy == 'mean':
            agg_data.fillna(agg_data.mean(), inplace=True)
        elif strategy =='median':
            agg_data.fillna(agg_data.median(), inplace=True)
        elif strategy =='mode':
            agg_data.fillna(agg_data.mode().iloc[0], inplace=True)
        else:
            raise ValueError('Invalid strategy for handling missing values. Choose from: mean, median, mode')
    
    
        