import pandas as pd 
class Preprocessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        # Load the dataset from the provided path
        # Return the loaded dataset 
        dataset = pd.read_csv(self.dataset_path)
        return dataset
    
    def handle_missing_values(self, strategy='mean'):
        # Handle missing values in the dataset based on the specified strategy
        # Replace missing values with the mean, median, or mode of the respective columns
        if strategy == 'mean':
            self.dataset.fillna(self.dataset.mean(), inplace=True)
        elif strategy =='median':
            self.dataset.fillna(self.dataset.median(), inplace=True)
        elif strategy =='mode':
            self.dataset.fillna(self.dataset.mode().iloc[0], inplace=True)
        else:
            raise ValueError('Invalid strategy for handling missing values. Choose from: mean, median, mode')
    
    
        