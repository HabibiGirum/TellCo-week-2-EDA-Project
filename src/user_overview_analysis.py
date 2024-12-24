import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the UserOverviewAnalysis class
class UserOverviewAnalysis:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return pd.read_csv(self.dataset_path)
    
    def data_info(self):
        st.write("### Dataset Information")
        buffer = self.dataset.info(buf=None)
        st.text(buffer)
    
    def check_dataset_missing(self):
        st.write("### Missing Values Overview")
        missing_percentage = (self.dataset.isnull().sum() / len(self.dataset)) * 100
        st.write(missing_percentage[missing_percentage > 0])
    
    def column(self):
        st.write("### Column Names")
        st.write(self.dataset.columns.tolist())
    
    def top_10_Handset(self):
        st.write("### Top 10 Handsets Used by Customers")
        top_10_handsets = self.dataset['Handset Type'].value_counts().head(10)
        fig = px.bar(
            x=top_10_handsets.values, 
            y=top_10_handsets.index, 
            orientation='h',
            title="Top 10 Handsets Used by Customers",
            labels={"x": "Count", "y": "Handset Type"}
        )
        st.plotly_chart(fig)
    
    def top_3_manufacturers(self):
        st.write("### Top 3 Handset Manufacturers")
        manufacturer_counts = self.dataset['Handset Manufacturer'].value_counts().head(3)
        st.write(manufacturer_counts)
        fig = px.bar(
            x=manufacturer_counts.index, 
            y=manufacturer_counts.values,
            title="Top 3 Manufacturers with the Most Handsets",
            labels={"x": "Manufacturer", "y": "Frequency"}
        )
        st.plotly_chart(fig)
        return manufacturer_counts.index
    
    def top_5_handset_top_3_manufacturers(self):
        st.write("### Top 5 Handsets for Top 3 Manufacturers")
        top_manufacturers = self.top_3_manufacturers()
        
        for manufacturer in top_manufacturers:
            st.write(f"#### {manufacturer}")
            top_handsets = self.dataset[self.dataset['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            fig = px.bar(
                x=top_handsets.index, 
                y=top_handsets.values, 
                title=f"Top 5 Handsets for {manufacturer}",
                labels={"x": "Handset", "y": "Frequency"}
            )
            st.plotly_chart(fig)
    
    def aggregate_per_user(self):
        st.write("### Aggregate Data Per User")
        agg_data = self.dataset.groupby('IMSI').agg(
            number_of_xdr_sessions=('Bearer Id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_download=('Total DL (Bytes)', 'sum'),
            total_upload=('Total UL (Bytes)', 'sum')
        ).reset_index()
        agg_data['total_data_volume'] = agg_data['total_download'] + agg_data['total_upload']
        st.write(agg_data.head())
        return agg_data
    
    def decile_classification(self, data):
        st.write("### Decile Classification")
        decile_labels = ['Decile 1', 'Decile 2', 'Decile 3', 'Decile 4', 'Decile 5']
        data['duration_decile'] = pd.qcut(data['total_duration'], q=5, labels=decile_labels)
        
        decile_summary = data.groupby('duration_decile')[['total_data_volume']].sum()
        st.write(decile_summary)
        
        st.write("### Correlation Matrix")
        corr_matrix = data[['total_duration', 'total_download', 'total_upload', 'total_data_volume']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.write("### PCA: Dimensionality Reduction")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data[['total_duration', 'total_download', 'total_upload', 'total_data_volume']])
        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        
        fig = px.scatter(
            x=reduced_data[:, 0], 
            y=reduced_data[:, 1],
            title="PCA: Dimensionality Reduction",
            labels={"x": "Principal Component 1", "y": "Principal Component 2"}
        )
        st.plotly_chart(fig)

# Streamlit App Interface
def user_overview_page():
    st.title("📊 User Overview Analysis")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        analysis = UserOverviewAnalysis(uploaded_file)
        
        # Sidebar for Navigation
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis option",
            [
                "Dataset Info", "Check Missing Values", "Show Columns",
                "Top 10 Handsets", "Top 3 Manufacturers", "Top 5 Handsets per Manufacturer",
                "Aggregate Per User", "Decile Classification"
            ]
        )
        
        if analysis_option == "Dataset Info":
            analysis.data_info()
        elif analysis_option == "Check Missing Values":
            analysis.check_dataset_missing()
        elif analysis_option == "Show Columns":
            analysis.column()
        elif analysis_option == "Top 10 Handsets":
            analysis.top_10_Handset()
        elif analysis_option == "Top 3 Manufacturers":
            analysis.top_3_manufacturers()
        elif analysis_option == "Top 5 Handsets per Manufacturer":
            analysis.top_5_handset_top_3_manufacturers()
        elif analysis_option == "Aggregate Per User":
            agg_data = analysis.aggregate_per_user()
            st.write(agg_data)
        elif analysis_option == "Decile Classification":
            agg_data = analysis.aggregate_per_user()
            analysis.decile_classification(agg_data)

# Run the Streamlit app
if __name__ == '__main__':
    user_overview_page()
