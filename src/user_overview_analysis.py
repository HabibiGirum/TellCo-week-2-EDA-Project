import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Utility functions
def plot_bar_chart(x, y, title, x_label, y_label, orientation='v'):
    """Helper function to create and display a bar chart using Plotly."""
    fig = px.bar(
        x=x,
        y=y,
        orientation=orientation,
        title=title,
        labels={"x": x_label, "y": y_label},
    )
    st.plotly_chart(fig)

def display_correlation_matrix(data, features):
    """Helper function to display a correlation matrix."""
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def perform_pca(data, features, n_components=2):
    """Helper function to perform PCA and display results."""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data[features])
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    fig = px.scatter(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        title="PCA: Dimensionality Reduction",
        labels={"x": "Principal Component 1", "y": "Principal Component 2"},
    )
    st.plotly_chart(fig)

# User Overview Analysis Class
class UserOverviewAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset

    def data_info(self):
        st.write("### Heading 5 rows")
        st.text(self.dataset.head(5))

    def check_missing_values(self):
        st.write("### Missing Values Overview")
        missing_percentage = (self.dataset.isnull().sum() / len(self.dataset)) * 100
        st.write(missing_percentage[missing_percentage > 0])

    def show_columns(self):
        st.write("### Column Names")
        st.write(self.dataset.columns.tolist())

    def top_n_handsets(self, n=10):
        st.write(f"### Top {n} Handsets Used by Customers")
        top_handsets = self.dataset['Handset Type'].value_counts().head(n)
        plot_bar_chart(
            x=top_handsets.values,
            y=top_handsets.index,
            title=f"Top {n} Handsets Used by Customers",
            x_label="Count",
            y_label="Handset Type",
            orientation="h"
        )

    def top_n_manufacturers(self, n=3):
        st.write(f"### Top {n} Handset Manufacturers")
        manufacturer_counts = self.dataset['Handset Manufacturer'].value_counts().head(n)
        st.write(manufacturer_counts)
        plot_bar_chart(
            x=manufacturer_counts.index,
            y=manufacturer_counts.values,
            title=f"Top {n} Manufacturers with the Most Handsets",
            x_label="Manufacturer",
            y_label="Frequency"
        )
        return manufacturer_counts.index

    def top_5_handset_top_3_manufacturers(self):
        st.write("### Top 5 Handsets per Top 3 Manufacturers")
        top_manufacturers = self.top_n_manufacturers(n=3)
        for manufacturer in top_manufacturers:
            st.write(f"#### Top 5 Handsets for {manufacturer}")
            top_handsets = self.dataset[self.dataset['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            st.write(top_handsets)
            fig = px.bar(
                x=top_handsets.values,
                y=top_handsets.index,
                orientation="h",
                title=f"Top 5 Handsets for {manufacturer}",
                labels={"x": "Count", "y": "Handset Type"},
            )
            st.plotly_chart(fig)
        combined_data = []
        for manufacturer in top_manufacturers:
            top_handsets = self.dataset[self.dataset['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            for handset, count in top_handsets.items():
                combined_data.append({"Manufacturer": manufacturer, "Handset": handset, "Count": count})
        combined_df = pd.DataFrame(combined_data)
        fig_combined = px.bar(
            combined_df,
            x="Count",
            y="Handset",
            color="Manufacturer",
            orientation="h",
            title="Top 5 Handsets per Top 3 Manufacturers (Combined)",
            labels={"Count": "Frequency", "Handset": "Handset Type"},
        )
        st.plotly_chart(fig_combined)

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
        display_correlation_matrix(data, ['total_duration', 'total_download', 'total_upload', 'total_data_volume'])
        perform_pca(data, ['total_duration', 'total_download', 'total_upload', 'total_data_volume'])

# Streamlit App Interface
def user_overview_page():
    st.title("📊 User Overview Analysis")
    st.write("After uploading your dataset, use the sidebar to explore detailed insights about user overviews, including various analytical options for your data.")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        analysis = UserOverviewAnalysis(dataset)
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis option",
            [
                "Dataset Info",
                "Check Missing Values",
                "Show Columns",
                "Top 10 Handsets",
                "Top 3 Manufacturers",
                "Top 5 Handsets per Top 3 Manufacturers",
                "Aggregate Per User",
                "Decile Classification"
            ]
        )

        if analysis_option == "Dataset Head 5 row":
            analysis.data_info()
        elif analysis_option == "Check Missing Values":
            analysis.check_missing_values()
        elif analysis_option == "Show Columns":
            analysis.show_columns()
        elif analysis_option == "Top 10 Handsets":
            analysis.top_n_handsets()
        elif analysis_option == "Top 3 Manufacturers":
            analysis.top_n_manufacturers()
        elif analysis_option == "Top 5 Handsets per Top 3 Manufacturers":
            analysis.top_5_handset_top_3_manufacturers()
        elif analysis_option == "Aggregate Per User":
            agg_data = analysis.aggregate_per_user()
            st.write(agg_data)
        elif analysis_option == "Decile Classification":
            agg_data = analysis.aggregate_per_user()
            analysis.decile_classification(agg_data)

if __name__ == '__main__':
    user_overview_page()
