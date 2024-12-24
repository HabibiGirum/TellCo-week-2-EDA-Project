import streamlit as st 

# Assuming the ExperienceAnalytics class is implemented as mentioned earlier
from scripts.experience_analytics import ExperienceAnalytics

from scripts.satisfaction_analysis import SatisfactionAnalysis
# Streamlit App Interface
def satisfaction_analysis_page():
    st.title("Satisfaction Analysis")
    st.write("After uploading your dataset, use the sidebar to explore detailed insights about user satisfaction, including various analytical.")
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
