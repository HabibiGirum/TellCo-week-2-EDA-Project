import streamlit as st
from src.user_overview_analysis import user_overview_page
from src.user_engagement_analysis import user_engagement_page
from src.experience_analytics import experience_analytics_page
from src.satisfaction_analysis import satisfaction_analysis_page

# Configure Streamlit app
st.set_page_config(
    page_title="Data Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    [
        "User Overview Analysis",
        "User Engagement Analysis",
        "Experience Analysis",
        "Satisfaction Analysis"
    ]
)

# Render Selected Page
if page == "User Overview Analysis":
    user_overview_page()
elif page == "User Engagement Analysis":
    user_engagement_page()
elif page == "Experience Analysis":
    experience_analytics_page()
elif page == "Satisfaction Analysis":
    satisfaction_analysis_page()
