import streamlit as st
import plotly.express as px
import pandas as pd

def user_engagement_page():
    st.title("📊 User Engagement Analysis")
    
    # Load dataset
    df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')
    
    # Plot 1: Daily Active Users
    st.subheader("Daily Active Users")
    fig = px.line(df, x='Date', y='Active Users')
    st.plotly_chart(fig)
    
    # Plot 2: Engagement Score Distribution
    st.subheader("Engagement Score Distribution")
    fig = px.histogram(df, x='Engagement Score')
    st.plotly_chart(fig)
