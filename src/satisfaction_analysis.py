import streamlit as st
import plotly.express as px
import pandas as pd

def satisfaction_analysis_page():
    st.title("📊 Satisfaction Analysis")
    
    # Load dataset
    df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')
    
    # Plot 1: Top Satisfied Users
    st.subheader("Top 10 Satisfied Users")
    top_users = df.nlargest(10, 'Satisfaction Score')
    fig = px.bar(top_users, x='User ID', y='Satisfaction Score', color='Satisfaction Score')
    st.plotly_chart(fig)
    
    # Plot 2: Satisfaction Score Distribution
    st.subheader("Satisfaction Score Distribution")
    fig = px.histogram(df, x='Satisfaction Score')
    st.plotly_chart(fig)
