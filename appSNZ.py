import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import os

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def load_data(dataset):
    if isinstance(dataset, str):
        df = pd.read_csv(dataset)
    else:
        df = pd.read_csv(dataset)
    return df

def main():
    # Load CSS
    if os.path.exists('style.css'):
        load_css('style.css')
    
    st.markdown("<h1 class='main-header'>Social Media Viral Content Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Viral Content Analysis</h2>", unsafe_allow_html=True)

    menu = ['Home', 'Analysis', 'Data Visualization', 'Machine Learning']
    choice = st.sidebar.selectbox("Select a page", menu)
    
    try:
        data = load_data('social_media_viral_content_dataset.csv')
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'social_media_viral_content_dataset.csv' is in the same directory.")
        return
        
    if choice == 'Home':
        left, middle, right = st.columns((2, 3, 2))
        # with middle:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("This application is designed to predict whether a content is viral or not based on various social media parameters. The dataset includes information about social media posts and their viral status.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("About Viral Content")
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("Viral content prediction is crucial for social media marketing and content strategy. Understanding what makes content go viral helps creators and marketers optimize their posts for maximum engagement and reach.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif choice == 'Analysis':
        st.subheader("Exploratory Data Analysis")
        st.write(data.head())
        if st.checkbox("Summary"):
            st.write(data.describe())
        elif st.checkbox("Correlation"):
            numeric_data = data.select_dtypes(include=[np.number])
            fig1 = plt.figure(figsize=(12,10))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig1)

    elif choice == 'Data Visualization':
        if st.checkbox('Viral Content Distribution'):
            fig2 = plt.figure(figsize=(8,6))
            sns.countplot(data=data, x='is_viral')
            plt.title('Distribution of Viral vs Non-Viral Content')
            st.pyplot(fig2)
            
        elif st.checkbox('Engagement Rate by Platform'):
            fig3 = plt.figure(figsize=(10,6))
            sns.boxplot(data=data, x='platform', y='engagement_rate')
            plt.xticks(rotation=45)
            st.pyplot(fig3)

    elif choice == 'Machine Learning':
        tab1, tab2, tab3 = st.tabs([":clipboard: Data", ":bar_chart: Visualisation", ":robot_face: Prediction"])
        upload_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        
        if upload_file:
            df = load_data(upload_file)

            with tab1:
                st.subheader("Uploaded Dataset")
                st.write(df)
                st.write(f"Dataset shape: {df.shape}")
                
            with tab2:
                st.subheader("Data Visualization")
                if 'engagement_rate' in df.columns:
                    fig4 = plt.figure(figsize=(10,6))
                    sns.histplot(data=df, x='engagement_rate', bins=30)
                    plt.title('Distribution of Engagement Rate')
                    st.pyplot(fig4)
                else:
                    st.warning("Engagement rate column not found in uploaded data")
                    
            with tab3:
                try:
                    model = pickle.load(open('brfSNZ.pkl', 'rb'))
                    
                    # Load original dataset to get expected features
                    original_data = load_data('social_media_viral_content_dataset.csv')
                    expected_features = original_data.select_dtypes(include=[np.number]).drop('is_viral', axis=1, errors='ignore').columns
                    
                    # Prepare data for prediction
                    prediction_data = df.select_dtypes(include=[np.number]).copy()
                    
                    # Add missing features with default values (mean from original data)
                    for feature in expected_features:
                        if feature not in prediction_data.columns:
                            default_value = original_data[feature].mean() if feature in original_data.columns else 0
                            prediction_data[feature] = default_value
                    
                    # Reorder columns to match expected order
                    prediction_data = prediction_data[expected_features]
                    
                    if prediction_data.empty:
                        st.error("No numeric columns found for prediction")
                    else:
                        st.info(f"Using {len(expected_features)} features for prediction")
                        # Convert to numpy array to avoid feature names warning
                        prediction = model.predict(prediction_data.values)
                        st.subheader("Prediction Results")
                        
                        pp = pd.DataFrame(prediction, columns=['Prediction'])
                        ndf = pd.concat([df, pp], axis=1)
                        
                        # Correct replacement syntax
                        ndf['Prediction'] = ndf['Prediction'].replace({0: 'Not Viral', 1: 'Viral'})
                        
                        # Style predictions
                        for idx, row in ndf.iterrows():
                            if row['Prediction'] == 'Viral':
                                st.markdown(f"<div class='prediction-viral'>Row {idx}: {row['Prediction']}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='prediction-not-viral'>Row {idx}: {row['Prediction']}</div>", unsafe_allow_html=True)
                        
                        st.write(ndf)
                        
                        # Summary
                        viral_count = (ndf['Prediction'] == 'Viral').sum()
                        total_count = len(ndf)
                        st.metric("Viral Content Predicted", f"{viral_count}/{total_count}", f"{viral_count/total_count*100:.1f}%")
                        
                except FileNotFoundError:
                    st.error("Model file 'brfSNZ.pkl' not found. Please ensure the model file is in the same directory.")
                except Exception as e:
                    st.error(f"Error loading model or making predictions: {str(e)}")
if __name__ == '__main__':
    main()