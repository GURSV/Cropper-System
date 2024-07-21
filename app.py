import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title='Croper System ● Gurmehar Singh ● CO21318', page_icon='🌱', layout='centered', initial_sidebar_state='collapsed')

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    title_html = '''
        <div>
            <h1 style='color: MEDIUMSEAGREEN; text-align: left;'>🌱 -Croper System- 🌱</h1>
        <div/>
    '''

    st.markdown(title_html, unsafe_allow_html=True)

    file_path = 'Crop_recommendation.csv'

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    dset = pd.read_csv(file_path)

    menu = ['Home', 'Analysis', 'Recommend', 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader('Home') 
        st.dataframe(dset.head(10))

    elif choice == 'Analysis':
        st.subheader('Data Analysis')

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Summary Statistics")
        st.write(dset.describe())

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Correlation Matrix")
        corr = dset.iloc[:, :-1].corr()
        fig, ax = plt.subplots()
        cmap_choice = st.selectbox("Choose a colormap for the correlation matrix", 
                                   ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        sns.heatmap(corr, annot=True, ax=ax, cmap=cmap_choice)
        st.pyplot(fig)

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Pair Plot - 1")
        important_features = ['N', 'P', 'K']
        fig2 = sns.pairplot(dset[important_features + ['label']], hue='label')
        st.pyplot(fig2)

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Pair Plot - 2")
        important_features_ = ['temperature', 'ph', 'rainfall']
        fig2_ = sns.pairplot(dset[important_features_ + ['label']], hue='label')
        st.pyplot(fig2_)

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Scatter Plot")
        fig4 = px.bar(dset, x='humidity', y='label', color='label', title='Humidity vs. Label')
        st.plotly_chart(fig4)

        st.markdown(''' <div> <br/> <div/>''', unsafe_allow_html=True)

        st.write("### Distribution of Each Feature")
        for col in dset.columns[:-1]:  # Exclude the label column
            fig3, ax3 = plt.subplots()
            sns.histplot(dset[col], kde=True, ax=ax3)
            st.pyplot(fig3)

    elif choice == 'Recommend':
        col1, col2 = st.columns([2, 2])

        with col1:
            with st.expander("ℹ️ Information", expanded=True):
                st.write('''
                    Crop recommenders are integral to the success of precision agriculture, drawing on a diverse array of factors to offer targeted advice. Precision agriculture strives to uncover these factors for each unique site, fine-tuning crop selection decisions. Despite the advancements brought by this individualized approach, continuous monitoring of system outcomes remains critical. It's essential to acknowledge that precision agriculture systems vary widely. 
                    In agriculture, the need for accurate and precise recommendations is paramount, as mistakes can result in significant resource and financial losses.
                ''')

        with col2:
            st.subheader('Discover the best crop to cultivate on your farm 👨🏻‍🌾')
            N = st.number_input('NITROGEN', 1, 10000)
            P = st.number_input('PHOSPHORUS', 1, 10000)
            K = st.number_input('POTASSIUM', 1, 10000)
            temp = st.number_input('TEMPERATURE', 0.0, 100000.0)
            humidity = st.number_input('Humidity in %', 0.0, 100000.0)
            ph = st.number_input("POTENZ HYDROGEN (Ph)", 0.0, 100000.0)
            rainfall = st.number_input('RAINFALL in mm', 0.0, 100000.0)

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            if st.button('Predict'):
                loaded_model = load_model('model.pkl')
                prediction = loaded_model.predict(single_pred)
                col1.write('''
                    ## RESULTS 🔎
                ''')
                col1.success(f'🌾 {prediction.item().title()} IS RECOMMENDED BY THE 1st BEST A.I. MODEL FOR YOUR FARM 🌾')

                st.markdown('''<br/>''', unsafe_allow_html=True)

                loaded_model2 = load_model('model_best_2nd.pkl')
                prediction2 = loaded_model2.predict(single_pred)
                col1.success(f'🌾 {prediction2.item().title()} IS RECOMMENDED BY THE 2nd BEST A.I. MODEL FOR YOUR FARM 🌾')

                st.markdown('''<br/>''', unsafe_allow_html=True)
                
                loaded_model3 = load_model('model_best_3rd.pkl')
                prediction3 = loaded_model3.predict(single_pred)
                col1.success(f'🌾 {prediction3.item().title()} IS RECOMMENDED BY THE 3rd BEST A.I. MODEL FOR YOUR FARM 🌾')

                st.warning('NOTE: This A.I. application is for educational purposes only!')
            
            hide_menu_style = '''
                <style> 
                    #MainMenu {
                        visibility: hidden;
                    }
                </style>
            '''

    elif choice == 'About':
        about_html = '''
        <div>
            <h3 style='color: white; text-align: left;'>About Croper Sytem</h3>
        <div/>
    '''
        st.markdown(about_html, unsafe_allow_html=True)

        st.write('''
                    Welcome to the Croper System, a cutting-edge AI-driven tool designed to assist farmers in making informed decisions about crop selection. Our system leverages advanced machine learning algorithms to analyze soil and environmental conditions, providing precise crop recommendations tailored to your farm’s unique characteristics.
                ''')
        
        about_html = '''
        <div>
            <h3 style='color: white; text-align: left;'>Key Features</h3>
        <div/>
    '''
        st.markdown(about_html, unsafe_allow_html=True)

        st.write('''
                    🌾 Exploratory Data Analysis (EDA): We perform thorough data cleaning, visualization, and statistical analysis to understand the underlying patterns in agricultural data.
                 
                    🌾 Machine Learning Models: We employ models such as Logistic Regression, Decision Trees, Random Forest, and K-Nearest Neighbors (KNN) to predict the best crops for your land.
                 
                    🌾 Seamless Integration: Using 'make_pipeline', we ensure efficient preprocessing, training, and evaluation of models, reducing data leakage and enhancing prediction accuracy.
                 
                    🌾 User-Friendly Interface: Our platform is designed to be intuitive and accessible, allowing farmers to input key parameters and receive recommendations easily.
                ''')
        
        about_html = '''
        <div>
            <h3 style='color: white; text-align: left;'>Mission</h3>
        <div/>
    '''
        st.markdown(about_html, unsafe_allow_html=True)
        
        st.write('''
                    At Croper System, our mission is to empower farmers with the tools and knowledge they need to optimize crop production and sustainability. By integrating technology and agriculture, we aim to contribute to the advancement of precision farming and ensure food security.
                ''')
        
        about_html = '''
        <div>
            <h3 style='color: white; text-align: left;'>Disclaimer</h3>
        <div/>
    '''
        st.markdown(about_html, unsafe_allow_html=True)

        st.write('''
                    Please note that while our recommendations are based on sophisticated algorithms and extensive data analysis, they are intended for educational purposes. We advise farmers to consider local conditions and expert advice before making final decisions.

                    Thank you for using Croper System. We are committed to continuous improvement and welcome your feedback.
                ''')
        
        st.markdown('''<br/>''', unsafe_allow_html=True)

        st.write('''
                    🌾 © 2024 Croper System • Created by Gurmehar Singh • CO21318 🌾
                ''')
    
hide_menu_style = '''
            <style> 
                #MainMenu {
                    visibility: hidden;
                }
            </style>
        '''



st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()