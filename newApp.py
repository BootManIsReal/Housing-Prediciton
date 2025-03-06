import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
import gdown

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

FILE_ID = "10UtDe8J9Isz5js5B-ErOxnvgktzfVGCf"
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "my_model.pkl", quiet=False)
model = joblib.load("my_model.pkl")

def preprocess_input(data):
    data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)
    model_columns = model.feature_names_in_
    data = data.reindex(columns=model_columns, fill_value=0)
    return data

def main():
    st.markdown(
        """
        <h1 style='text-align: center; color: #2E8B57;'>🏡 Housing Price Prediction</h1>
        <h4 style='text-align: center; color: #555;'>Enter the details below to estimate the house price.</h4>
        """, unsafe_allow_html=True
    )
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            longitude = st.number_input("Longitude", value=-122.23)
            latitude = st.number_input("Latitude", value=37.88)
            housing_median_age = st.number_input("Housing Median Age", value=41.0)
            total_rooms = st.number_input("Total Rooms", value=880.0)
            total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
        
        with col2:
            population = st.number_input("Population", value=322.0)
            households = st.number_input("Households", value=126.0)
            median_income = st.number_input("Median Income", value=8.3252)
            ocean_proximity = st.selectbox(
                "🌊 Ocean Proximity",
                ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
                index=0
            )
    
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    input_data = preprocess_input(input_data)
    
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.success(f"The estimated housing price is: **${prediction[0]:,.2f}**")
    
    st.markdown("<br><hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
