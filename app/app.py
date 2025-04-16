# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import zipfile

# Download dataset from Kaggle if not present
@st.cache_data
def download_and_extract_data():
    if not os.path.exists("data/realtor-data.csv"):
        os.makedirs("data", exist_ok=True)
        os.system("kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset -p data")
        with zipfile.ZipFile("data/usa-real-estate-dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
    return pd.read_csv("data/realtor-data.csv")

df = download_and_extract_data()

# ----------------------------------------
# Load and clean data
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/realtor-data.zip.csv")

    # Drop unnecessary columns
    df = df.drop(['prev_sold_date', 'status', 'street', 'brokered_by'], axis=1)

    # Drop missing values and filter invalid states
    df = df.dropna()
    excluded = ['GUAM', 'DISTRICT OF COLUMBIA', 'NEW BRUNSWICK', 'PUERTO RICO', 'VIRGIN ISLANDS']
    df = df[~df['state'].isin(excluded)]

    # Remove outliers
    def remove_outliers(df, columns, threshold=1.5):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
    df = remove_outliers(df, numeric_cols)

    df["price_per_sqft"] = df["price"] / df["house_size"]
    df['state'] = df['state'].str.upper().str.strip()

    return df

# Load data
df = load_data()

# ----------------------------------------
# Sidebar Filters
# ----------------------------------------
st.sidebar.header("🔎 Filter Options")
selected_state = st.sidebar.selectbox("Select State", sorted(df['state'].unique()))
price_range = st.sidebar.slider("Price Range", int(df['price'].min()), int(df['price'].max()), (100000, 600000))
show_hist = st.sidebar.checkbox("Show Price Distribution")
show_map = st.sidebar.checkbox("Show Choropleth Map", value=True)

filtered_df = df[(df['state'] == selected_state) & 
                 (df['price'] >= price_range[0]) & 
                 (df['price'] <= price_range[1])]

# ----------------------------------------
# Main Title
# ----------------------------------------
st.title("🏡 US Housing Price Dashboard")
st.markdown(f"Currently viewing **{selected_state}** properties priced between **${price_range[0]:,}** and **${price_range[1]:,}**.")

# ----------------------------------------
# Price Distribution Plot
# ----------------------------------------
if show_hist:
    st.subheader("📊 Price Distribution")
    fig1, ax = plt.subplots()
    sns.histplot(filtered_df["price"], bins=50, kde=True, ax=ax)
    st.pyplot(fig1)

# ----------------------------------------
# Choropleth Map
# ----------------------------------------
if show_map:
    st.subheader("🗺️ Median House Price by State")
    state_prices = df.groupby("state")["price"].median().reset_index()
    
    # Map state names to abbreviations
    us_states = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
        'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
        'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA',
        'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO', 'MONTANA': 'MT',
        'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
        'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
        'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
    }

    state_prices["state_abbr"] = state_prices["state"].map(us_states)

    fig = px.choropleth(
        state_prices,
        locations="state_abbr",
        locationmode="USA-states",
        color="price",
        hover_name="state",
        color_continuous_scale="Blues",
        scope="usa"
    )
    st.plotly_chart(fig)

# ----------------------------------------
# Raw Data (Optional)
# ----------------------------------------
if st.checkbox("Show Raw Data"):
    st.write(filtered_df)

