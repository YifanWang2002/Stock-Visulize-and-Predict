import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

# Function to load data
@st.cache_data
def load_data(ticker="IBM"):
    data = pd.read_csv(f'data/daily_{ticker}.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Load the data
data = load_data()

st.title('Stock Information Dashboard')

files = os.listdir('data')

# Filter and extract ticker names
tickers = [file.replace('daily_', '').replace('.csv', '') for file in files if file.startswith('daily_') and file.endswith('.csv')]

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox('Select Ticker', tickers)
    data = load_data(ticker)
    # Add a multi-select box for selecting data columns to display
    data_options = ['open', 'high', 'low', 'close', 'volume']
    selected_data = st.multiselect('Select data to display', data_options, default=['close'])
    if 'volume' in selected_data:
        selected_data = ['volume']

with col2:
    start_date = st.date_input('Start date', value=data['timestamp'].min())

with col3: 
    end_date = st.date_input('End date', value=data['timestamp'].max())

# Convert start_date and end_date to datetime64[ns] type
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Validate and filter data based on selected date range
if start_date <= end_date:
    filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

    # Plotting the selected data series
    st.subheader(f'Stock Price Data for {ticker}')
    fig, ax = plt.subplots(figsize=(15, 5))
    for data_type in selected_data:
        if 'volume' not in selected_data:
            ax.plot(filtered_data['timestamp'], filtered_data[data_type], label=f'{data_type.capitalize()} Price')
            ax.set_ylabel('Price')
        else: 
            ax.plot(filtered_data['timestamp'], filtered_data[data_type], label=f'{data_type.capitalize()}')
            ax.set_ylabel('Volume')
        ax.set_xlabel('Date')
    ax.legend()
    st.pyplot(fig)
else:
    st.error('Start date must be earlier than End date')

