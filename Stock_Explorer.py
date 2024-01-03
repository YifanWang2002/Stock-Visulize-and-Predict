import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(layout="wide")

st.title("Explore Stock Prices and More")

st.subheader("Made with :heartpulse: by Yifan")

st.markdown("Please choose a dashboard using the sidebar on the left.")

st.subheader("References:")

reference1 = """
Jingles (Hong Jing). (2021, April 29). *Stock Prediction using Deep Neural Networks and LSTM*. Alpha Vantage. https://www.alphavantage.co/stock-prediction-deep-neural-networks-lstm/
"""

st.markdown(reference1)


