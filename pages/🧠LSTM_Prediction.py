import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries 
import streamlit as st
import altair as alt
from Prediction import *
import os

WINDOW_SIZE = 20
TRAIN_SIZE = 0.8
LR = 0.01
STEP = 40
DEVICE = "cpu"
INPUT_SIZE = 1
LSTM_SIZE = 32
NUM_LSTM_LAYERS = 2
DROP_OUT = 0.2

st.set_page_config(layout="wide")

# Function to load data
@st.cache_data
def load_data(ticker="IBM"):
    data = pd.read_csv(f'data/daily_{ticker}.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Load the data
data = load_data()

st.title('Predicting Future Stock Prices')

files = os.listdir('data')

tickers = [file.replace('daily_', '').replace('.csv', '') for file in files if file.startswith('daily_') and file.endswith('.csv')]

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.selectbox('Select Ticker', tickers)
    data = load_data(ticker)
    WINDOW_SIZE = st.selectbox('Select Window Size', [2, 3, 5, 10, 15, 20, 40])

    # More function to add technical indicators after high volume account acqurired
with col2:
    model = st.selectbox('Select Model', ['SMA','EMA',"LSTM"])
with col3:
    if model == "LSTM":
        batch_size = st.selectbox('Select Batch Size', [32,64,128])
        num_epoch = st.selectbox('Select Training Epoch', [50,100,200])

data_date, data_close_price, num_data_points, display_date_range = load_data_from_csv(ticker)


#st.subheader(f"Daily close price for {ticker}: "+display_date_range)
#fig, ax = plt.subplots(figsize=(15, 5))
#ax.plot(data_date, data_close_price, label='Close Price')
#ax.set_xlabel('Date')
#ax.set_ylabel('Price')
#plt.tick_params(bottom = False) 
#xticks = [data_date[i] if ((i%180==0 and (num_data_points-i) > 90) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
#x = np.arange(0,len(xticks))
#plt.xticks(x, xticks, rotation='vertical')
#ax.legend()
#st.pyplot(fig)
st.markdown("Please be patient, this might take a mintue to run")

if model == "LSTM":
    st.markdown(" :red[Click the button to Start Training]")
    if st.button('Start'):
        with st.status("Preparing Training Data ..."):
            st.write('Normalizing Data')
            scaler = Normalizer()
            normalized_data_close_price = scaler.fit_transform(data_close_price)

            st.write("Train-test Split ...")
            data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=WINDOW_SIZE)
            data_y = prepare_data_y(normalized_data_close_price, window_size=WINDOW_SIZE)
            split_index = int(data_y.shape[0]*TRAIN_SIZE)
            data_x_train = data_x[:split_index]
            data_x_val = data_x[split_index:]
            data_y_train = data_y[:split_index]
            data_y_val = data_y[split_index:]

            to_plot_data_y_train = np.zeros(num_data_points)
            to_plot_data_y_val = np.zeros(num_data_points)

            to_plot_data_y_train[WINDOW_SIZE:split_index+WINDOW_SIZE] = scaler.inverse_transform(data_y_train)
            to_plot_data_y_val[split_index+WINDOW_SIZE:] = scaler.inverse_transform(data_y_val)

            to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
            to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

            st.write('Loading Data into Model ...')   
            dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
            dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
            print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
            print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True) 

            st.write("Success!")

        with st.status("Start Model Training"):
            lstm_model = LSTMModel(input_size=INPUT_SIZE, hidden_layer_size=LSTM_SIZE, num_layers=NUM_LSTM_LAYERS, output_size=1, dropout=DROP_OUT)
            lstm_model = lstm_model.to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(lstm_model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=0.1)

            progress_text = "Training in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for epoch in range(num_epoch):
                loss_train, lr_train = run_epoch(train_dataloader, lstm_model, criterion, optimizer, scheduler, is_training=True)
                loss_val, lr_val = run_epoch(val_dataloader, lstm_model, criterion, optimizer, scheduler)
                scheduler.step()
                my_bar.progress((epoch + 1)/num_epoch, text=progress_text)
                
                st.write('Epoch[{}/{}] | loss train:{:.4f}, test:{:.4f} | lr:{:.4f}'
                        .format(epoch+1, num_epoch, loss_train, loss_val, lr_train))
                
            my_bar.empty()

        with st.status("Model Evaluation"):
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
            
            lstm_model.eval()
            # predict on the training data, to see how well the model managed to learn and memorize

            predicted_train = np.array([])

            for idx, (x, y) in enumerate(train_dataloader):
                x = x.to(DEVICE)
                out = lstm_model(x)
                out = out.cpu().detach().numpy()
                predicted_train = np.concatenate((predicted_train, out))

            # predict on the validation data, to see how the model does

            predicted_val = np.array([])

            for idx, (x, y) in enumerate(val_dataloader):
                x = x.to(DEVICE)
                out = lstm_model(x)
                out = out.cpu().detach().numpy()
                predicted_val = np.concatenate((predicted_val, out))

            # prepare data for plotting
            to_plot_data_y_train_pred = np.zeros(num_data_points)
            to_plot_data_y_val_pred = np.zeros(num_data_points)

            to_plot_data_y_train_pred[WINDOW_SIZE:split_index+WINDOW_SIZE] = scaler.inverse_transform(predicted_train)
            to_plot_data_y_val_pred[split_index+WINDOW_SIZE:] = scaler.inverse_transform(predicted_val)

            to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
            to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

            # plots
            st.subheader(f"Daily close price for {ticker}: Prediction")
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(data_date, data_close_price, label="Actual prices")
            ax.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)")
            ax.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)")
            xticks = [data_date[i] if ((i%180==0 and (num_data_points-i) > 180) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
            x = np.arange(0,len(xticks))
            plt.xticks(x, xticks, rotation='vertical')
            plt.tick_params(bottom = False) 
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

elif model == "SMA":
    sma_values = simple_moving_average(data['close'][::-1], WINDOW_SIZE)
    st.subheader(f"Daily close price for {ticker}: SMA Prediction")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data_date, data['close'][::-1], label="Actual prices", alpha=0.7)
    ax.plot(data_date, sma_values, label='SMA',alpha=0.7)
    xticks = [data_date[i] if ((i%180==0 and (num_data_points-i) > 180) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation='vertical')
    ax.tick_params(axis='x', which='both', length=0) 
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

elif model == "EMA":
    ema_values = exponential_moving_average(data['close'][::-1], WINDOW_SIZE)
    st.subheader(f"Daily close price for {ticker}: EMA Prediction")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data_date, data['close'][::-1], label="Actual prices", alpha=0.7)
    ax.plot(data_date, ema_values, label='EMA',alpha=0.7)
    xticks = [data_date[i] if ((i%180==0 and (num_data_points-i) > 180) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation='vertical')
    ax.tick_params(axis='x', which='both', length=0)     
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

st.subheader("Next Day Prediction")

if model == "LSTM":
    if "lstm_model" not in locals():
        st.write("Please click on the above button to train the model first")
    
    else:
        lstm_model.eval()

        x = torch.tensor(data_x_unseen).float().to(DEVICE).unsqueeze(0).unsqueeze(2)
        prediction = lstm_model(x)
        prediction = prediction.cpu().detach().numpy()

        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]
        to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        plot_date_test = data_date[-plot_range+1:]
        plot_date_test.append("next day")

        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10)
        ax.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10)
        ax.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20)
        ax.set_title("Predicted close price of the next trading day")
        ax.grid(which='major', axis='y', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.write("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))

elif model == "SMA":
    plot_range = 10
    
    to_plot_data_y_val = data_close_price[-plot_range:]
    sma_values = simple_moving_average(pd.Series(data_close_price[-plot_range-WINDOW_SIZE:]), WINDOW_SIZE)
    to_plot_data_y_val_pred = sma_values[WINDOW_SIZE-1:-1]
    to_plot_data_y_val_pred_next = sma_values.iloc[-1]

    plot_date_test = data_date[-plot_range:]
    plot_date_test.append("next day")

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(plot_date_test[:-1], to_plot_data_y_val, label="Actual prices", marker=".", markersize=10)
    ax.plot(plot_date_test[:-1], to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10)
    ax.scatter(plot_date_test[-1], to_plot_data_y_val_pred_next, label="Predicted price for next day", marker=".", s=100)
    ax.set_title("Predicted close price of the next trading day")
    ax.grid(which='major', axis='y', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.write("Predicted close price of the next trading day:", round(to_plot_data_y_val_pred_next, 2))

elif model == "EMA":
    plot_range = 10
    
    to_plot_data_y_val = data_close_price[-plot_range:]
    ema_values = exponential_moving_average(pd.Series(data_close_price[-plot_range-WINDOW_SIZE:]), WINDOW_SIZE)
    to_plot_data_y_val_pred = ema_values[WINDOW_SIZE-1:-1]
    to_plot_data_y_val_pred_next = ema_values.iloc[-1]

    plot_date_test = data_date[-plot_range:]
    plot_date_test.append("next day")

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(plot_date_test[:-1], to_plot_data_y_val, label="Actual prices", marker=".", markersize=10)
    ax.plot(plot_date_test[:-1], to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10)
    ax.scatter(plot_date_test[-1], to_plot_data_y_val_pred_next, label="Predicted price for next day", marker=".", s=100)
    ax.set_title("Predicted close price of the next trading day")
    ax.grid(which='major', axis='y', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.write("Predicted close price of the next trading day:", round(to_plot_data_y_val_pred_next, 2))


st.subheader("Performance Evaluation")

st.image("metrics.png", caption="RMSE for different window size, by model, using AAPL's daily unjusted close price")


