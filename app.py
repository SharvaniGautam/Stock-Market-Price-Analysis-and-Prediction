import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries

# Load LSTM model
model = load_model(r"C:\Users\skgau\Downloads\Stock_Market_Prediction_ML\Stock Predictions Model.keras")

st.set_page_config(layout="centered")
st.title('📈 Stock Market Predictor')

# --- User Input ---
api_key = 'DT4HF7G0HHZ2ALGC'
stock = st.text_input('Enter Stock Symbol', 'GOOG').upper().strip()
start = '2012-01-01'
end = '2022-12-31'

# --- Fetch Data from Alpha Vantage ---
ts = TimeSeries(key=api_key, output_format='pandas')
try:
    data_raw, _ = ts.get_daily(symbol=stock, outputsize='full')
except:
    st.error("🚫 API Error: Check your internet connection or API key.")
    st.stop()

if data_raw.empty:
    st.error(f"⚠️ No data returned for '{stock}'. Please check the symbol.")
    st.stop()

# --- Process & Slice Data ---
data_raw.index = pd.to_datetime(data_raw.index)
data_raw = data_raw.sort_index()

if pd.to_datetime(start) > data_raw.index.max():
    st.error(f"⚠️ Date range exceeds available data. Latest: {data_raw.index.max().date()}")
    st.stop()

data = data_raw.loc[start:end]
if data.empty:
    st.error(f"⚠️ No data available between {start} and {end}.")
    st.stop()

data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

st.subheader('📊 Raw Stock Data')
st.dataframe(data.tail())

# --- Moving Averages ---
ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

# --- Plots ---
st.subheader('Price vs MA50')
fig1 = plt.figure(figsize=(8, 4))
plt.plot(data['Close'], label='Close')
plt.plot(ma_50, label='MA50', color='red')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(8, 4))
plt.plot(data['Close'], label='Close')
plt.plot(ma_50, label='MA50', color='red')
plt.plot(ma_100, label='MA100', color='blue')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(8, 4))
plt.plot(data['Close'], label='Close')
plt.plot(ma_100, label='MA100', color='red')
plt.plot(ma_200, label='MA200', color='blue')
plt.legend()
st.pyplot(fig3)

# --- Train/Test Split ---
data_close = data[['Close']]
train_data = data_close[0:int(len(data_close) * 0.80)]
test_data = data_close[int(len(data_close) * 0.80):]

scaler = MinMaxScaler(feature_range=(0, 1))
past_100 = train_data.tail(100)
test_combined = pd.concat([past_100, test_data], ignore_index=True)

if test_combined.shape[0] <= 100:
    st.error("⚠️ Not enough data for prediction. Use a stock with more historical data.")
    st.stop()

test_scaled = scaler.fit_transform(test_combined)

# --- Create Test Sequences ---
x_test = []
y_test = []
for i in range(100, test_scaled.shape[0]):
    x_test.append(test_scaled[i-100:i])
    y_test.append(test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# --- Predict ---
predictions = model.predict(x_test)

# --- Reverse Scaling ---
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# --- Plot Prediction ---
st.subheader('🔮 Original vs Predicted Price')
fig4 = plt.figure(figsize=(8, 4))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(predictions, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
