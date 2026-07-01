import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt


from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# Hide Streamlit branding
hide_style = """
<style>

.main {
    padding-top:20px;
}
.metric-box{
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

hr{
    margin-top:40px;
}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

model = load_model('Stock Prediction Model.keras')

st.sidebar.title("📊 Stock Predictor")

st.sidebar.write(
"""
This application predicts stock prices using a trained LSTM Deep Learning model.

**Model**
- LSTM Neural Network
- Historical Yahoo Finance Data
Developed using:
- Streamlit
- TensorFlow/Keras
- Yahoo Finance API
"""
)

#st.header('Stock Market Predictor')

st.title("📈 Stock Market Prediction SYSTEM")
st.write("Predict stock prices using Deep Learning (LSTM Model).")

st.divider()


stock =st.text_input('Enter Stock Symbol', 'TATASTEEL.NS').upper()
start = '2016-06-30'
end = '2026-06-30'

with st.spinner("Downloading stock data..."):
    data = yf.download(stock, start ,end)

if data.empty:
    st.error("No stock data found. Please enter a valid Yahoo Finance ticker.")
    st.stop()

st.success("Stock data loaded successfully!")

st.subheader('Historical Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Stock Price vs 50-Days Moving Average')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(12,5))
plt.plot(ma_50_days, label="MA50", color="orange")
plt.plot(data.Close, label="Closing Price", color="royalblue")
plt.show()
plt.grid(alpha=0.3)
st.pyplot(fig1)

st.subheader('50-Days vs 100-Days Moving Average')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12,5))
plt.plot(ma_50_days,  label="MA50", color="orange")
plt.plot(ma_100_days, label="MA100", color="green")
plt.plot(data.Close, label="Closing Price", color="royalblue")
plt.show()
plt.grid(alpha=0.3)
st.pyplot(fig2)

st.subheader('100-Days vs 200-Days Moving Average')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12,5))
plt.plot(ma_100_days, label="MA100", color="green")
plt.plot(ma_200_days, label="MA200", color="red")
plt.plot(data.Close, label="Closing Price", color="royalblue")
plt.show()
plt.grid(alpha=0.3)
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

prediction = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(12,5))
plt.plot(prediction, label="Predicted Price", color="red")
plt.plot(y, label="Original Price", color="green")
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
plt.grid(alpha=0.3)
st.pyplot(fig4)
#------------------------------------------------------------------------------------------------------------------------

# Recommendation part

similar_stocks = {
    # Technology
    "GOOG": ["MSFT", "AAPL", "META", "AMZN"],
    "MSFT": ["GOOG", "AAPL", "META", "ORCL"],
    "AAPL": ["GOOG", "MSFT", "META", "AMZN"],
    "META": ["GOOG", "MSFT", "AAPL", "SNAP"],

    # Indian IT
    "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "WIPRO.NS": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],

    # Steel
    "TATASTEEL.NS": ["JSWSTEEL.NS", "JINDALSTEL.NS", "SAIL.NS", "HINDALCO.NS"],
    "JSWSTEEL.NS": ["TATASTEEL.NS", "JINDALSTEL.NS", "SAIL.NS", "HINDALCO.NS"],
    "SAIL.NS": ["TATASTEEL.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "HINDALCO.NS"],

    # Banking
    "HDFCBANK.NS": ["ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "ICICIBANK.NS": ["HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],

    # Automobile
    "TATAMOTORS.NS": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "BAJAJ-AUTO.NS"],
    "MARUTI.NS": ["TATAMOTORS.NS", "M&M.NS", "ASHOKLEY.NS", "BAJAJ-AUTO.NS"],

    # Oil & Gas
    "RELIANCE.NS": ["ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS"],

    # FMCG
    "HINDUNILVR.NS": ["ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"]
}

st.divider()
st.subheader("📌 Similar Stocks")

stock = stock.upper()

if stock in similar_stocks:
    cols = st.columns(4)

    for i, ticker in enumerate(similar_stocks[stock]):
        with cols[i % 4]:
            st.info(f"📈 {ticker}")
else:
    st.warning("No similar stock recommendations available.")

st.divider()

st.markdown(
"""
<div style='text-align:center; color:white; font-size:14px background-color:gray; bottom-margin:0px'>

© 2026 <b>Stock Market Prediction Dashboard</b><br>

Developed by <b>Khushi</b><br><br>

This application is intended solely for educational and research purposes.
The predictions generated by this model should not be interpreted as financial or investment advice.
Always perform your own research before making investment decisions.

</div>
""",
unsafe_allow_html=True
)
