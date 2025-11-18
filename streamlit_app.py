import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timedelta
import threading
import time
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')

# IB API Integration
class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        
    def error(self, reqId, errorCode, errorString, *args):
        if errorCode == 2176 and "fractional share" in errorString.lower():
            return
        print(f"Error {errorCode}: {errorString}")
        
    def nextValidId(self, orderId):
        self.connected = True
        print("Connected to IB")
        
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data received for reqId {reqId}")

# Helper Functions
def create_equity_contract(symbol):
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract

def create_vix_contract():
    contract = Contract()
    contract.symbol = "VIX"
    contract.secType = "IND"
    contract.exchange = "CBOE"
    contract.currency = "USD"
    return contract

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

def calculate_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

# Initialize session state
if 'ib_app' not in st.session_state:
    st.session_state.ib_app = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'vix_data' not in st.session_state:
    st.session_state.vix_data = None
if 'iv_data' not in st.session_state:
    st.session_state.iv_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Page config
st.set_page_config(page_title="Earnings Trading Dashboard", layout="wide")
st.title("📊 Earnings Trading Dashboard - IV Crush Analysis")

# Sidebar - IB Connection
st.sidebar.header("Interactive Brokers Connection")
host = st.sidebar.text_input("Host", value="127.0.0.1")
port = st.sidebar.number_input("Port", value=7497, step=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Connect", disabled=st.session_state.connected):
        try:
            st.session_state.ib_app = IBApp()
            
            def connect_thread():
                try:
                    st.session_state.ib_app.connect(host, int(port), 0)
                    st.session_state.ib_app.run()
                except Exception as e:
                    st.error(f"Connection error: {e}")
            
            thread = threading.Thread(target=connect_thread, daemon=True)
            thread.start()
            
            # Wait for connection
            for i in range(100):
                if st.session_state.ib_app.connected:
                    try:
                        server_version = st.session_state.ib_app.serverVersion()
                        if server_version is not None and server_version > 0:
                            st.session_state.connected = True
                            st.sidebar.success(f"Connected (Server: {server_version})")
                            break
                    except:
                        pass
                time.sleep(0.1)
            
            if not st.session_state.connected:
                st.sidebar.error("Failed to connect")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

with col2:
    if st.button("Disconnect", disabled=not st.session_state.connected):
        if st.session_state.ib_app:
            st.session_state.ib_app.disconnect()
            st.session_state.connected = False
            st.session_state.ib_app = None
            st.sidebar.info("Disconnected")

# Main interface
st.sidebar.header("Analysis Setup")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
earnings_date_str = st.sidebar.text_input("Earnings Date (YYYY-MM-DD)", value="2025-08-27")
days_to_expiry = st.sidebar.number_input("Days to Expiry", value=30, min_value=1, max_value=365)

if st.sidebar.button("Analyze IV Crush", disabled=not st.session_state.connected):
    try:
        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        
        with st.spinner("Fetching data from Interactive Brokers..."):
            # Calculate date range
            start_date = earnings_date - timedelta(days=10)
            end_date = earnings_date + timedelta(days=10)
            
            # Clear previous data
            st.session_state.ib_app.historical_data.clear()
            
            # Query stock data
            stock_contract = create_equity_contract(ticker)
            if 1 in st.session_state.ib_app.historical_data:
                del st.session_state.ib_app.historical_data[1]
            
            st.session_state.ib_app.reqHistoricalData(
                reqId=1,
                contract=stock_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr="3 W",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            # Wait for stock data
            timeout = 15
            start_time = time.time()
            while 1 not in st.session_state.ib_app.historical_data and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if 1 in st.session_state.ib_app.historical_data:
                stock_data = pd.DataFrame(st.session_state.ib_app.historical_data[1])
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                stock_data.set_index('date', inplace=True)
                st.session_state.stock_data = stock_data
            
            # Query VIX data
            vix_contract = create_vix_contract()
            if 2 in st.session_state.ib_app.historical_data:
                del st.session_state.ib_app.historical_data[2]
            
            st.session_state.ib_app.reqHistoricalData(
                reqId=2,
                contract=vix_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr="3 W",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            start_time = time.time()
            while 2 not in st.session_state.ib_app.historical_data and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if 2 in st.session_state.ib_app.historical_data:
                vix_data = pd.DataFrame(st.session_state.ib_app.historical_data[2])
                vix_data['date'] = pd.to_datetime(vix_data['date'])
                vix_data.set_index('date', inplace=True)
                st.session_state.vix_data = vix_data
            
            # Query IV data
            if 3 in st.session_state.ib_app.historical_data:
                del st.session_state.ib_app.historical_data[3]
            
            st.session_state.ib_app.reqHistoricalData(
                reqId=3,
                contract=stock_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr="3 W",
                barSizeSetting="1 day",
                whatToShow="OPTION_IMPLIED_VOLATILITY",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            start_time = time.time()
            while 3 not in st.session_state.ib_app.historical_data and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if 3 in st.session_state.ib_app.historical_data:
                iv_data = pd.DataFrame(st.session_state.ib_app.historical_data[3])
                iv_data['date'] = pd.to_datetime(iv_data['date'])
                iv_data.set_index('date', inplace=True)
                
                raw_iv = iv_data['close']
                if raw_iv.max() > 5:
                    daily_iv_decimal = raw_iv / 100.0
                    iv_data['implied_vol'] = daily_iv_decimal
                else:
                    iv_data['implied_vol'] = raw_iv
                
                st.session_state.iv_data = iv_data
        
        # Perform analysis
        stock_data = st.session_state.stock_data
        stock_dates = stock_data.index
        
        pre_date_actual = stock_dates[stock_dates <= earnings_date].max() if len(stock_dates[stock_dates <= earnings_date]) > 0 else stock_dates.min()
        post_date_actual = stock_dates[stock_dates > earnings_date].min() if len(stock_dates[stock_dates > earnings_date]) > 0 else stock_dates.max()
        
        pre_stock_price = stock_data.loc[pre_date_actual, 'close']
        post_open = stock_data.loc[post_date_actual, 'open']
        post_close = stock_data.loc[post_date_actual, 'close']
        post_stock_price = (post_open + post_close) / 2
        
        # Get IV values
        if st.session_state.iv_data is not None:
            iv_dates = st.session_state.iv_data.index
            pre_iv_date = iv_dates[iv_dates <= earnings_date].max() if len(iv_dates[iv_dates <= earnings_date]) > 0 else iv_dates.min()
            post_iv_date = iv_dates[iv_dates > earnings_date].min() if len(iv_dates[iv_dates > earnings_date]) > 0 else iv_dates.max()
            
            pre_iv = st.session_state.iv_data.loc[pre_iv_date, 'implied_vol']
            post_iv = st.session_state.iv_data.loc[post_iv_date, 'implied_vol']
        else:
            # Estimate from VIX
            if st.session_state.vix_data is not None:
                vix_dates = st.session_state.vix_data.index
                pre_vix_date = vix_dates[vix_dates <= earnings_date].max()
                post_vix_date = vix_dates[vix_dates > earnings_date].min()
                pre_vix = st.session_state.vix_data.loc[pre_vix_date, 'close']
                post_vix = st.session_state.vix_data.loc[post_vix_date, 'close']
                pre_iv = pre_vix / 100.0 * 1.5
                post_iv = post_vix / 100.0 * 1.2
            else:
                pre_iv = 0.40
                post_iv = 0.25
        
        # Calculate IV crush
        iv_crush_pct = (pre_iv - post_iv) / pre_iv * 100
        
        # Option pricing
        time_to_expiry = days_to_expiry / 365.0
        atm_strike_price = pre_stock_price
        risk_free_rate = 0.05
        
        pre_call_price = black_scholes_call(pre_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, pre_iv)
        pre_put_price = black_scholes_put(pre_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, pre_iv)
        post_call_price = black_scholes_call(post_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, post_iv)
        post_put_price = black_scholes_put(post_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, post_iv)
        
        pre_straddle_price = pre_call_price + pre_put_price
        post_straddle_price = post_call_price + post_put_price
        straddle_change = post_straddle_price - pre_straddle_price
        straddle_change_pct = straddle_change / pre_straddle_price * 100
        
        # Greeks
        pre_call_delta = calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, pre_iv, 'call')
        pre_put_delta = calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, pre_iv, 'put')
        post_call_delta = calculate_delta(post_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, post_iv, 'call')
        post_put_delta = calculate_delta(post_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, post_iv, 'put')
        
        pre_straddle_delta = pre_call_delta + pre_put_delta
        post_straddle_delta = post_call_delta + post_put_delta
        
        pre_vega = calculate_vega(pre_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, pre_iv) * 2
        post_vega = calculate_vega(post_stock_price, atm_strike_price, time_to_expiry, risk_free_rate, post_iv) * 2
        
        # Store results
        st.session_state.analysis_results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'pre_date': pre_date_actual,
            'post_date': post_date_actual,
            'pre_stock_price': pre_stock_price,
            'post_stock_price': post_stock_price,
            'atm_strike': atm_strike_price,
            'pre_iv': pre_iv,
            'post_iv': post_iv,
            'iv_crush_pct': iv_crush_pct,
            'pre_call': pre_call_price,
            'post_call': post_call_price,
            'pre_put': pre_put_price,
            'post_put': post_put_price,
            'pre_straddle': pre_straddle_price,
            'post_straddle': post_straddle_price,
            'straddle_change': straddle_change,
            'straddle_change_pct': straddle_change_pct,
            'pre_delta': pre_straddle_delta,
            'post_delta': post_straddle_delta,
            'pre_vega': pre_vega,
            'post_vega': post_vega,
        }
        
        st.success("Analysis complete!")
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# Display results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.header(f"📈 Analysis Results for {results['ticker']}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stock Price (Pre)", f"${results['pre_stock_price']:.2f}")
    with col2:
        st.metric("Stock Price (Post)", f"${results['post_stock_price']:.2f}", 
                 f"{((results['post_stock_price'] - results['pre_stock_price']) / results['pre_stock_price'] * 100):+.2f}%")
    with col3:
        st.metric("Pre-Earnings IV", f"{results['pre_iv']:.1%}")
    with col4:
        st.metric("IV Crush", f"-{results['iv_crush_pct']:.1f}%", delta_color="inverse")
    
    st.divider()
    
    # Options pricing
    st.subheader("ATM Options Pricing & Straddle")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Pre-Earnings**")
        st.write(f"Call: ${results['pre_call']:.2f}")
        st.write(f"Put: ${results['pre_put']:.2f}")
        st.write(f"**Straddle: ${results['pre_straddle']:.2f}**")
    
    with col2:
        st.write("**Post-Earnings**")
        st.write(f"Call: ${results['post_call']:.2f}")
        st.write(f"Put: ${results['post_put']:.2f}")
        st.write(f"**Straddle: ${results['post_straddle']:.2f}**")
    
    with col3:
        st.write("**Change**")
        call_change = results['post_call'] - results['pre_call']
        put_change = results['post_put'] - results['pre_put']
        st.write(f"Call: ${call_change:+.2f}")
        st.write(f"Put: ${put_change:+.2f}")
        st.write(f"**Straddle: ${results['straddle_change']:+.2f} ({results['straddle_change_pct']:+.1f}%)**")
    
    st.divider()
    
    # P/L Analysis
    st.subheader("P/L Analysis")
    col1, col2 = st.columns(2)
    with col1:
        long_pnl = results['straddle_change']
        st.metric("LONG Straddle P/L", f"${long_pnl:+.2f}", f"{results['straddle_change_pct']:+.1f}%")
    with col2:
        short_pnl = -results['straddle_change']
        st.metric("SHORT Straddle P/L", f"${short_pnl:+.2f}", f"{-results['straddle_change_pct']:+.1f}%")
    
    st.divider()
    
    # Greeks
    st.subheader("Greeks Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pre-Earnings Delta", f"{results['pre_delta']:.3f}")
    with col2:
        st.metric("Post-Earnings Delta", f"{results['post_delta']:.3f}", 
                 f"{results['post_delta'] - results['pre_delta']:+.3f}")
    with col3:
        st.metric("Pre-Earnings Vega", f"{results['pre_vega']:.2f}")
    with col4:
        st.metric("Post-Earnings Vega", f"{results['post_vega']:.2f}", 
                 f"{results['post_vega'] - results['pre_vega']:+.2f}")
    
    st.divider()
    
    # Visualizations
    st.subheader("IV Crush Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Stock price and IV
    if st.session_state.stock_data is not None:
        earnings_date = results['earnings_date']
        window_stock = st.session_state.stock_data[
            (st.session_state.stock_data.index >= earnings_date - timedelta(days=5)) &
            (st.session_state.stock_data.index <= earnings_date + timedelta(days=5))
        ]
        
        ax1.plot(window_stock.index, window_stock['close'], 'b-', linewidth=2, label='Stock Price')
        ax1.axvline(x=earnings_date, color='red', linestyle='--', alpha=0.7, label='Earnings Date')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'{results["ticker"]} Stock Price Around Earnings')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot IV if available
        if st.session_state.iv_data is not None:
            window_iv = st.session_state.iv_data[
                (st.session_state.iv_data.index >= earnings_date - timedelta(days=5)) &
                (st.session_state.iv_data.index <= earnings_date + timedelta(days=5))
            ]
            
            if len(window_iv) > 0:
                ax1_twin = ax1.twinx()
                iv_percentage = window_iv['implied_vol'] * 100
                ax1_twin.plot(window_iv.index, iv_percentage, 'g-', linewidth=2, label='Implied Volatility')
                ax1_twin.set_ylabel('Implied Volatility (%)', color='green')
                ax1_twin.tick_params(axis='y', labelcolor='green')
                ax1_twin.legend(loc='upper right')
        
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Options comparison
    option_types = ['Call', 'Put', 'Straddle']
    pre_prices = [results['pre_call'], results['pre_put'], results['pre_straddle']]
    post_prices = [results['post_call'], results['post_put'], results['post_straddle']]
    
    x = np.arange(len(option_types))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, pre_prices, width, label='Pre-Earnings (High IV)', color='lightblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, post_prices, width, label='Post-Earnings (Low IV)', color='lightcoral', alpha=0.8)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                f'${height1:.1f}', ha='center', va='bottom', fontsize=9)
        ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                f'${height2:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Option Strategy')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('ATM Options & Straddle: IV Crush Impact')
    ax2.set_xticks(x)
    ax2.set_xticklabels(option_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)

else:
    st.info("Connect to Interactive Brokers and run an analysis to see results.")
