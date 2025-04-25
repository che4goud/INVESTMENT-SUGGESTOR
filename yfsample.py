import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
import streamlit as st


# CSS to inject contained in a string
css = """
<style>
/* Dark Theme with Gradients */
/* General settings for the entire page */
body {
    background: linear-gradient(to right, #141E30, #243B55);
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}
/* Style the header */
h1 {
    color: #E0E0E0;
    text-align: center;
    margin-bottom: 20px;
}

h2, h3 {
    color: #CCCCCC;
    text-align: center;
    margin-top: 20px;
}
</style>
"""

# Inject CSS with Markdown
st.markdown(css, unsafe_allow_html=True)

# Streamlit app setup
st.title('📈STOCK ANALYSIS AND PREDICTION')
st.sidebar.title('USER INPUT')

# User inputs
ticker = st.sidebar.text_input('Enter ticker symbol for company:', 'AAPL')
index_choice = st.sidebar.text_input('Enter index ticker symbol like ^NSEI:', '^GSPC')
years = st.sidebar.slider('Number of years for data:', 1, 20, 5)
mcyears = st.sidebar.slider('Number of years montecarlo',1,20,5)
desired_return = st.sidebar.number_input('Enter your desired return (%):', value=10.0) / 100  # Convert to decimal

# DATE OPTIMIZING
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365*years)

# Load company data
company_data = yf.download(ticker, start=start_date, end=end_date,progress=False)
market_data = yf.download(index_choice, start=start_date, end=end_date,progress=False)

# Check if data is loaded properly
if company_data.empty or market_data.empty:
    st.error(f"Failed to load data for {ticker} or {index_choice}. Please check the ticker symbols and try again.")
    st.stop()

# Calculate returns
company_data['Returns'] = company_data['Close'].pct_change()
market_data['Returns'] = market_data['Close'].pct_change()

# Prepare data for regression
X = market_data['Returns'].dropna()
y = company_data['Returns'].dropna()

# Align X and y to ensure same index
X, y = X.align(y, join='inner')

# Add constant to independent variable
X = sm.add_constant(X)

# Run regression
model = sm.OLS(y, X).fit()



# Plot the data points and regression line
st.subheader('COMPANY VS MARKET RETURNS')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Returns'], y, color='black', label='Data Points')
predicted_returns = model.predict(X)
ax.plot(X['Returns'], predicted_returns, color='midnightblue' , label='Regression Line')
ax.set_xlabel('Market Returns (%)')
ax.set_ylabel('Company Returns (%)')
ax.set_title('Company Returns vs. Market Returns')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)

# Explanation of the outcome
st.write("""
The scatter plot shows the relationship between the company's returns and the market's returns. 
- The Black dots represent the data points.
- The Blue line represents the regression line, which predicts the company's returns based on the market's returns.
The slope of this line is the beta, measuring the volatility of the company relative to the market.
""")

# Conclusions from the Linear Regression
beta = model.params['Returns']
alpha = model.params['const']
st.subheader('')

col1, col2 = st.columns(2)
col1.metric("     BETA",beta)
col2.metric("     ALPHA",alpha)
st.subheader('')


# Calculating Risks
r_f = 0.069  # risk-free rate (6.9% annualized)
r_m = market_data['Returns'].mean() * 252  # annualized average market return
r_p = r_m - r_f  # risk premium

# Check if risk premium is negative
if r_p < 0:
    r_p = 0  # Set to 0 if negative to avoid issues

# Using CAPM model
capm = r_f + beta * r_p  # cost of equity using CAPM model


st.subheader('Risk & CAPM Analysis')
st.subheader('')

col1, col2, col3 = st.columns(3)
col1.metric("     COST OF EQUITY",capm)
col2.metric("     RISK PREMIUM",r_p)
col3.metric("     AVG MARKET RETURN",r_m)
st.subheader('')

# Explanation of the outcome
st.write("""
The cost of equity is calculated using the Capital Asset Pricing Model (CAPM), which represents the return required by investors to compensate for the risk of investing in the company.
- The risk premium is the additional return over the risk-free rate that investors expect from investing in the market.
""")

# Simulated returns for the next 252 days
mu = company_data['Returns'].mean()
sigma = company_data['Returns'].std()

# Example of simulating future prices
if isinstance(company_data.columns, pd.MultiIndex):
    company_data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in company_data.columns]

if 'Adj Close' in company_data.columns:
    initial_price = company_data['Adj Close'].iloc[-1]
elif 'Close' in company_data.columns:
    initial_price = company_data['Close'].iloc[-1]
else:
    st.error("Neither 'Adj Close' nor 'Close' column found in the data.")
    st.stop()

st.write(f"Initial price is ${initial_price:.2f}")

st.subheader('Monte Carlo Simulations')
fig, ax = plt.subplots(figsize=(10, 6))

simulated_days = 252 * mcyears # Simulating for 5 years to reduce data size
simulated_paths = 100  # Reduced paths for quicker plotting

simulation_results = []

for i in range(simulated_paths):
    sim_rets = np.random.normal(mu, sigma, simulated_days)
    sim_price = initial_price * (sim_rets + 1).cumprod()
    ax.plot(sim_price, alpha=0.3, color='darkgreen')
    simulation_results.append(sim_price)

ax.axhline(initial_price, c='k', linewidth=0.5, label='Initial Price')
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')
ax.set_title('Monte Carlo Simulation of Future Prices')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)

# Explanation of the outcome
st.write("""
The Monte Carlo simulation generates multiple potential future price paths for the company based on historical returns and volatility.
- Each grey line represents a possible trajectory of the company's stock price over the next 1260 trading days (approximately 5 years).
- The black horizontal line represents the initial stock price.
This visual representation shows the range of possible future outcomes.
""")

# Investment Recommendation and Time to Desired Return
st.subheader('Investment Recommendation')
if capm >= desired_return:
    st.header(f"**Recommendation: Yes, you should invest in {ticker}.** The expected return of {capm:.4f} meets or exceeds your desired return of {desired_return:.4f}.")
else:
    st.header(f"**Recommendation: No, you should not invest in {ticker}.** The expected return of {capm:.4f} does not meet your desired return of {desired_return:.4f}.")

# Calculate the time required to achieve the desired return
desired_return_factor = 1 + desired_return
time_to_desired_return = []

for path in simulation_results:
    for day in range(simulated_days):
        if path[day] / initial_price >= desired_return_factor:
            time_to_desired_return.append(day)
            break

if time_to_desired_return:
    avg_days_to_desired_return = np.mean(time_to_desired_return)
    st.header(f"On average, it will take approximately {avg_days_to_desired_return:.0f} trading days to achieve your desired return of {desired_return:.2%}.")
else:
    st.header("Based on the simulations, it is unlikely to achieve the desired return within the simulated period.")

# Display regression results
st.subheader('REGRESSION RESULTS')
st.write(model.summary())

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")

# Load Animation
animation_symbol = "❄"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)

