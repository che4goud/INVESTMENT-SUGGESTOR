import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
import streamlit as st

# Streamlit app setup
st.title('Stock Analysis and Prediction')
st.sidebar.header('User Input')

# User inputs
ticker = st.sidebar.text_input('Enter ticker symbol for company:', 'AAPL')
index_choice = st.sidebar.text_input('Enter index ticker symbol like ^NSEI:', '^GSPC')
years = st.sidebar.slider('Number of years for data:', 1, 20, 5)
desired_return = st.sidebar.number_input('Enter your desired return (%):', value=10.0) / 100  # Convert to decimal

# DATE OPTIMIZING
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365*years)

# Load company data
company_data = yf.download(ticker, start=start_date, end=end_date)
market_data = yf.download(index_choice, start=start_date, end=end_date)

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

# Display regression results
st.subheader('Regression Results')
st.write(model.summary())

# Plot the data points and regression line
st.subheader('Company Returns vs. Market Returns')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Returns'], y, color='blue', label='Data Points')
predicted_returns = model.predict(X)
ax.plot(X['Returns'], predicted_returns, color='red', label='Regression Line')
ax.set_xlabel('Market Returns (%)')
ax.set_ylabel('Company Returns (%)')
ax.set_title('Company Returns vs. Market Returns')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)

# Explanation of the outcome
st.write("""
The scatter plot shows the relationship between the company's returns and the market's returns. 
- The blue dots represent the data points.
- The red line represents the regression line, which predicts the company's returns based on the market's returns.
The slope of this line is the beta, measuring the volatility of the company relative to the market.
""")

# Conclusions from the Linear Regression
beta = model.params['Returns']
alpha = model.params['const']
st.subheader('Conclusions from Linear Regression')
st.write(f"""
- **Alpha (Intercept):** {alpha:.4f}
  - This indicates the expected return of the company when the market return is zero.
- **Beta (Slope):** {beta:.4f}
  - This indicates the volatility of the company relative to the market. A beta greater than 1 indicates higher volatility, while a beta less than 1 indicates lower volatility.
""")

# Calculating Risks
r_f = 0.069 / 252  # daily risk-free rate (6.9% annualized)
r_m = market_data['Returns'].mean() * 252  # annualized average market return
r_p = r_m - (0.069)  # risk premium

# Check if risk premium is negative
if r_p < 0:
    r_p = 0  # Set to 0 if negative to avoid issues

# Using CAPM model
capm = r_f * 252 + beta * r_p  # cost of equity using CAPM model

st.subheader('Risk and CAPM Analysis')
st.write(f"The cost of equity is {capm:.4f}")
st.write(f"The risk premium is {r_p:.4f}")
st.write(f"The average market return is {r_m:.4f}")

# Explanation of the outcome
st.write("""
The cost of equity is calculated using the Capital Asset Pricing Model (CAPM), which represents the return required by investors to compensate for the risk of investing in the company.
- The risk premium is the additional return over the risk-free rate that investors expect from investing in the market.
- If the risk premium is negative, it is set to zero to avoid unrealistic cost of equity values.
""")

# Simulated returns for the next 252 days
mu = company_data['Returns'].mean()
sigma = company_data['Returns'].std()

# Example of simulating future prices
initial_price = company_data['Adj Close'].iloc[-1]
st.write(f"Initial price is ${initial_price:.2f}")

st.subheader('Monte Carlo Simulations')
fig, ax = plt.subplots(figsize=(10, 6))

simulated_days = 252 * 10  # Simulating for 10 years
simulated_paths = 100

simulation_results = []

for i in range(simulated_paths):
    sim_rets = np.random.normal(mu, sigma, simulated_days)
    sim_price = initial_price * (sim_rets + 1).cumprod()
    ax.plot(sim_price, alpha=0.3, color='grey')
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
- Each grey line represents a possible trajectory of the company's stock price over the next 2520 trading days (approximately 10 years).
- The black horizontal line represents the initial stock price.
This visual representation shows the range of possible future outcomes.
""")

# Conclusions from the Monte Carlo Simulation
st.subheader('Conclusions from Monte Carlo Simulation')
st.write(f"""
- The Monte Carlo simulation provides a range of potential future stock prices based on historical performance.
- The range of predicted prices helps in understanding the potential volatility and risk associated with the stock.
- This simulation can be used by investors to gauge the possible future performance and make informed investment decisions.
""")

# Investment Recommendation and Time to Desired Return
st.subheader('Investment Recommendation')
if capm >= desired_return:
    st.title(f"**Recommendation: Yes, you should invest in {ticker}.** The expected return of {capm:.4f} meets or exceeds your desired return of {desired_return:.4f}.")
else:
    st.title(f"**Recommendation: No, you should not invest in {ticker}.** The expected return of {capm:.4f} does not meet your desired return of {desired_return:.4f}.")

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
    st.title(f"On average, it will take approximately {avg_days_to_desired_return:.0f} trading days to achieve your desired return of {desired_return:.2%}.")
else:
    st.title("Based on the simulations, it is unlikely to achieve the desired return within the simulated period.")

# Explanation of the outcome
st.write("""
The average number of trading days to achieve the desired return is calculated based on the Monte Carlo simulations.
This provides an estimate of how long you might have to wait to see the desired return on your investment.
""")

