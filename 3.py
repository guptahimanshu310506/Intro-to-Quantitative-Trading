import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import io

# Calibration Data (ATM Options for Implied Vols)
calib_data = [
    # Stock, Strike, Maturity, Price
    ('DTC', 50, 1, 52.44),
    ('DTC', 50, 2, 54.77),
    ('DTC', 50, 5, 61.23),
    ('DTC', 75, 1, 28.97),
    ('DTC', 75, 2, 33.04),
    ('DTC', 75, 5, 43.47),
    ('DTC', 100, 1, 10.45),
    ('DTC', 100, 2, 16.13),
    ('DTC', 100, 5, 29.14),
    ('DTC', 125, 1, 2.32),
    ('DTC', 125, 2, 6.54),
    ('DTC', 125, 5, 18.82),
    ('DTC', 150, 1, 0.36),
    ('DTC', 150, 2, 2.34),
    ('DTC', 150, 5, 11.89),

    ('DFC', 50, 1, 52.45),
    ('DFC', 50, 2, 54.9),
    ('DFC', 50, 5, 61.87),
    ('DFC', 75, 1, 29.11),
    ('DFC', 75, 2, 33.34),
    ('DFC', 75, 5, 43.99),
    ('DFC', 100, 1, 10.45),
    ('DFC', 100, 2, 16.13),
    ('DFC', 100, 5, 29.14),
    ('DFC', 125, 1, 2.8),
    ('DFC', 125, 2, 7.39),
    ('DFC', 125, 5, 20.15),
    ('DFC', 150, 1, 1.26),
    ('DFC', 150, 2, 4.94),
    ('DFC', 150, 5, 17.46),

    ('DEC', 50, 1, 52.44),
    ('DEC', 50, 2, 54.8),
    ('DEC', 50, 5, 61.42),
    ('DEC', 75, 1, 29.08),
    ('DEC', 75, 2, 33.28),
    ('DEC', 75, 5, 43.88),
    ('DEC', 100, 1, 10.45),
    ('DEC', 100, 2, 16.13),
    ('DEC', 100, 5, 29.14),
    ('DEC', 125, 1, 1.96),
    ('DEC', 125, 2, 5.87),
    ('DEC', 125, 5, 17.74),
    ('DEC', 150, 1, 0.16),
    ('DEC', 150, 2, 1.49),
    ('DEC', 150, 5, 9.7),
]

spot_price = 100
risk_free_rate = 0.05
stocks = ['DTC', 'DFC', 'DEC']

# Correlation matrix
correlations = {
    ('DTC', 'DFC'): 0.75,
    ('DTC', 'DEC'): 0.5,
    ('DFC', 'DEC'): 0.25,
}

corr_matrix = np.ones((3, 3))
for i in range(3):
    for j in range(3):
        if i != j:
            pair = (stocks[i], stocks[j])
            corr_matrix[i, j] = correlations.get(pair, correlations.get((stocks[j], stocks[i]), 1.0))

# Black-Scholes
def bs_call_price(S, K, T, r, sigma):
    if T == 0 or sigma == 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol_call(mkt_price, S, K, T, r):
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - mkt_price
    try:
        return brentq(objective, 1e-6, 5)
    except:
        return 0.2  # fallback

# Compute Implied Vols
implied_vols = {}
for stock, K, T, price in calib_data:
    vol = implied_vol_call(price, spot_price, K, T, risk_free_rate)
    implied_vols[(stock, T)] = vol

# GBM Simulator
def simulate_correlated_gbm(S0_vec, r, vols, corr_mat, T, n_steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    n_assets = len(S0_vec)
    L = np.linalg.cholesky(corr_mat)
    paths = np.zeros((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0_vec

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=(n_paths, n_assets))
        dW = Z @ L.T * np.sqrt(dt)
        for i in range(n_assets):
            paths[:, t, i] = paths[:, t - 1, i] * np.exp((r - 0.5 * vols[i] ** 2) * dt + vols[i] * dW[:, i])
    return paths

# Basket Knockout Pricer
def price_knockout_basket_option(option_type, strike, maturity, knockout_barrier,
                                 spot_vec, vols, corr_mat, r, n_steps=252, n_paths=1000):
    paths = simulate_correlated_gbm(spot_vec, r, vols, corr_mat, maturity, n_steps, n_paths)
    basket_paths = paths.mean(axis=2)
    knocked_out = (basket_paths[:, :-1] > knockout_barrier).any(axis=1)
    final_price = basket_paths[:, -1]

    if option_type.lower() == 'call':
        payoffs = np.maximum(final_price - strike, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(strike - final_price, 0)
    else:
        raise ValueError("Option type must be Call or Put")

    payoffs[knocked_out] = 0
    discounted_payoff = np.exp(-r * maturity) * payoffs
    price = discounted_payoff.mean()
    return max(price, 0)

# Helper
def parse_maturity(maturity_str):
    if maturity_str.endswith('y'):
        return float(maturity_str[:-1])
    return float(maturity_str)

# Full Input Data (Id 1 to 36)
input_data = """Id,Asset,KnockOut,Maturity,Strike,Type
1,Basket,150,2y,50,Call
2,Basket,175,2y,50,Call
3,Basket,200,2y,50,Call
4,Basket,150,5y,50,Call
5,Basket,175,5y,50,Call
6,Basket,200,5y,50,Call
7,Basket,150,2y,100,Call
8,Basket,175,2y,100,Call
9,Basket,200,2y,100,Call
10,Basket,150,5y,100,Call
11,Basket,175,5y,100,Call
12,Basket,200,5y,100,Call
13,Basket,150,2y,125,Call
14,Basket,175,2y,125,Call
15,Basket,200,2y,125,Call
16,Basket,150,5y,125,Call
17,Basket,175,5y,125,Call
18,Basket,200,5y,125,Call
19,Basket,150,2y,75,Put
20,Basket,175,2y,75,Put
21,Basket,200,2y,75,Put
22,Basket,150,5y,75,Put
23,Basket,175,5y,75,Put
24,Basket,200,5y,75,Put
25,Basket,150,2y,100,Put
26,Basket,175,2y,100,Put
27,Basket,200,2y,100,Put
28,Basket,150,5y,100,Put
29,Basket,175,5y,100,Put
30,Basket,200,5y,100,Put
31,Basket,150,2y,125,Put
32,Basket,175,2y,125,Put
33,Basket,200,2y,125,Put
34,Basket,150,5y,125,Put
35,Basket,175,5y,125,Put
36,Basket,200,5y,125,Put"""

def price_basket_options_from_input(data_csv):
    df = pd.read_csv(io.StringIO(data_csv))
    spot_vec = np.array([spot_price] * 3)
    price_dict = {}

    for idx, row in df.iterrows():
        Id = int(row['Id'])
        option_type = row['Type']
        strike = float(row['Strike'])
        maturity = parse_maturity(row['Maturity'])
        knockout = float(row['KnockOut'])

        vols = np.array([implied_vols[(stock, maturity)] for stock in stocks])
        price = price_knockout_basket_option(
            option_type, strike, maturity, knockout,
            spot_vec, vols, corr_matrix, risk_free_rate
        )
        price_dict[Id] = round(price, 2)

    print("Id,Price")
    for i in range(1, 37):
        price = price_dict.get(i, 1)
        print(f"{i},{price}")

# Run the code
price_basket_options_from_input(input_data)