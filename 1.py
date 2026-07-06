import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def main():
    # Step 1: Read portfolio P&L from stdin
    line = sys.stdin.read().strip()
    parts = line.split()
    portfolio_id = parts[0]
    pnl = np.array(list(map(float, parts[1:])))  # shape (250,)

    # Step 2: Read stocks metadata to get stock costs
    metadata_df = pd.read_csv("stocks_metadata.csv")
    stock_costs = dict(zip(metadata_df['Stock_Id'], metadata_df['Capital_Cost']))

    # Step 3: Read historical returns and drop 'date' column
    returns_df = pd.read_csv("stocks_returns.csv")/100
    returns_df = returns_df.drop(columns=['Date'])
    stock_ids = returns_df.columns.tolist()

    # Step 4: Convert returns to numpy matrix (shape: days x stocks)
    returns_matrix = returns_df.values
    n_days, n_stocks = returns_matrix.shape

    # Check if pnl length matches returns
    if len(pnl) != n_days:
        raise ValueError(f"Portfolio P&L length {len(pnl)} does not match returns days {n_days}")

    # Step 5: Prepare cost vector aligned with stock order
    cost_vector = np.array([stock_costs.get(stock, 1e6) for stock in stock_ids])  # large cost if missing

    # Step 6: Define optimization problem:
    # Objective: minimize variance of residual pnl (portfolio pnl - hedge pnl)
    # plus a small penalty on total cost (to keep hedge cost low)
    # Variables: weights (quantities) of stocks

    # Define function to minimize
    def objective(weights):
        hedge_pnl = returns_matrix @ weights  # shape (days,)
        residual = pnl + hedge_pnl
        var_residual = np.var(residual)
        cost = np.sum(np.abs(weights) * cost_vector)
        penalty = 1e-3   # small weight for cost penalty; tweak as needed
        return var_residual + penalty * cost

    # Optional: bounds on weights (e.g., limit max position size)
    bounds = [(-1e5, 1e5) for _ in range(n_stocks)]

    # Initial guess: zero weights
    x0 = np.zeros(n_stocks)

    # Run optimizer
    result = minimize(objective, x0, method='SLSQP',bounds=bounds, options={'maxiter': 100})

    

    if not result.success:
        print("Optimization failed:", result.message, file=sys.stderr)

    weights_opt = result.x

    # Step 7: Output stock_id and rounded quantities (int)
    # Only output stocks with non-zero weights (above a small threshold)
    threshold = 1
    for stock, qty in zip(stock_ids, weights_opt):
        qty_int = int(round(qty))
        if abs(qty_int) >= threshold:
            print(f"{stock} {qty_int}")

if __name__ == "__main__":
    main()
