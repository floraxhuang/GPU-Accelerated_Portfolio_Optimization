import numpy as np
import pandas as pd
from time import time
from os import path, environ
from juliacall import Main as jl

# For sparse matrix generation
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix, diags

# --- Julia Environment Setup ---
# Set environment variables before importing Julia code
environ['PYTHON_JULIACALL_THREADS'] = 'auto'      # Use available threads
environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
environ['PYTHON_JULIACALL_OPTIMIZE'] = '3'        # Max optimization level
environ['PYTHON_JULIACALL_COMPILE'] = 'yes'       # Precompile Julia code called
environ['JULIA_CPU_TARGET'] = 'native'            # Optimize for local CPU
environ['JULIA_PKG_PRECOMPILE_AUTO'] = '1'        # Auto precompile packages

# --- Load Julia Code ---
julia_script_path = 'Mean-Variance-conic.jl'
assert path.exists(julia_script_path), f"Julia script not found: {julia_script_path}"
jl.include(julia_script_path)
print("Julia code included successfully.")

# --- Determine Float Type ---
# Match Python's float type to Julia's MyFloat for consistency
if jl.MyFloat == jl.Float64:
    MyFloat = np.float64
    print("Using Float64.")
else:
    MyFloat = np.float32
    print("Using Float32.")

# --- Python Wrapper Class for Julia Optimizer ---
class PortfolioOptimizer:
    """
    A Python wrapper for the Julia PortfolioModel.
    Handles model creation, setup, solving, and updates.
    """
    def __init__(self, n_assets, lbd_risk, leverage_limit, c_min, c_max):
        """
        Initializes the optimizer parameters.

        Args:
            n_assets (int): Number of assets.
            lbd_risk (float): Risk aversion parameter.
            leverage_limit (float): Max turnover leverage (sum|x_i - x0_i|).
            c_min (float): Minimum cash holding.
            c_max (float): Maximum cash holding.
        """
        self.n_assets = n_assets
        self.lbd_risk = MyFloat(lbd_risk)            # Ensure correct float type
        self.leverage_limit = MyFloat(leverage_limit)
        self.c_min = MyFloat(c_min)
        self.c_max = MyFloat(c_max)
        self.pm = None                          
        print(f"Optimizer initialized for {n_assets} assets.")

    def setup_model(self, x0, cov, cost, u_cpu):
        """
        Creates and sets up the Julia optimization model.

        Args:
            x0 (np.ndarray): Initial portfolio weights.
            cov (np.ndarray): Asset covariance matrix.
            cost (np.ndarray): Transaction costs per asset.
            u_cpu (np.ndarray): Initial expected returns vector (mu).
        """
        if self.pm is not None:
            print("Warning: Model already exists. Re-creating.")

        print("Creating Julia PortfolioModel...")
        self.pm = jl.CreatePortfolioModel(
            self.n_assets, self.lbd_risk,
            x0.astype(MyFloat),      # py_x0
            cov.astype(MyFloat),     # py_cov
            cost.astype(MyFloat),    # py_cost
            u_cpu.astype(MyFloat),   # py_u_cpu
            self.leverage_limit,     # leverage_limit
            self.c_min,              # c_min
            self.c_max               # c_max
        )
        print("Setting up Julia model constraints and objective...")
        # Call the Julia setup function
        jl.setup_model_b(self.pm)
        print("Model setup complete.")

    def initial_solve(self):
        """
        Performs the initial solve (warm-up) of the Julia model.
        This includes JIT compilation time.
        """
        if self.pm is None:
            raise RuntimeError("Model not set up. Call setup_model first.")
        print("Performing initial solve (warm-up)...")
        # Call the Julia initial solve function
        jl.initial_solve_b(self.pm)
        print("Initial solve finished.")
        try:
            # Check status after solve
            status = jl.JuMP.termination_status(self.pm.model)
            if status != jl.MOI.OPTIMAL and status != jl.MOI.ALMOST_OPTIMAL:
                 print("Warning: Initial solve did not reach optimality.")
        except Exception as e:
            print(f"Could not get initial solve status: {e}")

    def resolve(self, new_u_cpu):
        """
        Updates the expected returns and re-solves the model.
        Measures the time for the fast, post-compilation solve.

        Args:
            new_u_cpu (np.ndarray): New expected returns vector.
        """
        if self.pm is None:
            raise RuntimeError("Model not set up. Call setup_model first.")

        # Update return ratio in the Julia model object
        jl.update_return_ratio_b(self.pm, new_u_cpu.astype(MyFloat))

        # Re-solve the model
        jl.resolve_b(self.pm)

        try:
            # Check status after solve
            status = jl.JuMP.termination_status(self.pm.model)
            if status != jl.MOI.OPTIMAL and status != jl.MOI.ALMOST_OPTIMAL:
                 print(f"Warning: Resolve did not reach optimality. Status: {status}")
        except Exception as e:
            print(f"Could not get resolve solve status: {e}")

    def get_risk(self):
        """Gets the calculated risk (standard deviation) from the solved model."""
        if self.pm is None:
            raise RuntimeError("Model not set up.")
        try:
            return jl.get_risk(self.pm)
        except Exception as e:
            print(f"Error getting risk: {e}")
            return np.nan

    def get_objective_value(self):
        """Gets the objective value from the solved model."""
        if self.pm is None:
            raise RuntimeError("Model not set up.")
        try:
            return jl.JuMP.objective_value(self.pm.model)
        except Exception as e:
            print(f"Error getting objective value: {e}")
            return np.nan

# --- Data Loading and Preparation ---
def load_sp500_data(filepath='sp500.csv', lookback_window=252):
    """
    Loads SP500 price data, calculates returns, covariance, and expected returns.

    Args:
        filepath (str): Path to the CSV file.
        lookback_window (int): Window size for covariance/return calculation.

    Returns:
        tuple: Contains:
            - pd.DataFrame: Daily returns.
            - np.ndarray: Covariance matrix.
            - np.ndarray: Expected returns (mean of lookback window).
            - list: Asset tickers/names.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None, None, None

    # Ensure data is numeric, handle potential non-numeric columns if necessary
    # df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='any') # Drop columns with NaNs

    if df.empty:
        print("Error: No valid numeric data found after cleaning.")
        return None, None, None, None

    print(f"Calculating daily log returns for {len(df.columns)} assets...")
    returns = np.log(df / df.shift(1)).dropna()

    if returns.empty or len(returns) < lookback_window:
         print(f"Error: Not enough data for lookback window {lookback_window}. Need {lookback_window}, have {len(returns)}.")
         return None, None, None, None

    print(f"Calculating covariance matrix (window={lookback_window})...")
    cov_matrix = returns.iloc[-lookback_window:].cov().values * 252        # Annualize covariance

    print(f"Calculating expected log returns (mean of window={lookback_window})...")
    expected_returns = returns.iloc[-lookback_window:].mean().values * 252 # Annualize returns

    tickers = df.columns.tolist()
    print("Data preparation finished.")
    return returns, cov_matrix, expected_returns, tickers

# --- Synthetic Sparse Data Generation ---
def generate_sparse_mvo_data(n_assets=2000, n_sectors=20, avg_sector_size=100, seed=42):
    """
    Generates synthetic data with a block-diagonal sparse covariance matrix.

    Args:
        n_assets (int): Total number of assets.
        n_sectors (int): Number of sectors (blocks).
        avg_sector_size (int): Average number of assets per sector.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Contains:
            - np.ndarray: Sparse covariance matrix (ensure PSD).
            - np.ndarray: Expected returns vector.
            - np.ndarray: Transaction costs vector.
            - np.ndarray: Initial weights vector.
            - int: Number of assets.
    """
    print(f"\n--- Generating Synthetic Sparse Data ---")
    print(f"Target Assets: {n_assets}, Sectors: {n_sectors}")
    np.random.seed(seed)

    # Distribute assets into sectors and can be uneven
    sector_sizes = np.random.poisson(avg_sector_size, n_sectors)
    # Adjust sizes to approximately match n_assets
    current_total = np.sum(sector_sizes)
    if current_total > 0:
        sector_sizes = (sector_sizes * (n_assets / current_total)).astype(int)
        # Ensure total is exactly n_assets, add/remove difference to largest sector
        diff = n_assets - np.sum(sector_sizes)
        sector_sizes[np.argmax(sector_sizes)] += diff
    else: # Handle case where initial sum is 0
        sector_sizes = np.full(n_sectors, n_assets // n_sectors)
        sector_sizes[0] += n_assets % n_sectors # Add remainder to first sector

    n_assets = np.sum(sector_sizes)
    print(f"Actual Assets: {n_assets}")

    sector_cov_blocks = []
    min_diag_val = 1e-6 # To ensure positive definiteness

    print("Generating sector covariance blocks...")
    for size in sector_sizes:
        if size <= 0: continue
        # Generate a random PSD matrix for the sector
        A = np.random.randn(size, size) * 0.1 # Scale down random part
        sector_cov = np.dot(A, A.T) + np.eye(size) * np.random.uniform(min_diag_val, 0.05, size) # Add random diagonal variance
        # Ensure diagonal is sufficiently positive
        np.fill_diagonal(sector_cov, np.maximum(np.diag(sector_cov), min_diag_val))
        sector_cov_blocks.append(sector_cov)

    print("Assembling block-diagonal covariance matrix...")
    # Create the block diagonal matrix (dense numpy array)
    cov_matrix_dense = block_diag(*sector_cov_blocks)

    print("Generating expected returns and costs...")
    # Generate random expected returns (annualized)
    expected_returns = (np.random.randn(n_assets) * 0.15 + 0.05).astype(MyFloat) # Mean 5%, std dev 15%

    # Generate costs
    cost = np.full(n_assets, 0.0005, dtype=MyFloat) # Example transaction cost (0.05%)

    # Generate initial weights
    x0 = np.full(n_assets, 1.0/n_assets, dtype=MyFloat)

    print("Synthetic data generation finished.")
    return cov_matrix_dense, expected_returns, cost, x0, n_assets


# --- Benchmarking Function ---
def run_benchmark(num_resolves=50, lookback=252, use_synthetic_data=False, n_assets_synth=2000, n_sectors_synth=20):
    """
    Runs the portfolio optimization benchmark.

    Args:
        num_resolves (int): Number of re-solve steps to time.
        lookback (int): Lookback window (only for real data).
        use_synthetic_data (bool): If True, generate sparse data. Otherwise load SP500.
        n_assets_synth (int): Number of assets for synthetic data.
        n_sectors_synth (int): Number of sectors for synthetic data.
    """
    print("--- Starting Benchmark ---")
    # Initialize variables
    n_assets = 0
    resolve_mus = []
    cov = None
    initial_mu = None
    cost = None
    x0 = None
    
    if use_synthetic_data:
        cov, initial_mu, cost, x0, n_assets = generate_sparse_mvo_data(
            n_assets=n_assets_synth, n_sectors=n_sectors_synth
        )
        # For synthetic data, we need a sequence of returns for the resolve loop
        print(f"Generating {num_resolves} sets of random returns for benchmark loop...")
        resolve_mus = [ (np.random.randn(n_assets) * 0.15 + 0.05).astype(MyFloat) for _ in range(num_resolves) ]

    else:
        # 1. Load and Prepare Real Data using log returns
        all_returns, cov, initial_mu, tickers = load_sp500_data(lookback_window=lookback)
        if all_returns is None:
            print("Failed to load or process data. Exiting benchmark.")
            return

        if len(all_returns) < lookback:
             print(f"Error: Not enough historical return data points for initial calculation. Need {lookback}, have {len(all_returns)}.")
             return

        n_assets = len(tickers)                               
        cost = np.full(n_assets, 0.0005, dtype=MyFloat)
        initial_asset_weight = 1.0 / n_assets
        x0 = np.full(n_assets, initial_asset_weight, dtype=MyFloat)

        # Adjust num_resolves if not enough data
        available_resolves = len(all_returns) - lookback
        if available_resolves < num_resolves:
            print(f"Warning: Not enough historical return data points for all resolves. Need {lookback + num_resolves}, have {len(all_returns)}. Will run {available_resolves} resolves.")
            num_resolves = available_resolves # Adjust number of resolves

        print(f"Preparing {num_resolves} expected return vectors for benchmark loop...")
        for i in range(num_resolves):
            # The window for mu ends at index: lookback + i (relative to start of all_returns)
            window_end_idx = lookback + i
            window_start_idx = window_end_idx - lookback

            current_returns = all_returns.iloc[window_start_idx : window_end_idx + 1]

            resolve_mus.append(current_returns.mean().values * 252)


    print(f"Number of assets being optimized: {n_assets}")
    if n_assets == 0:
        print("Error: No assets available.")
        return
    if len(resolve_mus) < num_resolves:
        print(f"Warning: Only {len(resolve_mus)} resolve steps possible with available data/settings.")
        num_resolves = len(resolve_mus)

    # Define other parameters
    lbd_risk = 10.0                                 # risk aversion
    leverage_limit = 0.5                            # turnover limit (sum|x_i - x0_i|)
    c_min = 0.2                                     # min cash holdings
    c_max = 0.8                                     # max cash holdings
    cost = np.full(n_assets, 0.0005, dtype=MyFloat) # transaction cost (0.05%)
    # Initial weights - ensure they sum to <= 1 to allow for initial cash if c_min >= 0
    initial_asset_weight = 1.0 / n_assets
    x0 = np.full(n_assets, initial_asset_weight, dtype=MyFloat)

    # 2. Initialize Optimizer
    optimizer = PortfolioOptimizer(n_assets, lbd_risk, leverage_limit, c_min, c_max)

    # 3. Setup Model & Initial Solve (Warm-up)
    optimizer.setup_model(x0, cov, cost, initial_mu)
    optimizer.initial_solve() # Includes JIT compilation

    # Store initial results if needed
    initial_risk = optimizer.get_risk()
    initial_obj = optimizer.get_objective_value()
    print(f"Initial Risk (Std Dev of Log Returns): {initial_risk:.6f}")
    print(f"Initial Objective: {initial_obj:.6f}")

    # 4. Benchmarking Loop (Re-solve)
    print(f"\n--- Benchmarking {num_resolves} Re-solve Steps ---")
    resolve_times = []
    start_total_time = time()

    for i in range(num_resolves):
        new_mu = resolve_mus[i].astype(MyFloat)

        step_start_time = time()
        optimizer.resolve(new_mu)
        step_end_time = time()
        resolve_times.append(step_end_time - step_start_time)

        if (i + 1) % 10 == 0:
             print(f"Completed step {i+1}/{num_resolves}...")

    end_total_time = time()
    total_resolve_time = end_total_time - start_total_time

    # 5. Report Results
    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_resolves} resolves: {total_resolve_time:.4f} seconds")
    if num_resolves > 0:
        avg_resolve_time = total_resolve_time / num_resolves
        print(f"Average time per resolve: {avg_resolve_time:.6f} seconds ({avg_resolve_time * 1000:.3f} ms)")
    else:
        print("No resolve steps were timed.")

    # Print final state
    final_risk = optimizer.get_risk()
    final_obj = optimizer.get_objective_value()
    print(f"\nFinal Risk (Std Dev of Log Returns): {final_risk:.6f}")
    print(f"Final Objective: {final_obj:.6f}")

if __name__ == "__main__":
    # --- Configuration ---
    USE_SYNTHETIC = True # Set to True to use generated sparse data
    # Synthetic data parameters (if USE_SYNTHETIC is True)
    N_ASSETS_SYNTH = 10000
    N_SECTORS_SYNTH = 100
    NUM_RESOLVES = 3
    LOOKBACK_WINDOW = 252

    if USE_SYNTHETIC:
        run_benchmark(
            num_resolves=NUM_RESOLVES,
            use_synthetic_data=True,
            n_assets_synth=N_ASSETS_SYNTH,
            n_sectors_synth=N_SECTORS_SYNTH
        )
    else:
        run_benchmark(
            num_resolves=NUM_RESOLVES,
            lookback=LOOKBACK_WINDOW,
            use_synthetic_data=False
        )