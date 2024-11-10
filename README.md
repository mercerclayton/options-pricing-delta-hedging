# Options Analysis Toolkit

This Python module, `options_analysis_toolkit.py`, offers a suite of tools for analyzing European options and managing options portfolios. It uses the Black-Scholes model to compute option pricing and "Greeks" (sensitivities), as well as providing utilities for visualizing options payoffs, hedging strategies, and simulating portfolio outcomes.

## Features

- **Option Pricing and Greeks Calculation**: Calculates the theoretical price of European call and put options, along with sensitivities (Delta, Gamma, Vega, Theta, Rho) using the Black-Scholes model.
- **Interactive Payoff Visualizations**: Generates interactive payoff diagrams for various option and forward strategies, allowing you to visually assess combined positions.
- **Delta Hedging**: A delta-hedging function that simulates weekly adjustments and calculates associated costs over time.
- **3D and 2D Visualization of Greeks and Option Values**: Visualize option values or Greeks across a range of strike prices and maturities in either 2D or 3D.

## How to Use

1. **Installation**: Clone or download this repository.
2. **Dependencies**: Install required packages with the following command:
   ```bash
   pip install numpy pandas scipy matplotlib ipywidgets
   ```

## Code Overview

- **Option Class**: The core class for representing European options, including methods for computing option values and sensitivities.
- **plot_option()**: Visualizes option values or Greeks in 2D or 3D across varying strikes and maturities.
- **delta_hedge()**: Calculates weekly adjustments and associated costs for delta hedging an option position over time.
- **combined_payoff()**: Creates an interactive options payoff diagram, allowing for various combinations of call/put options and forward contracts.

## Example Usage

```python
from options_analysis_toolkit import Option, plot_option, delta_hedge, combined_payoff

# Define an option
option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

# Calculate option value and Greeks
print("Option Value:", option.value())
print("Option Greeks:", option.greeks())

# Plot Greeks across strike prices and maturities
plot_option(option, value='C', maturities=[0.5, 1.0, 1.5], strikes=[80, 100, 120], type='3d')
```
