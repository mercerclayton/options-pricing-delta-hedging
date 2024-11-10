import pandas as pd
import numpy as np
import scipy.stats
import random
from pylab import cm, mpl, plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interactive, widgets
from IPython.display import display


class Option:
    """
    A class representing a European option (call or put) and its greeks, calculated using the Black-Scholes model.

    Parameters:
    -----------
    S : float
        Current stock price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying asset (annualized standard deviation).
    option_type : str
        Type of the option, either 'call' for a call option or 'put' for a put option.

    Methods:
    --------
    value() -> float:
        Calculates the theoretical price of the option (call or put).
    
    delta() -> float:
        Calculates the delta of the option, indicating sensitivity to the stock price.
    
    gamma() -> float:
        Calculates the gamma of the option, indicating sensitivity of delta to the stock price.
    
    vega() -> float:
        Calculates the vega of the option, indicating sensitivity to volatility.
    
    theta() -> float:
        Calculates the theta of the option, indicating sensitivity to time decay.
    
    rho() -> float:
        Calculates the rho of the option, indicating sensitivity to the interest rate.
    
    greeks() -> tuple:
        Returns a tuple containing all the option greeks (delta, gamma, vega, theta, rho).
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = float(S)      # Current stock price
        self.K = float(K)      # Strike price
        self.T = float(T)      # Time to expiration in years
        self.r = float(r)      # Risk-free interest rate
        self.sigma = float(sigma)  # Volatility of the underlying asset
        self.option_type = option_type.lower()  # Option type, either 'call' or 'put'

    @property
    def d1(self):
        """Calculate d1 for use in Black-Scholes formulas, handling edge cases for zero volatility or zero time."""
        if self.T <= 0 or self.sigma <= 0:
            return float('inf') if self.S > self.K else float('-inf')
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    @property
    def d2(self):
        """Calculate d2 for use in Black-Scholes formulas."""
        return self.d1 - self.sigma * np.sqrt(self.T)

    def value(self):
        """
        Calculates the theoretical price of the option.

        Returns:
        --------
        float
            The Black-Scholes value of the option (call or put).
        """
        if self.option_type == 'call':
            return (self.S * scipy.stats.norm.cdf(self.d1)) - (self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2))
        elif self.option_type == 'put':
            return (self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2)) - (self.S * scipy.stats.norm.cdf(-self.d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self):
        """
        Calculates the delta of the option.

        Returns:
        --------
        float
            The delta of the option, representing sensitivity to the stock price.
        """
        if self.option_type == 'call':
            return scipy.stats.norm.cdf(self.d1)
        elif self.option_type == 'put':
            return scipy.stats.norm.cdf(self.d1) - 1

    def gamma(self):
        """
        Calculates the gamma of the option.

        Returns:
        --------
        float
            The gamma of the option, representing sensitivity of delta to stock price changes.
        """
        return scipy.stats.norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """
        Calculates the vega of the option.

        Returns:
        --------
        float
            The vega of the option, representing sensitivity to volatility changes.
        """
        return self.S * scipy.stats.norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self):
        """
        Calculates the theta of the option.

        Returns:
        --------
        float
            The theta of the option, representing sensitivity to time decay.
        """
        term1 = -(self.S * scipy.stats.norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2)
            return term1 - term2
        elif self.option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2)
            return term1 + term2

    def rho(self):
        """
        Calculates the rho of the option.

        Returns:
        --------
        float
            The rho of the option, representing sensitivity to interest rate changes.
        """
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2)
        elif self.option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2)

    def greeks(self):
        """
        Calculates all the greeks of the option.

        Returns:
        --------
        tuple
            A tuple containing (delta, gamma, vega, theta, rho) of the option.
        """
        return self.delta(), self.gamma(), self.vega(), self.theta(), self.rho()


def plot_option(option, value, maturities, strikes, type='2d'):
    """
    Plots the option value or specified Greek for varying strikes and maturities.

    This function creates a plot to visualize either the value of an option or one of its Greeks 
    (Delta, Gamma, Vega, Theta, Rho) as a function of strike prices and maturities. The plot can 
    be displayed in either 3D (surface plot) or 2D (line plots for strikes and maturities).

    Parameters:
    -----------
    option : object
        An option object with attributes `T` (maturity) and `K` (strike) and methods `value()` 
        and `greeks()`. The `value()` method should return the option's value, and `greeks()` 
        should return an array where each index corresponds to a specific Greek.
    
    value : str
        A string indicating whether to plot the option value or a specific Greek. 
        Use 'C' to plot the option value or one of the following keys for Greeks:
        - 'D': Delta
        - 'G': Gamma
        - 'V': Vega
        - 'O': Theta
        - 'R': Rho

    maturities : list of floats
        A list of maturities for which the option will be evaluated.

    strikes : list of floats
        A list of strike prices for which the option will be evaluated.

    type : str, optional
        The type of plot to create. 
        - '3d': Creates a 3D surface plot of the option's value or Greek across strikes and maturities.
        - '2d': Creates 2D line plots of the option's value or Greek against strikes and maturities.
        The default is '2d'.
    
    Raises:
    -------
    TypeError
        If `type` is not '2d' or '3d', a TypeError is raised.

    Example:
    --------
    # Assume `option` is an instance of an Option class with necessary methods.
    plot_option(option, value='C', maturities=[1, 2, 3], strikes=[50, 100, 150], type='3d')
    
    """

    greeks = {'D': (0, 'Delta'), 'G': (1, 'Gamma'), 'V': (2, 'Vega'), 'O': (3, 'Theta'), 'R': (4, 'Rho')}
    K, T = np.meshgrid(strikes, maturities)
    v = np.zeros_like(T)
    for t_index, t_value in enumerate(maturities):
        for k_index, k_value in enumerate(strikes):
            option.T = t_value
            option.K = k_value
            v[t_index, k_index] = option.value() if value == 'P' else option.greeks()[greeks[value][0]]
    
    plot_label = f'{'Value' if value == 'P' else greeks[value][1]}'
    
    if type == '3d':
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(K, T, v, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel(plot_label)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    elif type == '2d':
        vt = []
        for t in maturities:
            option.T = t
            vt.append(option.value() if value == 'P' else option.greeks()[greeks[value][0]])

        vs = []
        for s in strikes:
            option.K = s
            vs.append(option.value() if value == 'P' else option.greeks()[greeks[value][0]])

        fig = plt.figure(figsize=(12, 7))
        plt.subplot(121)
        plt.plot(strikes, vs)
        plt.xlabel('Strike')
        plt.ylabel(plot_label)
        plt.subplot(122)
        plt.plot(maturities, vt)
        plt.xlabel('Maturity')
        plt.ylabel(plot_label)
    else:
        raise TypeError('The function can plot either 2D or 3D graphics.')


def delta_hedge(contracts, contract_size, sigma, r, k, weeks, option_type='call', **kwargs):
    """
    Delta hedges an option position (either call or put) over a series of weeks, 
    calculating the required stock purchases and costs.

    Parameters:
    -----------
    contracts : int
        The number of option contracts to hedge.
    
    contract_size : int
        The size of each contract, representing the number of shares per contract.
    
    sigma : float
        The volatility of the underlying stock (annualized standard deviation).
    
    r : float
        The risk-free interest rate, annualized.
    
    k : float
        The strike price of the option.
    
    weeks : int
        The duration of the delta hedge in weeks.
    
    option_type : str, optional
        Type of option, either 'call' or 'put'. Default is 'call'.
    
    **kwargs : dict, optional
        Additional optional parameters:
        - `stock_prices` : list or np.array of floats
            Predefined stock prices for each week; if not provided, random stock prices are generated.
        - `seed` : int
            Random seed for reproducibility of generated stock prices (if `stock_prices` not provided).
        - `upper` : float
            Upper limit for random stock price generation.
        - `lower` : float
            Lower limit for random stock price generation.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing weekly calculations for stock price, delta, shares owned, 
        shares bought/sold, cost of shares, cumulative cost including interest, and interest cost.
    
    df_styled : pd.io.formats.style.Styler
        Styled DataFrame for display, with formatted values for easier readability.
    """

    # Create DataFrame for values
    df = pd.DataFrame(columns=['Stock Price', 'Delta', 'Shares Owned', 'Shares Bought/Sold',
                               'Cost of Shares', 'Cumulative Cost Including Interest', 'Interest Cost'], 
                      index=range(weeks + 1))
    df.index.name = 'Week'

    # Assign stock price
    if 'stock_prices' in kwargs.keys():
        df['Stock Price'] = kwargs['stock_prices']
    else:
        np.random.seed(kwargs.get('seed', None))
        df['Stock Price'] = np.random.rand(weeks + 1) * (kwargs.get('upper', 150) - kwargs.get('lower', 100)) + kwargs.get('lower', 100)

    # Calculate Delta using Option class
    df['Delta'] = 0.0
    for week in df.index:
        # Calculate time to expiration in years
        T = (weeks - week) / 52
        
        # Create an instance of the Option class for each week
        option = Option(S=df.loc[week, 'Stock Price'], K=k, T=T, r=r, sigma=sigma, option_type=option_type)
        df.loc[week, 'Delta'] = option.delta()

    # Calculate shares owned to hedge the delta of the option position
    df['Shares Owned'] = df['Delta'] * contracts * contract_size

    # Calculate shares bought and sold and the associated cost
    df['Shares Bought/Sold'] = (df['Shares Owned'] - df['Shares Owned'].shift(1)).fillna(df['Shares Owned'])
    df['Cost of Shares'] = df['Shares Bought/Sold'] * df['Stock Price']

    # Initialize cumulative cost including interest
    df.loc[0, 'Cumulative Cost Including Interest'] = df.loc[0, 'Cost of Shares']
    df.loc[0, 'Interest Cost'] = df.loc[0, 'Cumulative Cost Including Interest'] * r * (1 / 52)
    
    # Calculate cumulative cost including weekly interest compounding
    for i in range(1, weeks + 1):
        df.loc[i, 'Cumulative Cost Including Interest'] = (
            df.loc[i - 1, 'Cumulative Cost Including Interest'] +
            df.loc[i - 1, 'Interest Cost'] +
            df.loc[i, 'Cost of Shares']
        )
        
        # Interest cost calculation for all weeks except the last one
        df.loc[i, 'Interest Cost'] = (
            df.loc[i, 'Cumulative Cost Including Interest'] * r * (1 / 52)
            if i < weeks else 0
        )

    # Scale large values for readability (in thousands)
    df[['Cost of Shares', 'Cumulative Cost Including Interest', 'Interest Cost']] /= 1000

    # Format output for easy readability
    df_styled = df.style.format(
        {
        'Stock Price': '${:,.2f}',
        'Delta': '{:.3f}',
        'Shares Owned': '{:,.0f}',
        'Shares Bought/Sold': '{:,.0f}',
        'Cost of Shares': '${:,.1f}',
        'Cumulative Cost Including Interest': '${:,.1f}',
        'Interest Cost': '${:,.1f}'
        }
    )
    df_styled.columns = {'Cost of Shares': 'Cost of Shares ($000s)',
                         'Cumulative Cost Including Interest': 'Cumulative Cost Including Interest ($000s)',
                         'Interest Cost': 'Interest Cost ($000s)'}

    return df, df_styled


def combined_payoff():
    """
    Generates an interactive options payoff diagram that combines different financial instrument payoffs.
    
    The function defines payoff calculations for various option and forward strategies, including:
    - Long and short call options
    - Long and short put options
    - Long and short forward contracts
    
    These payoffs can be visually combined and customized using interactive widgets for each instrument type.
    
    Functions:
    ----------
    - long_call(S, K): Calculates the payoff for a long call option.
    - short_call(S, K): Calculates the payoff for a short call option.
    - long_put(S, K): Calculates the payoff for a long put option.
    - short_put(S, K): Calculates the payoff for a short put option.
    - long_forward(S, K): Calculates the payoff for a long forward contract.
    - short_forward(S, K): Calculates the payoff for a short forward contract.

    Interactive Plot Parameters:
    ---------------------------
    - show_long_call (bool): Show/hide long call option.
    - show_short_call (bool): Show/hide short call option.
    - show_long_put (bool): Show/hide long put option.
    - show_short_put (bool): Show/hide short put option.
    - show_long_forward (bool): Show/hide long forward contract.
    - show_short_forward (bool): Show/hide short forward contract.
    
    Strike Prices and Underlying Bound:
    -----------------------------------
    - long_call_strike (float): Strike price for long call option.
    - short_call_strike (float): Strike price for short call option.
    - long_put_strike (float): Strike price for long put option.
    - short_put_strike (float): Strike price for short put option.
    - long_forward_price (float): Price for long forward contract.
    - short_forward_price (float): Price for short forward contract.
    - underlying_ub (float): Upper bound for underlying asset price range.

    Quantity Controls:
    -----------------
    - n_call (int): Quantity of long calls.
    - n_short_call (int): Quantity of short calls.
    - n_put (int): Quantity of long puts.
    - n_short_put (int): Quantity of short puts.
    - n_forward (int): Quantity of long forwards.
    - n_short_forward (int): Quantity of short forwards.
    
    Returns:
    -------
    Displays an interactive plot that allows users to visualize the combined payoff for the selected options and forward contracts.
    """
    
    # Payoff functions
    def long_call(S, K):
        return np.maximum(S - K, 0)
        
    def short_call(S, K):
        return -long_call(S, K)
    
    def long_put(S, K):
        return np.maximum(K - S, 0)
    
    def short_put(S, K):
        return -long_put(S, K)
    
    def long_forward(S, K):
        return (S - K)
    
    def short_forward(S, K):
        return -long_forward(S, K)
        
    def plot_combined_payoff(show_long_call, show_short_call, show_long_put, show_short_put, show_long_forward, show_short_forward,
                             long_call_strike, short_call_strike, long_put_strike, short_put_strike, long_forward_price, short_forward_price,
                             underlying_ub, n_call, n_short_call, n_put, n_short_put, n_forward, n_short_forward):
        
        underlying_price_range = np.linspace(0, underlying_ub, 100)
        combined_payoff = np.zeros_like(underlying_price_range)

        if show_long_call:
            combined_payoff += n_call * long_call(underlying_price_range, long_call_strike)
        if show_short_call:
            combined_payoff += n_short_call * short_call(underlying_price_range, short_call_strike)
        if show_long_put:
            combined_payoff += n_put * long_put(underlying_price_range, long_put_strike)
        if show_short_put:
            combined_payoff += n_short_put * short_put(underlying_price_range, short_put_strike)
        if show_long_forward:
            combined_payoff += n_forward * long_forward(underlying_price_range, long_forward_price)
        if show_short_forward:
            combined_payoff += n_short_forward * short_forward(underlying_price_range, short_forward_price)

        # Plot the payoff diagram
        plt.figure(figsize=(18, 6))
        plt.plot(underlying_price_range, combined_payoff, label='Combined Payoff')
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title('Combined Option and Underlying Asset Payoff Diagram')
        plt.xlabel('Stock Price at Expiration')
        plt.ylabel('Profit / Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Interactive controls for the diagram
    show_long_call_widget = widgets.Checkbox(
        value=True,
        description='Show Long Call Option',
    )

    show_short_call_widget = widgets.Checkbox(
        value=True,
        description='Show Short Call Option',
    )

    show_long_put_widget = widgets.Checkbox(
        value=True,
        description='Show Long Put Option',
    )

    show_short_put_widget = widgets.Checkbox(
        value=True,
        description='Show Short Put Option',
    )

    show_long_forward_widget = widgets.Checkbox(
        value=True,
        description='Show Long Forward',
    )

    show_short_forward_widget = widgets.Checkbox(
        value=True,
        description='Show Short Forward',
    )

    long_call_strike_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Long Call Strike:',
    )

    short_call_strike_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Short Call Strike:',
    )

    long_put_strike_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Long Put Strike:',
    )

    short_put_strike_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Short Put Strike:',
    )

    long_forward_price_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Long Forward Price:',
    )

    short_forward_price_widget = widgets.FloatSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Short Forward Price:',
    )

    underlying_ub_widget = widgets.FloatSlider(
        value=100,
        min=10,
        max=200,
        step=1,
        description='Underlying Price Upper Bound:',
    )

    n_call_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Long Calls:',
    )

    n_short_call_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Short Calls:',
    )

    n_put_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Long Puts:',
    )

    n_short_put_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Short Puts:',
    )

    n_forward_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Long Forwards:',
    )

    n_short_forward_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Number of Short Forwards:',
    )

    interactive_plot = interactive(plot_combined_payoff, 
                                   show_long_call=show_long_call_widget, 
                                   show_short_call=show_short_call_widget,
                                   show_long_put=show_long_put_widget,
                                   show_short_put=show_short_put_widget,
                                   show_long_forward=show_long_forward_widget,
                                   show_short_forward=show_short_forward_widget,
                                   long_call_strike=long_call_strike_widget, 
                                   short_call_strike=short_call_strike_widget, 
                                   long_put_strike=long_put_strike_widget, 
                                   short_put_strike=short_put_strike_widget, 
                                   long_forward_price=long_forward_price_widget, 
                                   short_forward_price=short_forward_price_widget, 
                                   underlying_ub=underlying_ub_widget,
                                   n_call=n_call_widget, 
                                   n_short_call=n_short_call_widget, 
                                   n_put=n_put_widget, 
                                   n_short_put=n_short_put_widget, 
                                   n_forward=n_forward_widget, 
                                   n_short_forward=n_short_forward_widget)
    display(interactive_plot)

