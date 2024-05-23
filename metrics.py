

def rsi(dataframe, close, n=14):
    df = dataframe.copy()
    df['rsi14'] = df[close].diff(1).mask(df[close].diff(1) < 0, 0).ewm(alpha=1 / n, adjust=False).mean().div(
        df[close].diff(1).mask(df[close].diff(1) > 0, -0.0).abs().ewm(alpha=1 / n, adjust=False).mean()).add(
        1).rdiv(100).rsub(100)
    return df


def stochastics(dataframe, low, high, close, k, d):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal
    When the %K crosses below %D, sell signal
    """

    df = dataframe.copy()

    # Set minimum low and maximum high of the k stoch
    low_min = df[low].rolling(window=k).min()
    high_max = df[high].rolling(window=k).max()

    # Fast Stochastic
    df['k_fast'] = 100 * (df[close] - low_min) / (high_max - low_min)
    df['k_fast'] = df['k_fast'].ffill()
    df['d_fast'] = df['k_fast'].rolling(window=d).mean()

    # Slow Stochastic
    df['k_slow'] = df["d_fast"]
    df['d_slow'] = df['k_slow'].rolling(window=d).mean()

    return df
