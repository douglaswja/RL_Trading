import os
import pathlib
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import torch

from torch import nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from scipy.optimize import minimize_scalar

from LOB_analysis import *

MARKET_OPEN = '9:30'
MARKET_CLOSE = '16:00'

def load_message(filename):
    """
    filename: csv file
    
        Event Type:
        1: Submission of a new limit order
        2: Cancellation (partial deletion of a limit order)
        3: Deletion (total deletion of a limit order)
        4: Execution of a visible limit order
        5. direction:  
            +1: indicats: BUY order // seller initiated trade (a seller takes the initiative to consume some of the quoted offers at the BID// The volume at the BID decreases 
            -1: indicates: SELL order // buyer initiated trade (a buyer takes the initiative to consume some of the quoted offers at the ASK// The volume at the ASK decreases 
        6: Indicates a cross trade, e.g. auction trade
        7: Trading halt indicator (detailed information below)
    """
    message = pd.read_csv(filename, header=None, low_memory=False)
    message = message.drop(columns=[6])
    message = message.rename(columns={0:"time", 1:"type", 2: "id", 3: "vol", 4: "price", 5:"direct"})
    return message

def load_LOB(filename):
    """
    filename: csv file
    """
    #load data
    LOB = pd.read_csv(filename, header=None, low_memory=False)
    
    #rename columns
    kont = 0
    level = 1
    dico = {}
    n_levels = len(LOB.columns)//4
    for _ in range(n_levels):
        dico[kont] = f"ask_price_{level}"
        kont += 1
        dico[kont] = f"ask_size_{level}"
        kont += 1
        dico[kont] = f"bid_price_{level}"
        kont += 1
        dico[kont] = f"bid_size_{level}"
        kont += 1
        level += 1
    LOB = LOB.rename(columns=dico)
    return LOB


#####
##### Feature engineering
#####
def get_base_features(message, orderbook, weights = [0.7, 0.2, 0.05, 0.03, 0.02]):
    data = message.drop(['id'], axis=1).copy()
    
    data.loc[:, 'bid'] = orderbook.bid_price_1
    data.loc[:, 'ask'] = orderbook.ask_price_1
    data.loc[:, 'spread'] = data.ask - data.bid
    data.loc[:, 'mid_price'] = (data.ask + data.bid) / 2
    data.loc[:, 'mid_price_2'] = (orderbook.ask_price_2 + orderbook.bid_price_2) / 2
    data.loc[:, 'mid_price_3'] = (orderbook.ask_price_3 + orderbook.bid_price_3) / 2
    data.loc[:, 'log_return'] = np.log(data.mid_price) - np.log(data.mid_price).shift(1)
    
    # Order book imbalance
    data.loc[:, 'weighted_ask_liquidity'] = 0
    data.loc[:, 'weighted_buy_liquidity'] = 0
    weights = [0.7, 0.2, 0.05, 0.03, 0.02]
    for i in range(1, len(weights) + 1):
        data.loc[:, 'weighted_ask_liquidity'] += orderbook.loc[:, f'ask_price_{i}'] * orderbook.loc[:, f'ask_size_{i}'] * weights[i-1]
        data.loc[:, 'weighted_buy_liquidity'] += orderbook.loc[:, f'bid_price_{i}'] * orderbook.loc[:, f'bid_size_{i}'] * weights[i-1]
    data.loc[:, 'weighted_ask_liquidity'] = data.loc[:, 'weighted_ask_liquidity'].replace(0, np.nan)
    data.loc[:, 'order_book_imbalance'] = np.log(data.loc[:, 'weighted_buy_liquidity']) - np.log(data.loc[:, 'weighted_ask_liquidity'])
    
    # Weighted total volume
    data.loc[:, 'weighted_ask_volume'] = 0
    data.loc[:, 'weighted_buy_volume'] = 0
    for i in range(1, len(weights) + 1):
        data.loc[:, 'weighted_ask_volume'] += orderbook.loc[:, f'ask_size_{i}'] * weights[i-1]
        data.loc[:, 'weighted_buy_volume'] += orderbook.loc[:, f'bid_size_{i}'] * weights[i-1]
    
    return data.reset_index(drop=True)


def get_interval_insight_features(base_data, granularity = '250 ms'):
    base_group = base_data.groupby(base_data.time.dt.round(granularity))
    
    # Data for start-of-period
    interval_data = base_group.first().loc[:, ['bid', 'ask', 'spread', 'mid_price', 'mid_price_2', 'mid_price_3']]
    
    # Aggregate data
    interval_data = interval_data.merge(base_group.agg({
        'vol': np.sum,
        'log_return': np.sum,
        'weighted_ask_liquidity': np.mean,
        'weighted_buy_liquidity': np.mean,
        'order_book_imbalance': np.mean,
        'weighted_ask_volume': np.mean,
        'weighted_buy_volume': np.mean,
    }), left_index=True, right_index=True)
    
    # Counts
    interval_data = interval_data.merge(base_group.size().fillna(0).rename('interval_count'), left_index=True, right_index=True)
    interval_data = interval_data.merge(base_group.direct.value_counts().unstack().rename(columns = {-1: 'interval_sell_count', 1: 'interval_buy_count'}).loc[:, 'interval_buy_count'].fillna(0).rename('interval_buy_count'),
                        left_index=True,
                        right_index=True)
    # OHLC, STD
    interval_data = interval_data.merge(base_group.mid_price.agg([
            lambda x: x.iloc[0],   # Open
            np.max,                # High
            np.min,                # Low
            lambda x: x.iloc[-1],  # Close
            np.std,                # Std
        ]).rename({'<lambda_0>': 'open', 'amax': 'high', 'amin': 'low', '<lambda_1>': 'close', 'std': 'mid_price_std'}, axis=1).fillna(0), left_index=True, right_index=True)
    
    # Volume Weighted Average Price
    vwap = ((base_data.price * base_data.vol).groupby(base_data.time.dt.round(granularity)).sum() / base_data.vol.groupby(base_data.time.dt.round(granularity)).sum()).rename('vwap')
    interval_data = interval_data.merge(vwap, left_index=True, right_index=True)
    
    # Flow Quantity
    flow_qty = base_data.vol.groupby([base_data.time.dt.round(granularity), base_data.direct]).sum().unstack().fillna(1).apply(np.log).loc[:, [-1, 1]].diff(axis=1).loc[:, 1].rename('flow_qty')
    interval_data = interval_data.merge(flow_qty, left_index=True, right_index=True)
    
    # Standard deviation
    std = base_group.std().loc[:, ['spread', 'log_return', 'weighted_ask_liquidity', 'weighted_buy_liquidity', 'order_book_imbalance', 'weighted_ask_volume', 'weighted_buy_volume']].copy().fillna(0).add_suffix("_std")
    interval_data = interval_data.merge(std, left_index=True, right_index=True)
    
    return interval_data


def calc_momentum(interval_data, shift=10):
    momentum = (interval_data.mid_price - interval_data.mid_price.shift(shift)).rename(f'mid_price_momentum_{shift}')
    return momentum


def calc_williams_pct_r(interval_data, interval=14):
    mid_price_roll = interval_data.mid_price.rolling(interval)
    williams_pct_r = -100 * ((mid_price_roll.max() - interval_data.mid_price) / (mid_price_roll.max() - mid_price_roll.min())).rename('williams_pct_r')
    return williams_pct_r.fillna(-50) # NAs are introduced when dividing by 0, treat these as middle values from Williams %R range of [-100, 0]


def calculate_macd_signal(interval_data, fast_ma=12, slow_ma=26, signal_ma=10):
    macd = (interval_data.mid_price.ewm(fast_ma).mean() - interval_data.mid_price.ewm(slow_ma).mean()).rename('macd')
    signal_line = macd.rolling(signal_ma).mean().rename('signal_line')
    return macd.to_frame().merge(signal_line, left_index=True, right_index=True)


def calculate_pvt(interval_data):
    pvt = (((interval_data.close - interval_data.close.shift(1)) / interval_data.close.shift(1)) * interval_data.vol).cumsum().rename('pvt')
    return pvt

# This function adds 24 columns
# It can be improved by not spanning across days, i.e. 9:30am data should utilise previous days interval features
def get_interval_features(interval_data):
    interval_data = interval_data.merge(calc_momentum(interval_data), left_index=True, right_index=True)
    interval_data = interval_data.merge(calc_williams_pct_r(interval_data), left_index=True, right_index=True)
    interval_data = interval_data.merge(calculate_macd_signal(interval_data), left_index=True, right_index=True)
    interval_data = interval_data.merge(calculate_pvt(interval_data), left_index=True, right_index=True)
    
    roc_columns = ['spread', 'mid_price', 'log_return', 'weighted_ask_liquidity', 'weighted_buy_liquidity', 'order_book_imbalance', 'flow_qty', 'mid_price_std', 'weighted_ask_volume', 'weighted_buy_volume']
    
    # Rate of change: Velocity
    position = interval_data.loc[:, roc_columns].copy().add_suffix("_unit_per_s")
    time = pd.Series(position.index, index=position.index)
    seconds = (time - time.shift(1)).dt.total_seconds()
    displacement = position - position.shift(1)
    velocity = displacement.div(seconds, axis=0)
    interval_data = interval_data.merge(velocity, left_index=True, right_index=True)
    
    # Rate of change: Acceleration
    velocity = velocity.copy().add_suffix("_per_s")
    time = pd.Series(velocity.index, index=velocity.index)
    seconds = (time - time.shift(1)).dt.total_seconds()
    velocity_diff = velocity - velocity.shift(1)
    acceleration = velocity_diff.div(seconds, axis=0)
    interval_data = interval_data.merge(acceleration, left_index=True, right_index=True)
    
    return interval_data


def standardize_previous(interval_data, standardize_offset = 1):
    tmp = interval_data.reset_index()
    year_day = tmp.set_index([tmp.time.dt.year, tmp.time.dt.dayofyear - standardize_offset]) # -1 to normalize by previous day
    groupby = tmp.groupby([tmp.time.dt.year, tmp.time.dt.dayofyear])

    mean = groupby.mean()
    year_day_mean = mean[mean.index.isin(year_day.index)]

    std = groupby.std()
    year_day_std = std[std.index.isin(year_day.index)]

    standardized = ((year_day - year_day_mean) / year_day_std)
    standardized = standardized.reset_index(drop=True)
    standardized.loc[:, 'time'] = tmp.time
    return standardized.set_index('time')

"""
Assumes that all the data provided occurs after the market open.
Discards data outside of market hours.
"""
def get_time_features(interval_data):
    interval_data = interval_data.between_time(MARKET_OPEN, MARKET_CLOSE)
    idx = pd.Series(interval_data.index)
    interval_data = interval_data.reset_index(drop=True)
    
    # Seconds since Market Open
    seconds_since_MO = ((idx - pd.Timedelta(hours=int(MARKET_OPEN.split(':')[0]), minutes=int(MARKET_OPEN.split(':')[1]))).dt.time).apply(
        lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()  # Convert the timestamp to seconds
    ).rename("seconds_since_MO").reset_index(drop=True)
    interval_data = interval_data.merge(seconds_since_MO, left_index=True, right_index=True)
    
    # Day of week
    day_of_week = idx.dt.day_of_week.rename("day_of_week").reset_index(drop=True)
    interval_data = interval_data.merge(day_of_week, left_index=True, right_index=True)
    
    # Week of year
    week_of_year = idx.dt.isocalendar().week.rename("week_of_year")
    interval_data = interval_data.merge(week_of_year, left_index=True, right_index=True)
    
    # Month
    month = idx.dt.month.rename("month")
    interval_data = interval_data.merge(month, left_index=True, right_index=True)
    
    return interval_data.set_index(idx)


def generate_data(message, orderbook, granularity = '10s', standardize_offset = 1):
    base_data = get_base_features(message, orderbook)
    interval_insight_data = get_interval_insight_features(base_data, granularity = granularity)
    interval_data = get_interval_features(interval_insight_data)
    standardized_interval_data = standardize_previous(interval_data, standardize_offset)
    data = get_time_features(standardized_interval_data)
    return data


#####
##### Generate Labels
#####
"""
m_{+}(t) = (1/k) * \sum{p_{t+i}}
m_{-}(t) = (1/k) * \sum{p_{t-i+1}}
l_t = (m_{+}(t) - m_{-}(t)) / m_{-}(t)

Note that this function is for generating labels, hence it uses 
information from the future. Both in smoothing and in computing 
the standard deviation.
"""
def get_label(mid_price, time_horizon=50, threshold = 0.40):
    mp_current = mid_price.rolling(time_horizon).mean()
    mp_next = mp_current.shift(-time_horizon)
    tmp = (mp_next - mp_current) / mp_current
    std = tmp.std()
    results = 0 * tmp.copy()
    results.loc[tmp > threshold * std] = 1
    results.loc[tmp < -threshold * std] = -1
    return results.fillna(0).astype(int).rename('label')


#####
##### Time series sampler
##### - Normalization should be performed BEFORE time series sampling
def is_same_day(array):
    series = pd.Series(array)
    if not issubclass(series.dtype.__class__, np.dtype('datetime64').__class__):
        raise TypeError(f"Function is_same_day was called on array of class {series.dtype.__class__} instead of {np.dtype('datetime64').__class__}")
    return (series.dt.day == series.dt.day.iloc[0]).all()

def generate_time_series_samples(data, labels, sample_size):
    # Verify index of time column
    # Labels, bid, ask (with time)
    na_idx = data.isna().any(axis=1) | labels.isna()
    
    data = data[~na_idx].reset_index()  # Data has time as its index, use it as a column
    labels = labels[~na_idx]
    
    # Save original info for later reuse
    columns = data.columns
    time_idx = columns.get_loc('time')  # Raises key error if not found
    
    samples = sliding_window_view(data, sample_size, axis=0).transpose((0, 2, 1))  # Outputs: (Batch, Observation, Column)
    sample_labels = labels.iloc[ (sample_size-1) : ]                               # Labels corresponding to `samples`
    
    same_day_mask = np.apply_along_axis(is_same_day, axis=1, arr=samples[:, :, time_idx])
    non_time_mask = np.arange(samples.shape[-1]) != time_idx
    samples = samples[same_day_mask, :, :][:, :, non_time_mask]  # Filter for data from the same day, and exclude the time column
    sample_labels = sample_labels[same_day_mask]                 # Labels corresponding to `samples`
    
    return samples.copy().astype('float32'), sample_labels.copy()


#####
##### Performance evaluator
#####
"""
Evaluates the performance of a given strategy.
Ends by closing the last position that we took.

Assumes shorting is possible.

Bid: The bid price at the start of each interval
Ask: The ask price at the start of each interval
Position is a list or array:
  1  = BUY
  0  = EXIT
  -1 = SELL
"""
def evaluate_performance(bids, asks, signals, trading_fee = 0.001):
    not_same_shape = (signals.shape != asks.shape) or (signals.shape != bids.shape)
    if not_same_shape:
        raise ValueError(f"The arguments provided do not have matching shapes")
    
    has_timeindex_bids = issubclass(bids.index.__class__, pd.core.indexes.datetimes.DatetimeIndex)
    has_timeindex_asks = issubclass(asks.index.__class__, pd.core.indexes.datetimes.DatetimeIndex)
    if (bids.index != asks.index).any() or not (has_timeindex_bids and has_timeindex_asks):
        raise ValueError(f"The bids and asks arrays provided do not have matching time-series indexes")
    
    if not issubclass(signals.__class__, pd.Series):
        signals = pd.Series(signals, index=bids.index)
    
    if not issubclass(signals.index.__class__, pd.core.indexes.datetimes.DatetimeIndex) or (signals.index != asks.index).any():
        raise ValueError(f"The signals array provided does not have matching time-series indexes with the bids and asks time-series arrays provided")
        
    if bids.isna().any() or asks.isna().any() or signals.isna().any():
        raise ValueError(f"The arguments provided should not contain any NA values")
        
    timestamps = pd.Series(signals.index)

    # Signal: Do not hold positions overnight
    end_of_day_timestamps = timestamps.groupby([timestamps.dt.year, timestamps.dt.dayofyear]).max().reset_index(drop=True)
    signals.loc[end_of_day_timestamps] = 0

    # Signal: Identify the signal changes
    signal_change_mask = signals != signals.shift(1)     # First value True by default since not equal to NA
    signal_change_mask.iloc[0] = signals.iloc[0] != 0    # Set first value to True if given a signal

    # Position: Movements required
    signal_next = signals.loc[signal_change_mask]
    
    if signal_change_mask.iloc[0]:
        signal_prev = pd.Series(signals.loc[signal_change_mask.shift(-1).fillna(False)].values, index=signal_next.index[1:])
    else:
        signal_prev = pd.Series(signals.loc[signal_change_mask.shift(-1).fillna(False)].values, index=signal_next.index)
        
    movements = signals.copy() * 0
    movements.loc[signal_next.index] = (signal_next - signal_prev).fillna(signals.iloc[0])
    
    # Movements can only be -2, -1, 0, 1, 2
    assert movements.isin([-2, -1, 0, 1, 2]).all()
    
    # Movements should be net neutral, starting and ending with no assets
    assert movements.sum() == 0
    
    # Profit: Bid-ask spread
    buy_costs = -1 * (asks * movements.loc[movements > 0]).dropna()
    sell_profits = (bids * movements.loc[movements < 0].abs()).dropna()
    profit = sell_profits.sum() + buy_costs.sum()
    
    # Profit: Bid-ask spread cumulative
    profit_cumulative = movements.copy() * 0
    profit_cumulative.loc[buy_costs.index] = buy_costs
    profit_cumulative.loc[sell_profits.index] = sell_profits
    profit_cumulative = profit_cumulative.cumsum()
    
    # Profit: Mid-price
    mid_price = (asks + bids) / 2
    mid_buy_costs = -1 * (mid_price * movements.loc[movements > 0]).dropna()
    mid_sell_profits = (mid_price * movements.loc[movements < 0].abs()).dropna()
    profit_mid_price = mid_sell_profits.sum() + mid_buy_costs.sum()
    
    # Trading costs
    num_trades = movements.abs().sum()
    trading_costs = num_trades * trading_fee
    
    results = {
        'profit': profit,
        'profit_cumulative': profit_cumulative,
        'profit_mid_price': profit_mid_price,
        'trade_movements': movements,
        'trading_costs': trading_costs,
        'number_of_trades': num_trades.astype(int),
    }
    
    return results

def eval_performance(bids, asks, signals, plot_raw = False, fig_size = (8, 7), **kwargs):
    results = evaluate_performance(bids, asks, signals, **kwargs)
    print(f"Profit: {results['profit']}\nNumber of trades: {results['number_of_trades']}\nTrading costs: {results['trading_costs']}")
    
    cumu_profits = results['profit_cumulative']
    cumu_profits_index = cumu_profits.index.to_series()
    _end_of_day_profits = cumu_profits.groupby([cumu_profits_index.dt.year, cumu_profits_index.dt.dayofyear]).last().values
    _end_of_day_indexes = cumu_profits_index.groupby([cumu_profits_index.dt.year, cumu_profits_index.dt.dayofyear]).last().values
    end_of_day_profits = pd.Series(_end_of_day_profits, index=_end_of_day_indexes)
    
    end_of_day_profits.plot(color="black", marker='.', label="End of day balance", figsize=fig_size)
    if plot_raw:
        results['profit_cumulative'].plot(alpha=0.25, label="Intraday balance")
    plt.legend()
    plt.ylabel("Cumulative earnings")
    plt.title("Account balance as trading progressed");
    
    
#####
##### Data splitting
#####
def _data_split(X, y, n_splits):
    inc = X.shape[0] // n_splits
    start_idx = 0
    end_idx = inc
    for i in range(n_splits):
        _X = X[start_idx:end_idx]
        _y = y[start_idx:end_idx]
        start_idx, end_idx = end_idx, end_idx + inc
        yield _X, _y

def time_series_split(X, y, model, n_splits=4, test_size=0.2, **kwargs):
    preds = []
    for _X, _y in _data_split(X, y, n_splits):
        X_train, X_test, y_train, y_test = train_test_split(_X, _y, shuffle=False, test_size=test_size) # Check that test_size travels through
        _model = model().fit(X_train, y_train)
        _preds = _model.predict(X_test)
        print(f"Accuracy: {(_preds == y_test).mean()}")
        preds.append(pd.Series(_preds, index=y_test.index))
    return pd.concat(preds)
