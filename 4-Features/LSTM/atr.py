import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.copy_on_write = True


def load_dataset(file_path):
    df_split = pd.read_csv(file_path)
    column_name = df_split.columns[0]
    df_split = df_split[column_name].str.split('\t', expand=True)
    new_column_names = column_name.title().replace(
        '<', '').replace('>', '').split('\t')
    df_split.columns = new_column_names
    for column in df_split.columns[2:]:
        df_split[column] = df_split[column].astype(float)
    df_split.drop(columns=['Date', 'Time', 'Tickvol', 'Vol', 'Spread'], inplace=True)
    # df_split['Timestamp'] = pd.to_datetime(
    #     df_split['Date'] + ' ' + df_split['Time'])
    return df_split


def calculate_true_range(dataframe):
    dataframe['high_low'] = dataframe['High'] - dataframe['Low']
    dataframe['high_prev_close'] = abs(
        dataframe['High'] - dataframe['Close'].shift(1))
    dataframe['low_prev_close'] = abs(
        dataframe['Low'] - dataframe['Close'].shift(1))
    dataframe['True_Range'] = dataframe[['high_low',
                                         'high_prev_close', 'low_prev_close']].max(axis=1)
    return dataframe


def calculate_atr(dataframe, period_atr=14):
    dataframe = calculate_true_range(dataframe)
    dataframe['ATR'] = dataframe['True_Range'].ewm(span=period_atr, adjust=False).mean()
    return dataframe



def run(df):
    period = 21
    df = calculate_atr(df, period_atr=period)
    return df
