import os

import numpy as np
import pandas as pd

# import pyarrow as py # version: 12.0.0


def save_dataframe(df: pd.DataFrame, main_path: str, file_name: str = 'data', save_type: str = 'parquet') -> None:
    FILE = ''
    if save_type == 'csv':
        FILE = file_name+'.csv'
        PATH = os.path.join(main_path, FILE)
        df.to_csv(PATH)
        return

    if save_type == 'parquet':
        FILE = file_name+'.parquet'
        PATH = os.path.join(main_path, FILE)
        df.to_parquet(PATH)
        return


def check_num_none_and_inf(df: pd.DataFrame) -> None:
    print(f"data shape: {df.shape}")
    print('num Null: ', df.isnull().sum().sum())
    print('num Nan: ', df.isna().sum().sum())
    print('num EMPTY CELL: ', (df.values == '').sum())
    print('num NONE value: ', (df.values == 'NONE').sum())
    print('num np.inf value: ', (df.values == np.inf).sum())


def read_files(main_path: str, num_files: int = 0) -> pd.DataFrame:
    COUNT = 0
    df = pd.DataFrame()

    # Read "etfs" folder
    PATH = os.path.join(main_path, 'etfs')
    for _, _, files in os.walk(PATH):
        for file in files:
            if num_files != 0 and COUNT == num_files:
                break
            COUNT += 1
            data_path = os.path.join(PATH, file)
            df_tmp = pd.read_csv(data_path)
            # If number of records per day was less than 60 we cannnot calculate "rolling(30)"
            if df_tmp.shape[0] < 60:
                print(f'this {file} does not have enough data: {df_tmp.shape}')
                continue
            df_tmp['Symbol'] = file.replace('.csv', '')
            # Calculate the moving average
            df_tmp['vol_moving_avg'] = df_tmp.Volume.rolling(30).mean()
            # Calculate the rolling median
            df_tmp['adj_close_rolling_med'] = df_tmp['Adj Close'].rolling(
                30).median()
            if df is None:
                df = df_tmp
            else:
                df = pd.concat([df, df_tmp])
    # Also read "stocks" folder
    if num_files != 0 and COUNT < num_files:
        PATH = os.path.join(main_path, 'stocks')
        for _, _, files in os.walk(PATH):
            for file in files:
                if num_files != 0 and COUNT == num_files:
                    break
                COUNT += 1
                data_path = os.path.join(PATH, file)
                df_tmp = pd.read_csv(data_path)
                # If number of records per day was less than 60 we cannnot calculate "rolling(30)"
                if df_tmp.shape[0] < 60:
                    print(
                        f'this {file} does not have enough data: {df_tmp.shape}')
                    continue
                df_tmp['Symbol'] = file.replace('.csv', '')
                # Calculate the moving average
                df_tmp['vol_moving_avg'] = df_tmp.Volume.rolling(30).mean()
                # Calculate the rolling median
                df_tmp['adj_close_rolling_med'] = df_tmp['Adj Close'].rolling(
                    30).median()
                if df is None:
                    df = df_tmp
                else:
                    df = pd.concat([df, df_tmp])
    return df
