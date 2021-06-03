import pandas as pd
from tqdm import tqdm
from os import listdir
tqdm.pandas()
import logging
from datetime import datetime

logging.basicConfig(filename=f'{datetime.now().timestamp()}.log', level=logging.INFO)

"""
df = pd.read_csv('data/Binance_BTCUSDT_1h.csv')
df.rename(columns={'unix':'timestamp'}, inplace=True)
df.sort_values(by=['timestamp'], ascending=True, inplace=True)
"""




def execute_strategy(asset, df):
    lower_range = 5
    upper_range = 205
    for ma in range(lower_range,upper_range, 5):
        df[f'SMA_{ma}'] = df.close.rolling(window=ma).mean()


    for a in range(lower_range,upper_range, 5):
        for b in range(lower_range,upper_range, 5):
            df[f'{a}v{b}'] = df.apply(lambda row: 'above' if row[f'SMA_{a}'] > row[f'SMA_{b}'] else 'below', axis=1)
            df[f'prev{a}v{b}'] = df[f'{a}v{b}'].shift(1)
            df[f'crossed{a}v{b}'] = df.apply(lambda row: 'up' if row[f'{a}v{b}'] == 'below' and row[f'prev{a}v{b}'] == 'above' else '', axis=1)
            df[f'crossed{a}v{b}'] = df.apply(lambda row: 'down' if row[f'{a}v{b}'] == 'above' and row[f'prev{a}v{b}'] == 'below' else row[f'crossed{a}v{b}'], axis=1)

            #print(df.head(200).to_string())


            current_usd = 1000
            current_asset = 0
            for i, row in df.iterrows():
                if current_asset == 0:
                    if row[f'crossed{a}v{b}'] == 'up':
                        current_asset = current_usd / row['close']
                        current_usd = 0
                        #print(f'at {row["date"]} USD: {current_usd} BTC: {current_asset}')

                if current_usd == 0:
                    if row[f'crossed{a}v{b}'] == 'down':
                        current_usd = current_asset * row['close']
                        current_asset = 0
            print(f"Asset: {asset} , A: {a} , B: {b} , USD: {current_usd} , Asset: {current_asset}")
            logging.info(f"Asset: {asset} , A: {a} , B: {b} , USD: {current_usd} , Asset: {current_asset}")
                        #print(f'at {row["date"]} USD: {current_usd} BTC: {current_asset}')

                #print(f"A: {a} , B: {b} , USD: {current_usd} , BTC: {current_asset}")
