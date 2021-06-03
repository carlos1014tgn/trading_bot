import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from data.data_manager import filename_to_dataframe


#df = pd.read_csv('data/Binance_BTCUSDT_1h.csv')


def execute_strategy(df):
    df['prev_open'] = df.open.shift(-1)
    df['prev_close'] = df.close.shift(-1)
    df['post_open'] = df.open.shift(1)
    df['post_close'] = df.close.shift(1)
    df['2prev_open'] = df.open.shift(-2)
    df['2prev_close'] = df.close.shift(-2)
    df['2post_open'] = df.open.shift(2)
    df['2post_close'] = df.close.shift(2)


    df['classification'] = None
    df['classification'] = df.apply(lambda row: 'high' if 
        max(row['prev_open'], row['prev_close']) <= max(row['open'], row['close']) and
        max(row['2prev_open'], row['2prev_close']) <= max(row['open'], row['close']) and
        max(row['open'], row['close']) >= max(row['post_open'], row['post_close']) and 
        max(row['open'], row['close']) >= max(row['2post_open'], row['2post_close'])
        else row['classification'], axis=1)

    df['classification'] = df.apply(lambda row: 'low' if 
        min(row['prev_open'], row['prev_close']) >= min(row['open'], row['close']) and
        min(row['2prev_open'], row['2prev_close']) >= min(row['open'], row['close']) and
        min(row['open'], row['close']) <= min(row['post_open'], row['post_close']) and
        min(row['open'], row['close']) <= min(row['2post_open'], row['2post_close']) 
        else row['classification'], axis=1)


    def get_prev(timestamp, orientation='high', num=2):
        try:
            filtered_df = df[df['classification'] == orientation]
            filtered_df = filtered_df[filtered_df['timestamp'] < timestamp]
            filtered_df.reset_index(inplace=True)
            row = filtered_df.iloc[-1]
            if row['timestamp'] > timestamp-num*45*60*1000:
                row = filtered_df.iloc[-2]
            return max(row['open'], row['close'])
        except:
            return 0




    df['2prev_high'] = df.progress_apply(lambda row: get_prev(row['timestamp'], orientation='high', num=2) if row['classification'] == 'low' else None, axis=1)
    df['2prev_low'] = df.progress_apply(lambda row: get_prev(row['timestamp'], orientation='low', num=2) if row['classification'] == 'high' else None, axis=1)
    df['prev_high'] = df.progress_apply(lambda row: get_prev(row['timestamp'], orientation='high', num=1), axis=1)
    df['prev_low'] = df.progress_apply(lambda row: get_prev(row['timestamp'], orientation='low', num=1), axis=1)

    df['set_buy_price'] = df.apply(lambda row: row['prev_high'] if min(row['open'], row['close']) > row['prev_high'] and row['classification'] == 'low' else None, axis=1)
    df['buying_price'] = df.set_buy_price.shift(1)
    df['buy'] = df.apply(lambda row: row['close'] > row['buying_price'], axis=1)

    df['set_sell_price'] = df.apply(lambda row: row['prev_low'] if max(row['open'], row['close']) < row['prev_low'] and row['classification'] == 'high' else None, axis=1)
    df['selling_price'] = df.set_buy_price.shift(1)
    df['sell'] = df.apply(lambda row: row['close'] < row['selling_price'], axis=1)


    df.drop(columns=['prev_open',  'prev_close',  'post_open',  'post_close',  '2prev_open',  '2prev_close',  '2post_open',  '2post_close'], inplace=True)

    print(df.tail(300).to_string())


    for base_stoploss in range(60, 96, 2):
        for base_take_profit in range(104, 300, 2):
            base_stoploss = 94
            base_take_profit = 106
            current_usd = 1000
            current_btc = 0
            for i, row in df.iterrows():
                if current_btc == 0:
                    if row['buy']:
                        current_btc = current_usd / row['close']
                        current_usd = 0
                        take_profit = row['close']*base_take_profit/100
                        stoploss = row['close']*base_stoploss/100
                        print(f'at {row["date"]} USD: {current_usd} BTC: {current_btc}')

                if current_usd == 0:
                    if row['sell']:
                        current_usd = current_btc * row['close']
                        current_btc = 0
                        print(f'at {row["date"]} USD: {current_usd} BTC: {current_btc}')
                    
                    elif row['high'] > take_profit:
                        current_usd = current_btc *take_profit
                        current_btc = 0
                        print(f'at {row["date"]} USD: {current_usd} BTC: {current_btc}')

                    elif row['low'] < stoploss:
                        current_usd = current_btc * stoploss
                        current_btc = 0
                        print(f'at {row["date"]} USD: {current_usd} BTC: {current_btc}')
            exit()
            if current_usd > 4000 or current_btc > 0.3:
                print(f'base_stoploss {base_stoploss}, base_take_profit {base_take_profit} : USD: {current_usd} BTC: {current_btc}')
    
        

#66 -238 --144 usd

if __name__ =='__main__':
    df = filename_to_dataframe('EURUSD60.csv')
    execute_strategy(df)