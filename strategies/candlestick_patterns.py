import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from data.data_manager import filename_to_dataframe
from numpy import array

#df = pd.read_csv('data/Binance_BTCUSDT_1h.csv')

def is_382(o, h, l, c, po, ph, pl, pc):
    distance = abs(h-l)
    upper_threshold = h-0.382*distance
    lower_threshold = l+0.382*distance
    if o > upper_threshold and c > upper_threshold and po < pc:
        return 'bearish'
    elif o < lower_threshold and c < lower_threshold and pc < po:
        return 'bullish'
    return None

def is_engulfing(o, h, l, c, po, ph, pl, pc):
    if o == pc: # Market has no discontinuity
        if pc < po and c > po:
            return 'bullish'
        elif pc > po and c < po:
            return 'bearish'

def is_close_ab(o, h, l, c, po, ph, pl, pc):
        if o == pc: # Market has no discontinuity
            if pc < po and c > ph:
                return 'bullish'
            elif pc > po and c < pl:
                return 'bearish'

def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return ema[0]

# Including ema method to rolling object
#pd.DataFrame.rolling.ema = ema.__get__(pd.DataFrame.rolling)



def execute_strategy(pair, df):

    #Starts getting previous candle
    df['po'] = df.open.shift(1)
    df['ph'] = df.high.shift(1)
    df['pl'] = df.low.shift(1)
    df['pc'] = df.close.shift(1)
    #df['EMA_20'] = df.close.rolling(window=21).apply(lambda x: ema(x, 20))
    df['EMA_20'] = df.close.ewm(span=20, adjust=False).mean()

    df['type'] = None
    df['direction'] = None
    df['direction'] = df.apply(lambda row: is_382(row['open'], row['high'], row['low'], row['close'], row['po'], row['ph'], row['pl'], row['pc']), axis=1)
    df['type'] = df.apply(lambda row: '382' if row['direction'] is not None else None, axis=1)
    df['direction'] = df.apply(lambda row: is_close_ab(row['open'], row['high'], row['low'], row['close'], row['po'], row['ph'], row['pl'], row['pc']) if row['direction'] is None else row['direction'], axis=1)
    df['type'] = df.apply(lambda row: 'close_ab' if row['direction'] is not None and row['type'] is None else row['type'], axis=1)
    df['direction'] = df.apply(lambda row: is_engulfing(row['open'], row['high'], row['low'], row['close'], row['po'], row['ph'], row['pl'], row['pc']) if row['direction'] is None else row['direction'], axis=1)
    df['type'] = df.apply(lambda row: 'engulfing' if row['direction'] is not None and row['type'] is None else row['type'], axis=1)

    #382
    df['action'] = df.apply(lambda row: 'buy' if row['open'] > row['EMA_20'] and
                                                 row['close'] > row['EMA_20'] and
                                                 row['type'] == '382' and
                                                 row['direction'] == 'bullish' else None, axis=1)
                                                 
    df['action'] = df.apply(lambda row: 'sell' if row['open'] < row['EMA_20'] and
                                                 row['close'] < row['EMA_20'] and
                                                 row['type'] == '382' and
                                                 row['direction'] == 'bearish'
                                                 and row['action'] == None else row['action'], axis=1)

    df['entry'] = df.apply(lambda row: row['EMA_20'] if row['action'] == 'buy' or row['action'] == 'sell' else None, axis=1)
    # r:r ratio is 1:1
    df['stop'] = df.apply(lambda row: row['low']*0.999 if row['action'] == 'buy' else None, axis=1)
    df['stop'] = df.apply(lambda row: row['high']*1.001 if row['action'] == 'sell' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['low']*1.001 if row['action'] == 'buy' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['high']*0.999 if row['action'] == 'sell' else None, axis=1)


    #is_close_ab
    df['action'] = df.apply(lambda row: 'buy' if row['open'] > row['EMA_20'] and
                                                 row['close'] > row['EMA_20'] and
                                                 row['type'] == 'is_close_ab' and
                                                 row['direction'] == 'bullish' and row['action'] == None else row['action'], axis=1)
                                                 
    df['action'] = df.apply(lambda row: 'sell' if row['open'] < row['EMA_20'] and
                                                 row['close'] < row['EMA_20'] and
                                                 row['type'] == 'is_close_ab' and
                                                 row['direction'] == 'bearish'
                                                 and row['action'] == None else row['action'], axis=1)

    df['entry'] = df.apply(lambda row: row['EMA_20'] if row['action'] == 'buy' or row['action'] == 'sell' else row['action'], axis=1)
    # r:r ratio is 1:1
    df['stop'] = df.apply(lambda row: row['low']*0.999 if row['action'] == 'buy' else None, axis=1)
    df['stop'] = df.apply(lambda row: row['high']*1.001 if row['action'] == 'sell' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['low']*1.001 if row['action'] == 'buy' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['high']*0.999 if row['action'] == 'sell' else None, axis=1)

    #is_engulfing
    df['action'] = df.apply(lambda row: 'buy' if row['open'] > row['EMA_20'] and
                                                 row['close'] > row['EMA_20'] and
                                                 row['type'] == 'is_engulfing' and
                                                 row['direction'] == 'bullish' and row['action'] == None else row['action'], axis=1)
                                                 
    df['action'] = df.apply(lambda row: 'sell' if row['open'] < row['EMA_20'] and
                                                 row['close'] < row['EMA_20'] and
                                                 row['type'] == 'is_engulfing' and
                                                 row['direction'] == 'bearish'
                                                 and row['action'] == None else row['action'], axis=1)

    df['entry'] = df.apply(lambda row: row['EMA_20'] if row['action'] == 'buy' or row['action'] == 'sell' else None, axis=1)
    # r:r ratio is 1:1
    df['stop'] = df.apply(lambda row: row['low']*0.999 if row['action'] == 'buy' else None, axis=1)
    df['stop'] = df.apply(lambda row: row['high']*1.001 if row['action'] == 'sell' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['low']*1.001 if row['action'] == 'buy' else None, axis=1)
    df['takep'] = df.apply(lambda row: row['high']*0.999 if row['action'] == 'sell' else None, axis=1)

    print(df[df['action'] == 'sell'].to_string())
    




    leverage = 1
    money = 1000
    position = None #Asymetrical can be sell or buy
    stoploss = None
    take_profit = None
    for i, row in df.iterrows():
        if position == 'buy':
            money = money*(1+(row['close'] - row['pc'])*leverage)
        elif position == 'sell':
            money = money*(1-(row['close'] - row['pc'])*leverage)
        
        if position == None:
            if row['action'] == 'buy':
                position = 'buy'
                stoploss = row['stop']
                take_profit = row['takep']
            elif row['action'] == 'sell':
                position = 'sell'
                stoploss = row['stop']
                take_profit = row['takep']

        elif (position == 'buy' and row['action'] == 'sell') or (position == 'sell' and row['action'] == 'buy') or (position == 'buy' and row['low'] < stoploss) or (position == 'sell' and row['high'] > stoploss) or (position == 'buy' and row['high'] > take_profit) or (position == 'sell' and row['low'] < take_profit):
            position = None
            stoploss = None
            take_profit = None
    print(f'leverage: {leverage} money: {money}')
        





    # Entry point on candle close, stoploss 20 pips (0.1%) below candle low 





if __name__ =='__main__':
    df = filename_to_dataframe('EURUSD60.csv')
    execute_strategy(df)