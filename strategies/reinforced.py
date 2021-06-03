from typing import final
import pandas as pd
from random import choice, random
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
from kneed import DataGenerator, KneeLocator
import time

class Simulator:
    def __init__(self, pair_file='/home/ubuntu/Projects/tradingbot/data/forex/AUDCHF60.csv'):
        self.df = pd.read_csv(pair_file)
        self.status = None
        self.sl = None
        self.tp = None
        self.enter_price = None
        self.leverage = 1
        self.locked_amount = 0
        self.comission = 0.0025 # %
        # We start with 3 months data
        self.clock = 2160
        self.last_row = self.df.loc[self.clock]
    
    def step(self, order=None):
        """
        order: None or {type: buy/sell, amount: in base curr, leverage: multiplier, tp: price, sl: price}
        return: None or money if position is closed.
        """

        # Previous checks
        money_back = 0
        if self.clock >= len(self.df.index):
            print('finished')
            return 'finish', self.__amount_to_realize()

        self.last_row = self.df.loc[self.clock]
        final_status = None

        # Check if there is anything to do with open positions.
        if self.status == 'long':
            if self.last_row['high'] >= self.tp:
                print('tp hit')
                amount = self.__amount_to_realize(self.tp)
                self.reset_status()
                self.status = 'closed'
                return self.status, amount

            elif self.last_row['low'] <= self.sl:
                print('sl hit')
                amount = self.__amount_to_realize(self.sl)
                self.reset_status()
                self.status = 'closed'
                return self.status, amount
        
        elif self.status == 'short':
            if self.last_row['high'] >= self.sl:
                print('sl hit')
                amount = self.__amount_to_realize(self.sl)
                self.reset_status()
                self.status = 'closed'
                return self.status, amount

            elif self.last_row['low'] <= self.tp:
                print('tp hit')
                amount = self.__amount_to_realize(self.tp)
                self.reset_status()
                self.status = 'closed'
                return self.status, amount

        else: # Only when we are not closing positions, we evaluate orders.
            amount_multiplier = 1
            if order == None:
                final_status = None
            elif order['type'] == 'buy':
                final_status = 'long'
                self.locked_amount = order['amount'] * (1 - self.comission/100)
                self.enter_price = self.last_row['close']
                print(f'Order received: {order}')
            elif order['type'] == 'sell':
                final_status = 'short'
                self.locked_amount = order['amount'] * (1 - self.comission/100)
                self.enter_price = self.last_row['close']
                print(f'Order received: {order}')
        
            if order != None:
                if 'tp' in order.keys():
                    self.tp = order['tp']
                if 'sl' in order.keys():
                    self.sl = order['sl']

        self.clock +=1
        self.status = final_status
        return final_status, money_back

        
    def __str__(self) -> str:
        return f'Position: {self.status}, Amount: {self.__amount_to_realize()}'
    
    def __amount_to_realize(self, price=None):
        to_realize = 0
        if price == None:
            price = self.last_row['close']
        if self.status == None:
            to_realize == 0
        elif self.status == 'long':
            to_realize = self.locked_amount * (price / self.enter_price) * self.leverage
        elif self.status == 'short':
            to_realize = self.locked_amount * (price / self.enter_price) * self.leverage
        return to_realize

    def reset_status(self):
        self.status = None
        self.sl = None
        self.tp = None
        self.enter_price = None
        self.leverage = 1
        self.locked_amount = 0
    
    def actual_chart(self):
        return self.df.loc[self.clock-2160: self.clock]


class Bot:
    def __init__(self):
        self.money = 1000
        self.simulator = Simulator()
        self.strategy = Reinforced()

    def act(self):
        #actual_price = self.simulator.actual_chart().tail(1)['close'].item()
        if self.money > 0:
            order = Resistances.decide(self.simulator.actual_chart(), self.money)
            if order != None:
                self.money = 0
                final_status, money_back = self.simulator.step(order)
                print(self.simulator)
            else:
                final_status, money_back = self.simulator.step()
        else:
            final_status, money_back = self.simulator.step()
        if final_status == 'closed':
            print(f'Position closed: money received is {money_back}')
            self.money = money_back
        if final_status == 'finish':
            print(f'Finished: {self.money}')
            return 'finish'

    def decide(self, data):
        # Can only be called when there is no position running.
        # Feeds from the past data
        # Returns the order datastructure
        return self.strategy.decide(data)


class Resistances:
    def decide(data, money):
        clusters = Resistances.supports(data['close'])
        current_price = data.tail(1)['close'].item()
        order = None
        for upper, lower in clusters:
            if lower < current_price < upper:
                distance = upper - lower
                margin = 0.2
                if current_price < lower + distance * margin:
                    #buy
                    order = {'type': 'buy', 'amount': money, 'leverage':5, 'price':current_price, 'tp': upper*0.999, 'sl': current_price*(1-0.1 * distance)}
                elif current_price > upper - distance * margin:
                    #sell
                    order = {'type': 'sell', 'amount': money, 'leverage':5, 'price':current_price, 'tp': lower*1.001, 'sl': current_price*(1+0.1 * distance)}
        print(order)
        return order
                
        


    def supports(X):
        # X can start as a pandas series
        # Returns a list of clusters.
        X = np.array(X)
        sum_of_squared_distances = []
        K = range(2,15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(X.reshape(-1,1))
            sum_of_squared_distances.append(km.inertia_)
        kn = KneeLocator(K, sum_of_squared_distances,S=1.0, curve="convex", direction="decreasing")
        #print(kn)
        #kn.plot_knee()
        #plt.plot(sum_of_squared_distances)
        #plt.show()
        kmeans = KMeans(n_clusters= kn.knee).fit(X.reshape(-1,1))
        c = kmeans.predict(X.reshape(-1,1))
        minmax = []
        for i in range(kn.knee):
            minmax.append([-np.inf,np.inf])
        for i in range(len(X)):
            cluster = c[i]
            if X[i] > minmax[cluster][0]:
                minmax[cluster][0] = X[i]
            if X[i] < minmax[cluster][1]:
                minmax[cluster][1] = X[i]
        return minmax
        # Deprecated
        for i in range(len(X)):
            colors = ['b','g','r','c','m','y','k','w']
            c = kmeans.predict(X[i].reshape(-1,1))[0]
            color = colors[c]
            plt.scatter(i,X[i],c = color,s = 1)
        for i in range(len(minmax)):
            plt.hlines(minmax[i][0],xmin = 0,xmax = len(X),colors = 'g')
            plt.hlines(minmax[i][1],xmin = 0,xmax = len(X),colors = 'r')
        plt.show()

class Reinforced:
    def __init__(self) -> None:
        pass

    def decide(self, data):
        pass

if __name__ == '__main__':

    x=Bot()
    for i in range(50000-2150):
        r = x.act()
        if r == 'finish':
            break
