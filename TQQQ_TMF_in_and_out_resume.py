"""
Based on 'In & Out' strategy by Peter Guenther 4 Oct 2020
expanded/inspired by Tentor Testivis, Dan Whitnable (Quantopian), Vladimir, and Thomas Chang.

https://www.quantopian.com/posts/new-strategy-in-and-out
https://www.quantconnect.com/forum/discussion/9597/the-in-amp-out-strategy-continued-from-quantopian/p1

IDEAS: for 'IN' state, consider choice between DIA, QQQ, IWM (or the 3X leveraged version)
"""


import numpy as np
import pandas as pd
import scipy.optimize as sc_opt
import datetime as dt
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D


class TQQQ_InOut:
    def __init__(self,threshold,wait_days,num_avg,num_lag,start=None,end=None):
        self.INI_WAIT_DAYS = wait_days
        active_dir  = os.path.join(os.getcwd(),'ACTIVE_SYMBOLS')
        simulation_dir = os.path.join(os.getcwd(),'CREATED_DATA')
        #GLD: gold
        #SLV: silver
        #XLU: utilities
        #FXF: safe currency
        #FXA: risk currency
        #XLI: industrials
        #DBB: metals
        #IGE: natural resources
        #SHY: 1-3 years (short term) treasury bonds (Debt)
        #STIP: treasury-inflation-protected-securities(TIPS) 0-5 year (short term) maturity ETF
        #TIP: treasury-inflation-protected-securities(TIPS) ETF
        #UUP: dollar index

        self.FORPAIRS = ['GLD','SLV','XLU','FXF','FXA','TIP']
        self.SIGNALS = ['XLI','DBB','IGE','SHY','UUP']
        self.MRKT = 'SPY'
        self.TRADES = ['TMF','TQQQ']
        self.all = pd.concat([pd.read_csv(os.path.join(active_dir,symbol+'.csv'),parse_dates=['timestamp']).set_index('timestamp').rename(columns = {'close':symbol})[symbol] for symbol in [self.MRKT] + self.FORPAIRS+ self.SIGNALS],axis=1)
        # print(self.all.loc[pd.Timestamp(dt.date(2007,2,26)):])
        self.returns_sample = (self.all / self.all.rolling(num_avg).mean().shift(num_lag) - 1).dropna()
        # self.returns_sample = self.all.rolling(num_avg).mean().pct_change(num_lag).dropna()
        # Reverse code USDX: sort largest changes to bottom
        self.returns_sample['UUP'] = self.returns_sample['UUP'] * (-1)
        # For pairs, take returns differential, reverse coded
        self.returns_sample['G_S'] = -(self.returns_sample['GLD'] - self.returns_sample['SLV'])
        self.returns_sample['U_I'] = -(self.returns_sample['XLU'] - self.returns_sample['XLI'])
        self.returns_sample['C_A'] = -(self.returns_sample['FXF'] - self.returns_sample['FXA'])
        self.pairlist = ['G_S', 'U_I', 'C_A']
        self.out_signal = (self.returns_sample < self.returns_sample.rolling(187).quantile(threshold,interpolation='linear')) [[column_name for column_name in self.SIGNALS+self.pairlist]]

        #Interest rate expectations (cost of debt) may increase because the economic outlook improves (showing in rising input prices) = actually not a negative signal
        self.out_signal['SHY'] = np.where((self.all.loc[self.returns_sample.index[0]:][['DBB','IGE']]>=self.all.loc[self.returns_sample.index[0]:][['DBB','IGE']].rolling(187).median()).any(axis=1),False,self.out_signal['SHY'])

        ### GOLD/SLVA differential may increase due to inflation expectations which actually suggest an economic improvement = actually not a negative signal
        ### NEED TO FIGURE OUT WHICH WILL RISE AND WHICH WILL NOT
        self.out_signal['G_S'] = np.where(self.returns_sample['SHY']-self.returns_sample['TIP']<=(self.returns_sample['SHY']-self.returns_sample['TIP']).rolling(187).median(),False,self.out_signal['G_S'])
        self.out_signal = self.out_signal.astype(float).dropna()
        in_out_mkt_time = 'close'
        in_out_real = pd.concat([pd.read_csv(os.path.join(active_dir,symbol+'.csv'),parse_dates=['timestamp'],dayfirst=True).set_index('timestamp').rename(columns = {in_out_mkt_time:symbol})[symbol] for symbol in self.TRADES],axis=1)
        in_out_hist_simulated = pd.concat([pd.read_csv(os.path.join(simulation_dir,symbol+'_created.csv'),parse_dates=['timestamp'],dayfirst=True).set_index('timestamp').rename(columns = {in_out_mkt_time:symbol})[symbol] for symbol in self.TRADES if os.path.exists(os.path.join(simulation_dir,symbol+'_created.csv'))],axis=1)
        self.in_out = in_out_real.combine_first(in_out_hist_simulated).dropna()
        # self.in_out = self.in_out.shift(-1).dropna()
        # print(self.in_out)
        self.start = max(self.out_signal.loc[start:end].index[0],self.in_out.loc[start:end].index[0])
        self.end = min(self.out_signal.loc[start:end].index[-1],self.in_out.loc[start:end].index[-1])
        # print(self.enddate)
        # print('start:',str(self.startdate))
        ## Day count variables
        self.num_wait = 0
        self.init_cash = 10000

    def get_benchmark(self):
        hold_returns = self.in_out.loc[self.end]/self.in_out.loc[self.start]
        tqqq_returns = hold_returns['TQQQ']*self.init_cash
        tmf_returns = hold_returns['TMF']*self.init_cash
        print('TQQQ END VALUE:',tqqq_returns)
        print('TMF END VALUE:',tmf_returns)
        print('TQQQ hold ror:', (hold_returns['TQQQ'])**(1/((self.end-self.start).days/365.25)))
        return

    def BACKTEST(self,my_tau,weights,track_daily=False):
        #shape of weights must be equal to 1Xlen(row) in signal df (excluding the final output signal)
        #in the current case, weights correspond to ['XLI','DBB','IGE','SHY','UUP','G_S', 'U_I', 'C_A']
        self.cash = self.init_cash
        self.TMF_quant = 0
        self.TQQQ_quant = 0
        self.long=[]
        self.short=[]

        if track_daily:
            self.daily_tracker={'timestamp':[],'portfolio_value':[]}
        func_out_signal = self.out_signal.mul(weights,axis=1)
        backtest_df = self.in_out[self.TRADES].loc[self.start:self.end].copy()
        backtest_df['out_signal']=func_out_signal.loc[self.start:self.end].values.sum(axis = 1)
        backtest_df['wait_days']=self.INI_WAIT_DAYS*backtest_df[['out_signal']].rolling(5,win_type='exponential').sum(center=4,tau=my_tau,sym=False)
        self.prev_in_state = None
        def backtest_day(row):
            # Determine whether 'in' or 'out' of the market
            timestamp = row.name
            # if row['wait_days']>=1:
            #     self.num_wait = max(row['wait_days'],self.num_wait)
            # if self.num_wait < 1:
            if row['wait_days']<1:
                self.be_in = True
            else:
                self.be_in = False

            # Swap to 'out' assets if applicable
            if self.be_in==False and self.be_in != self.prev_in_state:
                self.cash += self.TQQQ_quant*row['TQQQ']
                self.TMF_quant = self.cash/row['TMF']
                self.TQQQ_quant = 0
                self.cash = 0
                self.short.append(timestamp)
            elif self.be_in==True and self.be_in != self.prev_in_state:
                self.cash += self.TMF_quant*row['TMF']
                self.TMF_quant = 0
                self.TQQQ_quant = self.cash/row['TQQQ']
                self.cash = 0
                self.long.append(timestamp)
            self.prev_in_state = self.be_in
            if track_daily:
                self.daily_tracker['timestamp'].append(timestamp)
                self.daily_tracker['portfolio_value'].append(self.cash+self.TQQQ_quant*row['TQQQ']+self.TMF_quant*row['TMF'] )

        backtest_df.apply(backtest_day,axis =1)
        num_trades = len(self.short)
        if track_daily:
            self.daily_tracker_df = pd.DataFrame.from_dict(self.daily_tracker)
            self.daily_tracker_df['portfolio_cummax']=self.daily_tracker_df['portfolio_value'].cummax()
            maxdrawdown = ((self.daily_tracker_df['portfolio_value']-self.daily_tracker_df['portfolio_value'].cummax())/self.daily_tracker_df['portfolio_value'].cummax()).min()
            print('MAX DRAWDOWN:',maxdrawdown)
            print('#num_trades:',num_trades)
            self.daily_tracker_df['timestamp_shift']=self.daily_tracker_df['timestamp'].shift()
            self.daily_tracker_df[['benchmark_value','SOY']]=self.daily_tracker_df.apply(lambda x: x[['portfolio_value','timestamp']] if x['timestamp'].year != x['timestamp_shift'].year else np.nan,axis = 1).ffill()
            self.daily_tracker_df['sawtooth']=self.daily_tracker_df['portfolio_value']/self.daily_tracker_df['benchmark_value']
            self.daily_tracker_df=self.daily_tracker_df.set_index('timestamp')
            self.in_out = self.in_out.loc[self.start:self.end]
            self.in_out['timestamp_shift']=self.in_out.index.to_series().shift()
            self.in_out['TQQQ_benchmark']=self.in_out.apply(lambda x: x['TQQQ'] if x.name.year != x['timestamp_shift'].year else np.nan,axis = 1).ffill()
            self.in_out['TQQQ_sawtooth']=self.in_out['TQQQ']/self.in_out['TQQQ_benchmark']
            self.daily_tracker_df['TQQQ_sawtooth']=self.in_out['TQQQ_sawtooth']

        final_value = self.cash+self.TQQQ_quant*self.in_out.loc[self.end]['TQQQ']+self.TMF_quant*self.in_out.loc[self.end]['TMF']
        # if track_daily:
        #     return ((final_value/self.init_cash)**(1/((self.end-self.start).days/365.25))-1)/maxdrawdown
        # print([str(elt) for elt in self.long])
        # print(len(self.long))
        # print('TQQQ final value:',final_value)
        return (final_value/self.init_cash)**(1/((self.end-self.start).days/365.25))
        # return -final_value
    def plot_trades_and_signals(self,track_daily=False):
        for col_num in range (1,len(self.TRADES)+1):
            ax = plt.subplot(3,1,col_num)
            ax.title.set_text(self.TRADES[col_num-1])
            ax_long = ax.twinx()
            ax_long.vlines(self.long,ymin=0,ymax=8,colors='green')
            ax_long.set_navigate(False)
            ax_long.axes.get_yaxis().set_visible(False)
            ax_short = ax.twinx()
            ax_short.vlines(self.short,ymin=0,ymax=8,colors='red')
            ax_short.set_navigate(False)
            ax_short.axes.get_yaxis().set_visible(False)
            cols = self.SIGNALS+self.pairlist
            signals_plot = self.out_signal.loc[self.start:self.end][cols].copy().astype(int)
            for i in range(len(cols)):
                signals_plot[[cols[i]]] = signals_plot[[cols[i]]]*(i+1)
            signals_plot = signals_plot.reset_index()
            ax_signal=signals_plot.plot(ax=ax,x='timestamp',kind='line',linestyle='',marker='o',markersize=1,legend=None)
            ax_signal.set_yticks([i for i in range(1,len(cols)+1)])
            ax_signal.set_yticklabels(cols)
            ax_twin = ax.twinx()
            ax_series = self.in_out.loc[self.start:self.end][self.TRADES[col_num-1]].plot(ax=ax_twin,kind='line',linewidth=0.5)
            ax.set_navigate(False)
        if track_daily:
            ax_sawtooth = plt.subplot(3,1,3)
            self.daily_tracker_df['ONE']=1
            self.daily_tracker_df = self.daily_tracker_df.reset_index()
            self.daily_tracker_df.plot(ax=ax_sawtooth,x='timestamp',y=['sawtooth','TQQQ_sawtooth','ONE'],color=['red','blue','black'],linewidth=0.5,legend=None)
            soy_dates = (self.daily_tracker_df['timestamp'])
            ax_soy = ax_sawtooth.twinx()
            ax_soy.vlines(self.daily_tracker_df['SOY'],ymin=0,ymax=1,colors='black',linestyle='dotted')
            ax_soy.set_navigate(False)
            ax_soy.axes.get_yaxis().set_visible(False)
            # self.daily_tracker_df.plot(ax=ax_sawtooth,x='timestamp',y='TQQQ_sawtooth',color='black')
        plt.show()
    def optim_weights(self,weights,my_tau,track_daily=False):
        return self.BACKTEST(my_tau,weights,track_daily)*(-1)


def main():
    start = None
    end = None
    #weights for ['XLI','DBB','IGE','SHY','UUP','G_S', 'U_I', 'C_A']

    threshold = 0.01
    window_width = 20
    window_lag = 55
    num_wait_days = 1.9
    my_tau = 4.6
    weights = [0.5,1,0,1,0.5,0.25,1,0.25]
    # weights = [0.5,0.75,0,1,0.5,0.25,1,0.5]  
    algo = TQQQ_InOut(threshold,num_wait_days,window_width,window_lag,start,end)
    algo.get_benchmark()
    print('TQQQ IN&OUT ror:',algo.BACKTEST(my_tau,weights,track_daily=True))
    algo.plot_trades_and_signals(track_daily=True)
main()

def obj_func(input_tup):
    window_width = int(input_tup[0])
    window_lag = int(input_tup[1])
    num_wait_days = input_tup[2]
    my_tau = input_tup[3]
    # weights = [0.5,0.75,0,1,0.5,0.25,1,0.5]
    weights = [1,1,0,0.75,0.25,0.5,1,0.5]
    threshold = 0.01

    start = None
    end = None
    track_daily = False
    algo = TQQQ_InOut(threshold,num_wait_days,window_width,window_lag,start,end)
    return algo.BACKTEST(my_tau,weights,track_daily)*(-1)

def optim_main():
    if __name__ == '__main__':
        optim_result = sc_opt.brute(obj_func,((10,110),(10,110),(1,10),(1,10)),Ns=21,workers=6)
        # threshold = 0.01
        # num_wait_days = 1.33684211
        # my_tau = 3.6
        # window_width = 23
        # window_lag = 47
        # start = None
        # end = None
        # algo = TQQQ_InOut(threshold,num_wait_days,window_width,window_lag,start,end)
        # optim_result = sc_opt.brute(algo.optim_weights,((0,1),)*8,args=(my_tau,),Ns=5,workers=-1)
        print(optim_result)
    return

# optim_main()