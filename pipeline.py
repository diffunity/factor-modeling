import os
import time
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split


class Data_pipeline:
    
    def __init__(self, data_config, logger):
        self.fundamentals_path = data_config["fundamentals_path"]
        self.stocks_path = data_config["stocks_path"]
        self.industries_path = data_config["industries_path"] 
        self.logger = logger
        self.fundamentals = pd.read_csv(self.fundamentals_path+'fundamentals.csv', 
                                        index_col=0)
        self.fundamentals["Period Ending"] = pd.to_datetime(self.fundamentals["Period Ending"], 
                                                            format="%Y-%m-%d")
        price_data_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
        
        self.industries = pd.read_csv(self.industries_path+"constituents_csv.csv")
      
        self.price_data = pd.DataFrame(columns=["Stock"]+price_data_columns)
        for company in self.companies:
          tmp_data = pd.read_csv(self.stocks_path+company.lower()+".us.txt")
          dates = self.fundamentals[self.fundamentals['Ticker Symbol']==company]["Period Ending"]
          begin, end = dates.min()-pd.DateOffset(years=1), dates.max()+pd.DateOffset(years=1)
          tmp_data["Date"] = pd.to_datetime(tmp_data["Date"], format="%Y-%m-%d")
          tmp_data = tmp_data[tmp_data["Date"].apply(lambda x: begin <= x <= end)]
          if len(tmp_data) > 1:
              tmp_data["Stock"] = company
              self.price_data = self.price_data.append(tmp_data)
              self.logger.info(f"Extracting {company} information") 
        self.logger.info(f"Extraction complete")
        self.price_data = self.price_data.reset_index(drop=True)

        self.price_data = self.price_data.merge(self.industries[["Symbol","Sector"]],\
                                                left_on="Stock", right_on="Symbol")\
                                                .drop(["Symbol"], axis=1)

    @property
    def companies(self):
        self.fund_set = set(self.fundamentals["Ticker Symbol"].unique())
        self.ind_set = set(self.industries["Symbol"].unique())
        self.stock_set = set([i.split(".")[0].upper() for i in os.listdir(self.stocks_path)\
                       if (os.path.isfile(self.stocks_path+i) and i.endswith("us.txt"))])
        if len(self.price_data) > 1:
            return self.fund_set.intersection(self.ind_set.intersection(self.price_data["Stock"]))
        return self.fund_set.intersection(self.ind_set.intersection(self.stock_set))
    
    def prices_pct(self, window=5):
        price_pct_change_year = self.price_data\
                                    .groupby([self.price_data.Stock, self.price_data.Date.dt.year])["Close"]\
                                    .pct_change(window)
        prices_pct = self.price_data.copy()
        prices_pct["Close"] = price_pct_change_year
        return prices_pct.dropna()

    def prices_vol(self):
        price_vol_year = self.price_data\
                                    .groupby([self.price_data.Stock, self.price_data.Date.dt.year])["Close"]\
                                    .std()
        prices_vol = self.price_data.copy()
        prices_vol["Close"] = price_vol_year.values
        return prices_vol.dropna()
        

    def fund_to_pct_change(self, window):
        tmp_fundamentals = self.fundamentals.copy()
        tmp_fundamentals["Period Ending"] = tmp_fundamentals["Period Ending"].apply(lambda x: x.year+1)
        prices_pct = self.prices_pct(window)
        return preprocessing_wrapper(prices_pct.groupby([prices_pct.Stock, prices_pct.Date.dt.year])[["Close"]]\
                                               .std().dropna().reset_index(level=[0])\
                                               .merge(tmp_fundamentals, left_on=["Stock", "Date"], \
                                                right_on=["Ticker Symbol", "Period Ending"])\
                                               .drop("Ticker Symbol", axis=1))
    def fund_to_vol(self):
        tmp_fundamentals = self.fundamentals.copy()
        tmp_fundamentals["Period Ending"] = tmp_fundamentals["Period Ending"].apply(lambda x: x.year+1)
        # prices_vol = self.prices_vol()
        return preprocessing_wrapper(self.price_data.groupby([self.price_data.Stock, self.price_data.Date.dt.year])[["Close"]]\
                                                    .std()\
                                                    .reset_index(level=[0])\
                                                    .merge(tmp_fundamentals, left_on=["Stock", "Date"], \
                                                     right_on=["Ticker Symbol", "Period Ending"])\
                                                    .drop("Ticker Symbol", axis=1))
        
    def calc_volatilities(self, quote, column, days=5):
        start_idx = self.price_data[self.price_data["Stock"]==quote].index[0]+days-1
        return self.price_data[self.price_data["Stock"]==quote].rolling(days)[column].std().loc[start_idx:]

    def stock_volatilities(self, days:List[int]):
        stocks_vol = pd.DataFrame(columns=["Stock"]+days)
        for company in self.companies:
            tmp_data = {"Stock":company, 
                        "Industry":self.industries["Sector"][self.industries["Symbol"]==company].item()}
            for day in days:
                tmp_data[day] = self.calc_volatilities(company, ["Close"], days=day).mean().item()
            stocks_vol = stocks_vol.append(tmp_data, ignore_index=True)
        return stocks_vol

    def industry_volatilities(self, days:List[int]):
        stocks_vol = self.stocks_volatilities(days)
        return stocks_vol.groupby("Industry").agg(["mean","std"])

    def plot(self, quote, ma=[3,5], filename=None):
        # defaults to moving average with [3,5]
        data = self.price_data[self.price_data['Stock']==quote]
        fig = plt.figure(figsize=(27,18))
        gs = fig.add_gridspec(nrows=4, ncols=1)
        price = fig.add_subplot(gs[:3,:])
        price.plot(data["Close"])
        if ma is not None:
            for ma_time in ma:
                price.plot(data["Close"].rolling(ma_time).mean(), lw=0.5)
        vol = fig.add_subplot(gs[-1,:])
        vol.plot(data["Date"],data["Volume"])
        ax = fig.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.gcf().autofmt_xdate()
        self.logger.info(f"Saving to {filename}")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.clf()

    @staticmethod
    def produce_report(fname, data, test_ratio=0.1, topn=15, save=True):
        fig = plt.figure(figsize=(40,100))
        gs = fig.add_gridspec(nrows=13, ncols=1)
        top_corr = fig.add_subplot(gs[:3,:])
        heatmap = fig.add_subplot(gs[3:7,:])
        ols_result = fig.add_subplot(gs[7:10,:])
        regplot = fig.add_subplot(gs[10:13,:])
    
        top_corr.text(0, 0, str(data.corr()["pct_change"].sort_values()[-15:-1]),
                            {'fontsize': 35}, fontproperties = 'monospace',
#                           ha='left',
#                            verticalalignment='center',
                     )
        top_corr.title(f"Correlation ranking (Top {topn})")
        top_corr.axis('off')
    
        sns.heatmap(data.corr(), ax=heatmap)
    
        columns_to_keep = data.corr()["pct_change"].sort_values()[-15:-1].index
        X, X_test, Y, Y_test = train_test_split(data.loc[:,columns_to_keep].values,\
                                                data["pct_change"].values,\
                                                test_size=test_ratio)
        X = sm.add_constant(X)
        X_test = sm.add_constant(X_test)
    
        model = sm.OLS(Y,X)
    
        results = model.fit()
    
        ols_result.text(0, 0, str(results.summary()),
                              {'fontsize': 35}, fontproperties = 'monospace',
#                             ha='center',
#                              verticalalignment='center'
                       )
        ols_result.axis('off')
        print(Y.shape, X.shape)
        print(Y_test.shape, X_test.shape, results.params.shape)
        sns.regplot(Y_test, (X_test @ results.params),
                    ax=regplot)
        plt.tight_layout()
        if save:
            self.logger.info(f"Saving to {fname}")
            plt.save(fname)
        else:
            plt.plot()

class preprocessing_wrapper:
    def __init__(self, data):
        self.data = data

    def raw(self):
        return self.data

    def minmax(self):
        aaa = self.data.drop(["Period Ending", "Close"],axis=1)\
                       .groupby("Stock")\
                       .transform(lambda x: (x - x.min()) / (x.max()-x.min()))
        idx_to_keep = aaa.isnull().sum(axis=0).sort_values()[aaa.isnull().sum(axis=0).sort_values() < 100].index
        data_preprocessed = self.data[["Stock","Close", "Period Ending"]].join(aaa[idx_to_keep])
        return data_preprocessed.fillna(0)

    def standardscaling(self):
        bbb = self.data.drop(["Period Ending", "Close"],axis=1)\
                       .groupby("Stock")\
                       .transform(lambda x: True if x.min()==x.max()==0 else False)
        idx_to_keep = bbb.sum(axis=0)[bbb.sum(axis=0)<100].index
        ccc = self.data.drop(["Period Ending", "Close"],axis=1)\
                       .groupby("Stock")\
                       .transform(lambda x: (x-x.mean())/(x.std()))
        data_preprocessed = self.data[["Stock","Close", "Period Ending"]].join(ccc[idx_to_keep])
        return data_preprocessed.fillna(0)
