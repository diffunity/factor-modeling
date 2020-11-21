import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split

class data_pipeline:
    
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

        self.price_data = self.price_data.merge(self.industries[["Symbol","Sector"]], 
                                                left_on="Stock", right_on="Symbol")\
                                                .drop(["Symbol"], axis=1)

    def spy_total(self):
        return self.price_data

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

    def OLS(self, columns_to_keep, window, test_ratio=0.1):
        n = len(self.fund_to_pct_change)
        data = self.fund_to_pct_change(window)
        X, X_test, Y, Y_test = train_test_split(data["pct_change"].values,\
                                                data.loc[:,columns_to_keep].values,\
                                                test_size=test_ratio)
#        Y = data["pct_change"].values[:-int(n*test_ratio)]
#        X = data.loc[columns_to_keep,:]\
#                .values[:-int(n*test_ratio),:]

#        Y_test = data["pct_change"].values[-int(n*test_ratio):]
#        X_test = data.loc[columns_to_keep,:]\
#                     .values[-int(n*test_ratio):,:]

        X = sm.add_constant(X)
        X_test = sm.add_constant(X_test)

        model = sm.OLS(Y,X)
        results = model.fit()
        return results

    @property
    def companies(self):
        self.fund_set = set(self.fundamentals["Ticker Symbol"].unique())
        self.ind_set = set(self.industries["Symbol"].unique())
        self.stock_set = set([i.split(".")[0].upper() for i in os.listdir(self.stocks_path)\
                       if (os.path.isfile(self.stocks_path+i) and i.endswith("us.txt"))])
        if len(self.price_data) > 1:
            return self.fund_set.intersection(self.ind_set.intersection(self.price_data["Stock"]))
        return self.fund_set.intersection(self.ind_set.intersection(self.stock_set))

    def fund_to_pct_change(self, window):
        tmp_fundamentals = self.fundamentals.copy()
        tmp_fundamentals["Period Ending"] = tmp_fundamentals["Period Ending"].apply(lambda x: x.year+1)
        prices_pct = self.prices_pct(window)
        return prices_pct.groupby([prices_pct.Stock, prices_pct.Date.dt.year])[["pct_change"]]\
                         .std().dropna().reset_index(level=[0])\
                         .merge(tmp_fundamentals, left_on=["Stock", "Date"], \
                          right_on=["Ticker Symbol", "Period Ending"])\
                         .drop("Ticker Symbol", axis=1) 

    def prices_pct(self, window=5):
        price_pct_change_year = self.price_data\
                                    .groupby([self.price_data.Stock, self.price_data.Date.dt.year])["Close"]\
                                    .pct_change(window)
        prices_pct = self.price_data.copy()
        prices_pct["pct_change"] = price_pct_change_year
        return prices_pct.dropna()

