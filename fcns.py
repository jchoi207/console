import pandas as pd
import numpy as np
import yfinance as yf
import os

import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import datetime as dt
import matplotlib.dates as mdates


class portfolio():
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df["Account Type"] = df["Account Type"].apply(
            lambda x: x.split(" ")[1])
        df["Transaction Date"] = df["Transaction Date"].apply(
            lambda x: x.split(" ")[0])
        df = df.drop(
            columns=["Description", "Settlement Date", "Account #"])
        df = df.sort_values(by=["Action", "Transaction Date"], ascending=[
            True, False]).reset_index(drop=True)
        df = df[df["Action"].isin(["Buy", "DIV", "CON"])]
        df["Action"] = df["Action"].apply(lambda x: str(x).capitalize())

        self.tickers = np.unique(df[df["Action"] == "Buy"]["Ticker"])
        self.num_positions = len(self.tickers)
        self.df_questrade = df
        self.columns = ['Avg cost/share',  'Qty', 'Pos cost',	'Mkt prc',
                        'Mkt val', 'Mkt P/L', '% rtn (com.)', '% rtn', 'Divs', 'Tot P/L', 'Curr']

        self.df_portfolio = pd.DataFrame(
            index=self.tickers, columns=self.columns)

        # pd.display(self.df_questrade)

        self.summarize_portfolio()
        self.summarized_dividends()
        self.overall_statistics()

    def get_todays_date(self):
        today = pd.Timestamp.today().normalize()
        if today.isoweekday() > 5:
            today -= BDay(1)
            return today
        else:  # ignoring holidays for now
            return today

    def get_yesterdays_date(self):
        return self.get_todays_date() - BDay(1)

    def summarize_portfolio(self):

        def calculate_metrics(df_ticker):

            ticker = df_ticker["Ticker"].iloc[0]

            avg_cost = ((-df_ticker["Gross Amount"].sum() -
                        df_ticker["Commission"].sum()) / df_ticker["Quantity"].sum())

            self.df_portfolio.loc[ticker, "Avg cost/share"] = avg_cost
            num_shares = df_ticker["Quantity"].sum()

            self.df_portfolio.loc[ticker, "Qty"] = num_shares
            self.df_portfolio.loc[ticker, "Pos cost"] = avg_cost * num_shares
            today = self.get_todays_date()
            mkt_prc = yf.Ticker(ticker).history(
                start=today, end=today + dt.timedelta(1))["Close"].iloc[0]

            self.df_portfolio.loc[ticker, "Mkt prc"] = mkt_prc
            self.df_portfolio.loc[ticker, "Mkt val"] = mkt_prc * num_shares
            self.df_portfolio.loc[ticker,
                                  "Mkt P/L"] = (mkt_prc - avg_cost) * num_shares
            self.df_portfolio.loc[ticker, "% rtn (com.)"] = (
                mkt_prc/avg_cost - 1) * 100
            self.df_portfolio.loc[ticker, "% rtn"] = (
                mkt_prc/(-df_ticker["Gross Amount"].sum() / df_ticker["Quantity"].sum()) - 1) * 100

            self.df_portfolio.loc[ticker,
                                  "Curr"] = df_ticker["Currency"].iloc[0]

        df_buy = self.df_questrade[self.df_questrade["Action"] == "Buy"]

        for ticker in self.tickers:
            calculate_metrics(df_buy[df_buy["Ticker"] == ticker])
        #  display(self.df_portfolio)

    def display_df(self, df):
        df_copy = df.copy()

        for col in self.columns:
            if col == 'Curr':
                pass
            else:
                df_copy[col] = df[col].apply(
                    lambda x: float("{:.2f}".format(float(x))))
        # pd.display(df_copy)

    def summarized_dividends(self):
        keys = ['BEP.UN', '.BN', '.BNS', '.ENB',
                'A036970', '.RY', '.TD', 'V007563']
        translations = ['BEP-UN.TO', 'BN.TO', 'BNS.TO',
                        'ENB.TO', 'GOOGL', 'RY.TO', 'TD.TO', 'VOO']
        translation_dict = dict(zip(keys, translations))

        def replace_ticker_name(key):
            return translation_dict[key]

        df_div = self.df_questrade[self.df_questrade["Action"] == "Div"].copy()

        df_div["Ticker"] = df_div["Ticker"].apply(
            lambda key: replace_ticker_name(key))
        self.dividend_sums_by_ticker = df_div.groupby("Ticker")[
            "Net Amount"].sum()
        # self.dividend_sums_by_curr = df_div.groupby("Currency")["Net Amount"].sum()
        self.df_portfolio["Divs"] = self.dividend_sums_by_ticker
        self.df_portfolio["Tot P/L"] = self.df_portfolio["Mkt P/L"] + \
            self.df_portfolio["Divs"]
        # self.display_df(self.df_portfolio)

    def get_exchange(self, date):
        exchange = "USDCAD=X"
        rate = yf.download(exchange, date, pd.to_datetime(
            date) + dt.timedelta(1))["Adj Close"].iloc[0]
        return rate

    def adjust_currency(self, df, col):
        sums = df.groupby("Curr")[col].sum()
        return sums.get("USD") * self.get_exchange(self.get_todays_date()) + sums.get("CAD")

    def overall_statistics(self):

        self.market_pl = self.adjust_currency(self.df_portfolio, "Mkt P/L")
        self.pos_cost = self.adjust_currency(self.df_portfolio, "Pos cost")
        self.total_pl = self.adjust_currency(self.df_portfolio, "Tot P/L")
        self.total_div = self.adjust_currency(self.df_portfolio, "Divs")
        self.todays_exchange = self.get_exchange(self.get_todays_date())
        self.last_updated_date = self.get_todays_date()
        self.get_plots()

        # print("Total Market P/L: {:.2f}\nTotal Book Cost: {:.2f}\nTotal Dividends: {:.2f}\nTotal P/L including Dividends: {:.2f}".format(
        #     self.market_pl,
        #     self.pos_cost,
        #     self.total_div,
        #     self.total_pl
        # ))

    def plot_portfolio(self, start, end):

        time_series = yf.download(tickers=list(
            self.tickers), start=start, end=end)["Adj Close"]
        time_series = time_series.ffill()
        quantities = self.df_portfolio["Qty"]

        rate = self.get_exchange(end)

        for ticker in self.tickers:
            time_series[ticker] *= quantities[ticker]
            if self.df_portfolio.loc[ticker, "Curr"] == "USD":
                time_series[ticker] *= rate

        time_series["Portfolio Value"] = time_series.apply(
            lambda row: sum(row[ticker] for ticker in self.tickers), axis=1)

        plt.plot(time_series.index, time_series["Portfolio Value"])
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("CAD$")
        plt.title("Market Value of Portfolio")

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        ax = plt.gca()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].set_title("Market P/L")
        axs[0].grid(zorder=-1)
        df_copy = self.df_portfolio.copy().sort_values(by=["Mkt P/L"])
        colors = ['red' if x < 0 else 'green' for x in df_copy["Mkt P/L"]]
        axs[0].bar(df_copy.index, df_copy["Mkt P/L"], zorder=5, color=colors)

        axs[1].set_title("Total P/L")
        axs[1].grid()
        df_copy = self.df_portfolio.copy().sort_values(by=["Tot P/L"])
        colors = ['red' if x < 0 else 'green' for x in df_copy["Tot P/L"]]

        axs[1].bar(df_copy.index, df_copy["Tot P/L"], zorder=5, color=colors)

        plt.xlabel("Tickers")
        plt.ylabel("P/L")

    def get_plots(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(current_dir, 'static', 'img')
        self.daily_pl = []
        self.daily_close = []
        self.prev_close = []
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for ticker in self.tickers:
            start = self.get_todays_date()
            start_str = start.strftime('%Y-%m-%d')
            tmr = start + BDay(1)
            yest = start - BDay(1)
            prev_close = yf.Ticker(ticker).history(
                start=yest, end=start, interval='1d')['Close']
            data = yf.Ticker(ticker).history(
                start=start, end=tmr, interval='5m')['Close']
            red = prev_close.iloc[0] > data.iloc[-1]
            xmin = data.index[0]
            xmax = data.index[-1]
            self.daily_close.append(data.iloc[-1])
            self.prev_close.append(prev_close.iloc[0])
            self.daily_pl.append(data.iloc[-1]/prev_close.iloc[0] - 1)
            if os.path.exists(os.path.join(img_dir, f'{ticker}_{start_str}_plot.png')):
                print(">> File already exists. Skipping")
            else:
                print(f">> Plotting for {ticker}")
                
                plt.plot(data.index, data, c='r' if red else 'g')
                plt.hlines(y=prev_close, xmin=xmin, xmax=xmax,
                           colors='k', linestyles='dashed')
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.savefig(
                    os.path.join(img_dir, f'{ticker}_{start_str}_plot.png'), dpi=300
                )
                plt.clf()


# # my_port = portfolio("Activities_for_01Dec2019_to_13Jul2024.csv")
# # print(my_port.df_portfolio)

# # my_port.plot_portfolio(start="2024-05-01", end="2024-07-12")


# my_port = portfolio(
#     "Activities_for_01Dec2019_to_13Jul2024.csv").df_portfolio.to_dict(orient='index')
# print(my_port)
