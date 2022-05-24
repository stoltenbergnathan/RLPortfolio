from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas
import yfinance as yf
from classes import *


def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]


def get_ticker_history(ticker: str, period: str = "1y"):
    tl = yf.Ticker(ticker)
    return tl.history(interval = "1d", start = "2020-01-01", end = "2021-01-01")


def get_price_history(history, min, max):
    price_hist = []
    for i in history.index[min:max]:
        high = history.loc[[i], ["High"]].values[0][0]
        price_hist.append(high)
    return price_hist


def get_diff_hist(history, min, max):
    dl = []
    for i in history.index[min:max]:
        open = history.loc[[i], ["Open"]].values[0][0]
        close = history.loc[[i], ["Close"]].values[0][0]
        change = ((close - open) / open) * 100
        if np.isnan(change):
            return []
        dl.append(change)
    return dl


def get_RSI(price_history: list[float]):
    gains = []
    losses = []
    window = []

    for i, price in enumerate(price_history):
        if i == 0:
            window.append(price)
            continue

        difference = price_history[i] - price_history[i - 1]

        if difference > 0:
            gain = difference
            loss = 0
        elif difference < 0:
            gain = 0
            loss = abs(difference)
        else:
            gain = 0
            loss = 0
        
        gains.append(gain)
        losses.append(loss)
    
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi
