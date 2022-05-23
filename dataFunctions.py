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


def get_percent_history(ticker: str):
    print(ticker)
    tl = yf.Ticker(ticker)
    history: pandas.DataFrame = tl.history(period = "1y", interval = "1d")
    dl = []
    for i in history.index:
        open = history.loc[[i], ["Open"]].values[0][0]
        close = history.loc[[i], ["Close"]].values[0][0]
        change = ((close - open) / open) * 100
        if np.isnan(change):
            return []
        dl.append(change)
    return dl
