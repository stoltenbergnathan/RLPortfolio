from threading import Thread
from bs4 import BeautifulSoup
import requests
import concurrent.futures
import pandas
import yfinance as yf

def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]

def get_price_history(ticker: str):
    tl = yf.Ticker(ticker)
    history: pandas.DataFrame = tl.history(period = "1y", interval = "1d")
    datalist = []
    for i in history.index:
        datalist.append(history.loc[[i], ["High"]].values[0][0])
    return datalist

def get_percent_history(ticker: str):
    tl = yf.Ticker(ticker)
    history: pandas.DataFrame = tl.history(period = "1y", interval = "1d")
    dl = []
    for i in history.index:
        open = history.loc[[i], ["Open"]].values[0][0]
        close = history.loc[[i], ["Close"]].values[0][0]
        change = ((close - open) / open) * 100
        dl.append(change)
    return dl
