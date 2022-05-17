from threading import Thread
from bs4 import BeautifulSoup
import requests
import concurrent.futures
import pandas
import yfinance as yf
from type import *

def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]

def get_price_history(ticker: str):
    tl = yf.Ticker(ticker)
    history: pandas.DataFrame = tl.history(period = "1y", interval = "1wk")
    datalist = []
    for i in history.index:
        datalist.append(StockData(date=str(i), high=history.loc[[i], ["High"]].values[0][0]))
    print(Stock(ticker=ticker, data=datalist, prediction=Action.hold))

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(get_price_history, get_SP_tickers())