from bs4 import BeautifulSoup
import requests
import aiohttp
import asyncio
from enum import Enum
import time

class State():
    def __init__(self) -> None:
        self.price = 0

class Action(Enum):
    buy = "buy"
    sell = "sell"
    hold = "hold"

class Stock():
    def __init__(self, ticker: str, price: float) -> None:
        self.ticker = ticker
        self.price = price

    def display(self):
        print("--- " + self.ticker + " " + str(self.price))

def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]


async def get_stock_price(ticker: str) -> float:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://finance.yahoo.com/quote/{ticker}") as resp:
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            try:
                price = soup.find('fin-streamer', {"class": "Fw(b) Fz(36px) Mb(-4px) D(ib)"})['value']
            except Exception as e:
                print(e)
                return -1
            print(ticker + " " + str(price))
            return float(price)

async def main():
    tickers = get_SP_tickers()
    tasks = [asyncio.create_task(get_stock_price(t)) for t in tickers]
    groups = await asyncio.gather(*tasks)
    stocks = [Stock(tickers[i], groups[i]) for i in range(len(tickers))]
    for i in stocks:
        i.display()
    

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.run_until_complete(asyncio.sleep(1))
loop.close()