from datetime import datetime
from bs4 import BeautifulSoup
import requests
import aiohttp
import asyncio
from type import *

def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]


def get_price(page: str) -> float:
    soup = BeautifulSoup(page, 'html.parser')
    try:
        price = soup.find('fin-streamer', {"class": "Fw(b) Fz(36px) Mb(-4px) D(ib)"})['value']
    except Exception as e:
        print(e)
        return -1
    return float(price)


def get_pe(page: str) -> float:
    soup = BeautifulSoup(page, 'html.parser')
    try:
        pe = soup.find('td', {"data-test": "PE_RATIO-value"}).text
        pe = float(pe)
    except Exception as e:
        print(e)
        return -1
    return pe

async def get_stock_info(ticker: str) -> StockData:
    async with aiohttp.ClientSession() as session:
        date = datetime.now()
        async with session.get(f"https://finance.yahoo.com/quote/{ticker}") as resp:
            page = await resp.text()
            price = get_price(page)
            pe = get_pe(page)
            return StockData(str(date), price, pe)


async def main():
    start = datetime.now()
    tickers = get_SP_tickers()
    tasks = [asyncio.create_task(get_stock_info(t)) for t in tickers]
    groups = await asyncio.gather(*tasks)
    end = datetime.now()
    for group in groups:
        print(group)
    print(f"That took {end - start} seconds")
    

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.run_until_complete(asyncio.sleep(1))
loop.close()