from enum import Enum


class Action(Enum):
    """
    Defines what actions the agent can take

    Values:
        buy: defines buying a stock
        hold: defines leaving a stock alone
        sell: defines selling a stock
    """
    buy = 1
    hold = 0
    sell = -1


class StockData():
    """
    Defines the data for a stock

    Attributes:
        date (str): date this data was recorded
        price (float): price of the stock
        rsi (float): the RSI of the stock
    """
    def __init__(self, date: str, price: float, pe: float) -> None:
        self.date = date
        self.price = price
        self.pe = pe
    

    def __str__(self) -> str:
        return f"StockData(date:{self.date}, price:{self.price}, pe:{self.pe})"


class Stock():
    """
    Defines a stock

    Attributes:
        ticker (str): name of the stock
        data (list[StockData]): historical data for the given stock
        prediction (Action): current prediction for the stock
    """
    def __init__(self, ticker: str, data: list[StockData], prediction: Action) -> None:
        self.ticker = ticker
        self.data = data
        self.prediciton = prediction


class Sector():
    """
    Defines a sector of the market

    Attributes:
        title (str): the title of the sector
        stocks (list[stock]): what stocks are included in the sector
        prediction (Action): current prediction for the sector
    """
    def __init__(self, title: str, stocks: list[Stock], prediction: Action) -> None:
        self.title = title
        self.stocks = stocks
        self.prediction = prediction 
