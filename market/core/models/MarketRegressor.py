from .BaseMarketModel import BaseMarketModel

#TODO: make use of this class to setup default usage for regressor
class MarketRegressor(BaseMarketModel):

    def __init__(self, config, train=True) -> None:
        super().__init__(config, train)