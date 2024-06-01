from .BaseMarketModel import BaseMarketModel

#TODO: make use of this class to setup default usage for classifier
class MarketClassifier(BaseMarketModel):

    def __init__(self, config, train=True) -> None:
        super().__init__(config, train)