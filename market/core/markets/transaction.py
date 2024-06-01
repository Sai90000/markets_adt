from datetime import datetime

class Transaction:
    """
    The transaction class keeps key data for all transactions that occur in a market:
    * time
    * agent
    * type
    * price
    * agent cash
    """
    def __init__(self, config, agent=None, t_type=None, price=None, positive_price=None, negative_price=None, time=None, num_shares=0):
        self.timestamp = datetime.now()
        self.agent = agent
        self.agent_id = self.agent.name
        self.t_type = t_type
        self.time = time
        self.price = price
        self.positive_purchase_price = positive_price
        self.negative_purchase_price = negative_price
        self.cash = self.agent.cash
        self.agent_estimate = self.agent.compute_estimate(config)
        self.buy_positive = self.agent.buy_positive
        self.num_shares = num_shares


    def output_transaction(self)-> dict:
        """
        Put all relevant transaction information into a dictionary
        :return: dictionary describing the key elements of the transaction
        """
        dic = {}
        dic['agent_id'] = vars(self)['agent_id']
        dic['t_type'] = vars(self)['t_type']
        dic['time'] = vars(self)['time']
        dic['price'] = vars(self)['price']
        # we can update this if synthetic agents ever buy multiple/fractional shares (as in matrix version of market)
        dic['num_shares'] = 1
        dic['date_time'] = vars(self)['timestamp']
        if 'negative' in self.t_type.lower():
            dic['negative_purchase_price'] = vars(self)['negative_purchase_price']
            dic['transaction_price'] = dic['num_shares'] * dic['negative_purchase_price']
            dic['positive_purchase_price'] = 'None'
        else:
            dic['positive_purchase_price'] = vars(self)['positive_purchase_price']
            dic['transaction_price'] = dic['num_shares'] * dic['positive_purchase_price']
            dic['negative_purchase_price'] = 'None'
        dic['cash'] = vars(self)['cash'] + dic['transaction_price']
        dic['agent_type'] = vars(self)['buy_positive']
        dic['agent_estimate'] = vars(self)['agent_estimate']
        dic['num_shares'] = vars(self)['num_shares']
        return dic
