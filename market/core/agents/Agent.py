"""
This file represents the base Agent class. It could be extended to support more exotic agents
                                                                        such as EllipticalExponentiallyRecurring Agents;
"""

import numpy as np
import math
from math import exp, log

def sigmoid(x: float) -> float:
    """Computing sigmoid function"""
    return 1 / (1 + np.exp(-x))


class Agent:
    """
    The agent class general class for an agent in the market. It is the decision maker for buy and sell.
    Each agent is described by its weights - which will produce an estimate of the score and then the
    buy/sell logic will use this to decide if it participates when it has a chance to.
    """
    def __init__(self, config, rng, buy_positive:bool=False, idx: int = 0, name: str = "", exponential=True):
        self.idx = idx
        self.buy_positive = buy_positive
        self.name = name  # The agent's name
        self.cash = config.init_cash  # The balance in agent's bank
        self.wx = None    # The weight on the feature
        self.wp = 0.0    # The weight on the price
        self.b = 0.0    # Bias
        self.percent = config.percent  # The percent the agent wants to make to trade
        self.x = None         # The actual feature value(s)
        self.p = config.init_price        # The current price
        self.estimate = 0.0     # The current estimate
        self.positive_asset_prices = [] # A record of the positive holdings
        self.negative_asset_prices = [] #A record of the negative holdings
        self.determine_next_participation(-1, config=config, rng=rng)
        self.exponential = True
        self.y = -1

    def initialize_weights(self, feat_vec_size: int, rng: np.random._generator.Generator) -> None:
        """
            :brief Initialize the weights wx, wp and b with normal distribution, mean: 0 and std: 0.01
            :param feat_vec_size  feature vector size
        """
        self.wx = rng.normal(0.0, 1, feat_vec_size)
        self.wp = rng.normal(0.0, 1)
        self.b = rng.normal(0.0, 1)
    
    def calc_shares_by_price(self, p, b):
        
        if p >= 0.5:
            if p == 1.0:
                p = 0.9999
            num = np.log(-p / (p - 1))
            return num // b, True
        else:
            if p == 0:
                p = 0.0001
            num = np.log((1 - p) / p)
            return num // b, False
    
    def compute_estimate(self, config) -> float:
        """
            :brief Compute the estimate of the true price of the asset using the sigmoid function.
            :return the actual estimate, also stored internally in self.estimate
        """

        assert self.wx.shape == self.x.shape,"Weights and Features must have the same shape"

        # include the price and weight on the price while computing the estimate

        w = np.append(self.wx, [self.wp])
        x = np.append(self.x, [self.p])
        self.estimate = sigmoid(np.dot(w, x) + self.b)
        return self.estimate

    def would_buy_positive(self, trans_price: float, rng, liquidity_constant) -> bool:
        """
            :brief Determine whether an agent will buy a positive asset at a market given price (between 0 and 1)
            agent buys if it has cash and it believes it is finding value of self.percent % of the transaction price
            based on its estimate of value.
            :param trans_price: the transaction price.
            :return true if and only the agent would buy a positive asset.
        """

        if self.estimate < 0.5 or trans_price > self.estimate or self.cash < trans_price:
            return False, 0, 0
        else:
            val, pos = self.calc_shares_by_price(trans_price, 1 / liquidity_constant)
            if pos:
                positive_shares = val
                negative_shares = 0
            else:
                positive_shares = 0
                negative_shares = val
                
            if self.buy_positive:

                c1 = (liquidity_constant * log(exp(positive_shares / liquidity_constant) + \
                                            exp(negative_shares / liquidity_constant)))
                n1 = (liquidity_constant * log(exp((self.cash + c1)/liquidity_constant) - \
                                                exp(negative_shares/liquidity_constant))) - positive_shares
                val, pos =  self.calc_shares_by_price(self.estimate, 1 / liquidity_constant)

                if pos:
                    ps = val
                    ns = 0
                else:
                    ps = 0
                    ns = val
                
                n2 = negative_shares + ps - positive_shares
                num_shares = math.floor(min(n1, n2))

                c2 = (liquidity_constant * log(exp((positive_shares + num_shares)/ liquidity_constant) +
                                                exp((negative_shares) / liquidity_constant)))

                cost = c2 - c1
                return True, num_shares, cost
            else:
                return False, 0, 0

    def would_buy_negative(self, trans_price: float, rng, liquidity_constant) -> bool:
        """
            :brief Determine whether an agent will buy a negative asset at a market given price (between 0 and 1)
            :param trans_price: the transaction price.
            :return true if and only if the agent would buy a negative asset.
        """
        
        if self.estimate < 0.5 or trans_price > self.estimate or self.cash < trans_price:
            return False, 0, 0
        else:
            val, pos = self.calc_shares_by_price(trans_price, 1 / liquidity_constant)
            if pos:
                positive_shares = val
                negative_shares = 0
            else:
                positive_shares = 0
                negative_shares = val

            if not self.buy_positive:
                c1 = (liquidity_constant * log(exp(positive_shares / liquidity_constant) + \
                                            exp(negative_shares / liquidity_constant)))
                n1 = (liquidity_constant * log(exp((self.cash + c1)/liquidity_constant) - \
                                                exp(positive_shares/liquidity_constant))) - negative_shares
                val, pos =  self.calc_shares_by_price(1-self.estimate, 1 / liquidity_constant)

                if pos:
                    ps = val
                    ns = 0
                else:
                    ps = 0
                    ns = val
                
                n2 = positive_shares + ns - negative_shares
                num_shares = math.floor(max(min(n1, n2),0))

                c2 = (liquidity_constant * log(exp(positive_shares / liquidity_constant) +
                                exp((negative_shares + num_shares) / liquidity_constant)))
                cost = c2 - c1
                return True, num_shares, cost
            else:
                return False, 0, 0

    def try_buy_positive(self, trans_price: float, rng, liquidity_constant) -> bool:
        """
            :brief Try to have the agent buy a positive share
            :param trans_price: the share price for 1 share
            :return true if and only if a sale happens.
        """

        buy = False
        decision, num_shares, cost = self.would_buy_positive(trans_price, rng, liquidity_constant=liquidity_constant)
        if decision:
            self.cash -= cost
            self.positive_asset_prices.append(cost)
            buy = True
        return buy, num_shares, cost

    def try_buy_negative(self, trans_price: float, rng, liquidity_constant) -> bool:
        """
            :brief Try to have the agent sell a negative share
            :param trans_price: the share price for 1 share
            :return true if and only if a sale happens.
        """

        buy_negative = False
        decision, num_shares, cost = self.would_buy_negative(trans_price, rng, liquidity_constant=liquidity_constant)
        if decision:
            self.cash -= cost
            self.negative_asset_prices.append(cost)
            buy_negative = True
        return buy_negative, num_shares, cost

    def determine_next_participation(self, current_time, config):
        """
            :brief Determine the next epoch an agent will participate in the market given the current time.
                    Default behavior is next round.
            :param current_time: the current time.
            : param config: used in this method in child classes of agent, config object for the run
        """
        if self.exponential:
            if self.estimate < 0.5:
                self.next_participation_round = 20000000
            else:
                self.next_participation_round = current_time + math.ceil(
                    np.random.exponential(scale=(1 / config.lambda_value)))
        else:
            self.next_participation_round = current_time + 1

    #TODO: should definition of profit be at setting put into the config file?
    # these next three functions are interchangeable depending on the definition of profit desired
    def compute_profit_score(self, ground_truth: float) -> float:
        """
            :brief  Compute the profit an agent made including any left over cash.
                    profit = ground truth score of shares - total cost to obtain them
                    this was designed for use with a softmax ground truth
            :param ground_truth: true or false
            :return: the profit
        """
        if len(self.positive_asset_prices) > 0:
            return np.sum(np.repeat(ground_truth, len(self.positive_asset_prices)) - self.positive_asset_prices)
        elif len(self.negative_asset_prices) > 0:
            return np.sum(np.repeat(ground_truth, len(self.negative_asset_prices)) - self.negative_asset_prices)
        else:
            return 0

    def compute_profit_bool(self, ground_truth: bool) -> float:
        """
            :brief  Compute the profit an agent made including any left over cash.
                    profit = number of correct shares held at end of market - total cost to obtain them
            :param ground_truth: true or false
            :return: the profit
        """
        if len(self.positive_asset_prices) > 0:
            return np.sum(np.repeat(ground_truth, len(self.positive_asset_prices)) - self.positive_asset_prices)
        elif len(self.negative_asset_prices) > 0:
            if ground_truth == 0:
                return np.sum(np.repeat(1, len(self.negative_asset_prices)) - self.negative_asset_prices)
            else:
                return -np.sum(self.negative_asset_prices)
        else:
            return 0

    def compute_profit(self, ground_truth: bool) -> (str, float):
        """
            :brief  Compute the profit an agent made including any left over cash.
                    profit = number of correct shares held at end of market
            :param ground_truth: true or false
            :return: the profit
        """
        return_status = ""
        return_value = 0.0

        if len(self.positive_asset_prices) == 0 and len(self.negative_asset_prices) == 0:
            return_status = "ignore"

        elif ground_truth == True and len(self.positive_asset_prices) > 0:
            return_status = "profit"
            return_value = len(self.positive_asset_prices)

        elif ground_truth == False and len(self.negative_asset_prices) > 0:
            return_status = "profit"
            return_value = len(self.negative_asset_prices)

        elif ground_truth == True and len(self.positive_asset_prices) == 0:
            return_status = "loss"
            return_value = len(self.negative_asset_prices)

        elif ground_truth == False and len(self.negative_asset_prices) == 0:
            return_status = "loss"
            return_value = len(self.positive_asset_prices)

        return return_status, return_value

    def reset(self):
        """ :brief Reset the agent to a market start state. """
        self.positive_asset_prices = []
        self.negative_asset_prices = []
        self.next_participation_round = 0

    def mutate(self, sigma: float):
        """
            :brief Use a random normal to mutate the strategy. The input sigma determines how far the strategy can wiggle.
            :param sigma: the standard deviation of the step size.
        """
        delta_wx = np.random.normal(0, 1, self.wx.shape[0])*sigma
        self.wx += delta_wx
        self.wp += np.random.normal(0, 1)*sigma
        self.b += np.random.normal(0, 1)*sigma