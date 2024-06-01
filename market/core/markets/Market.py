from .transaction import Transaction
from math import exp, log
import numpy as np


class Market:
    """
    The market produces the estimate of the score based on the transaction history
    of all of its agents. The market has a current price(score) for the paper, all of
    its agents. It will compute its top agents in terms of profit so that it can
    continue to evolve.
    """

    def __init__(self, config, market_id, agents=None, transactions=None, spot_prices=None, market_duration=None,
                 time_now=0, positive_shares=0.0, negative_shares=0.0, label=None):
        """
        :param agents: Agent objects corresponding to this market
        :param transactions: Records all the transactions made in this market
        :param: spot_prices Records all the spot-prices (or market prices) after a transaction
        :param market_duration: Total number of market rounds
        :param time_now: current market-round
        :param positive_shares: Total number of positive shares bought
        :param negative_shares: Total number of negative shares bought
        """
        self.market_id = market_id
        self.config = config
        if agents:
            self.agents = agents
        else:
            self.agents = []
        if transactions:
            self.transactions = transactions
        else:
            self.transactions = []
        if spot_prices:
            self.spot_prices = spot_prices
        else:
            self.spot_prices = []
        self.market_duration = config.market_duration
        self.time_now = time_now
        self.current_price = config.init_price
        self.positive_shares = positive_shares
        self.negative_shares = negative_shares
        self.liquidity_const = self.config.liquidity_constant
        self.top_agents = np.empty(0)
        self.gt = label

    def add_agent(self, agent):
        """
            :brief Add an agent into the market
            :param agent: agent that needs to be added to the market
        """
        self.agents.append(agent)

    def get_current_price(self):
        """
            :brief Get current price for positive asset
            :return Returns the current market price (for positive asset)
        """
        return self.current_price

    def compute_price(self):
        """
            :brief Compute current positive asset price using the formula from PNAS
            :return Returns the current price and also, updates the price in the Market class (for visibility)
        """
        self.current_price = (exp(self.positive_shares / self.liquidity_const) /
                              (exp(self.positive_shares / self.liquidity_const) +
                               exp(self.negative_shares / self.liquidity_const)))
        return self.current_price

    def compute_positive_purchase_price(self):
        """
            :brief Compute the cost of buying 1 share of a positive asset.
            :return The price of 1 share of the positive asset.
        """
        c1 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        c2 = (self.liquidity_const * log(exp((self.positive_shares + 1) / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        return c2 - c1

    def compute_negative_purchase_price(self):
        """
            :brief Compute the cost of buying 1 share of a negative asset.
            :return The price of 1 share of the negative asset.
        """
        c1 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        c2 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp((self.negative_shares + 1) / self.liquidity_const)))
        return c2 - c1

    def compute_positive_sale_price(self):
        """
            :brief Compute the value of selling one positive share (back to the market).
            :return The selling price of 1 share of the positive asset.
        """
        c1 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        c2 = (self.liquidity_const * log(exp((self.positive_shares - 1) / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        return c1 - c2

    def compute_negative_sale_price(self):
        """
            :brief Compute the value of selling one negative share (back to the market).
            :return The selling price of 1 share of the negative asset.
        """
        c1 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp(self.negative_shares / self.liquidity_const)))
        c2 = (self.liquidity_const * log(exp(self.positive_shares / self.liquidity_const) +
                                         exp((self.negative_shares - 1) / self.liquidity_const)))
        return c1 - c2

    def process_agent(self, agent_id: int, time_now, rng):
        """
            :brief Process an agent during a simulated market run
                  NOTE: SELLING OF AN ASSET IS CURRENTLY NOT SUPPORTED FOR SIMPLICITY IN THE MARKET.
            :param agent_id: holds the agent-id of an Agent that is about to participate in this market round
            :return a Boolean denoting whether a transaction occurred or not
        """
        self.time_now = time_now
        transaction_occured = False

        agent = self.agents[agent_id]
        # Tell the agent what the current market price is then make it compute its estimate
        agent.p = self.current_price
        agent.compute_estimate(self.config)

        # Compute all possible transaction prices
        #positve_purchase_price = self.compute_positive_purchase_price()
        #negative_purchase_price = self.compute_negative_purchase_price()
        # positive_sale_price = self.compute_positive_sale_price()
        # negative_sale_price = self.compute_negative_sale_price()

        """Ask if the agent would like to buy any shares first. If so, update shares and
           compute the new market price. If no buying occurs ask if the agent wants to sell
           some of its shares. Record all transactions that occur."""
        """ NOTE: SELLING IS CURRENTLY NOT SUPPORTED FOR SIMPLICITY OF THE MARKET...
                    UNCOMMENT CODE IF SELLING IS NEEDED """
        pdecision, pnum_shares, pcost = agent.try_buy_positive(self.current_price, rng, self.liquidity_const)
        ndecision, nnum_shares, ncost = agent.try_buy_negative(self.current_price, rng, self.liquidity_const)

            
        if pdecision:
            self.positive_shares += pnum_shares
            self.compute_price()

            self.transactions.append(Transaction(config=self.config,
                                                 agent=agent, 
                                                 t_type="positive_buy", 
                                                 price=self.current_price, 
                                                 num_shares=pnum_shares,
                                                 positive_price=pcost, 
                                                 negative_price=0, 
                                                 time=self.time_now))
            transaction_occured = True
        
       
        elif ndecision:
            self.negative_shares += nnum_shares
            self.compute_price()
            self.transactions.append(Transaction(config=self.config,
                                                 agent=agent, 
                                                 t_type="negative_buy", 
                                                 price=self.current_price, 
                                                 num_shares=nnum_shares,
                                                 positive_price=0, 
                                                 negative_price=ncost, 
                                                 time=self.time_now))
            transaction_occured = True

        # elif agent.try_sell_positive(positive_sale_price):
        #     self.positive_shares -= 1
        #     self.compute_price()
        #     self.transactions.append(Transaction(agent, "positive_sell", positive_sale_price, self.time_now))
        #     transaction_occured = True
        #
        # elif agent.try_sell_negative(negative_sale_price):
        #     self.negative_shares -= 1
        #     self.compute_price()
        #     self.transactions.append(Transaction(agent, "negative_sell", negative_sale_price, self.time_now))
        #     transaction_occured = True

        return transaction_occured

    def run_market(self, agent_id: int, time_now, rng) -> None:
        """

        :param agent_id: holds the agent-id of an Agent that is about to participate in this market round
        :param time_now: holds the current time (or market round)
        :return: None
        """


        # Push the first spot price onto the list of spot prices, also compute the price
        self.spot_prices.append(self.current_price)
        """ Set the current_price """
        self.agents[agent_id].p = self.current_price

        if self.process_agent(agent_id, time_now, rng):
            self.spot_prices.append(self.current_price)

    def compute_top_agents_new(self, ground_truth: float, m: int):
        """
        :brief Compute the top "m" highest profit agents, or least loss agents.
        If any agents were profitable, this will this will set self.top_agents to the
        top m of those. Else it returns the top m least loss agents.
        :param ground_truth: ground_truth corresponding to the input training data
        :param m: an integer that specifies the number of top-performing agents that needs to estimated
        """
        profit_agents = []
        loss_agents = []
        for agent_id in range(len(self.agents)):
            pl = self.agents[agent_id].compute_profit_score(ground_truth)
            if pl > 0:
                profit_agents.append(np.array([agent_id, pl]))
            elif pl < 0:
                loss_agents.append(np.array([agent_id, pl]))
        if len(profit_agents) > 0:
            profit_agents = np.array(profit_agents)
            # agent ids for top "m" profitable agents
            self.top_agents = profit_agents[profit_agents[:, 1].argsort()[::-1][:m], 0]
        elif len(loss_agents) > 0:
            loss_agents = np.array(loss_agents)
            # agent ids for top "m" least loss agents
            self.top_agents = loss_agents[loss_agents[:, 1].argsort()[:m], 0]
        else:
            self.top_agents = np.array([])
        return

    # deprecated - kept around because it was part of hand off from first paper
    def compute_top_agents(self, ground_truth: bool, m: int):
        """
            :brief Compute the top "m" performing agents (with highest profit and least loss)
            the top "m" in most profit and the top "m" in least loss are included in the list
            of top performers. This is used when participation in the market is low because it encourages
            participating over non-participation
            :param ground_truth: ground_truth corresponding to the input training data
            :param m: an integer that specifies the number of top-performing agents that needs to estimated
        """
        # should contain two columns - first column: agentID, second column: profit/loss
        profit_agents = []
        loss_agents = []

        for agent_id in range(len(self.agents)):
            ret_status, pl = self.agents[agent_id].compute_profit(ground_truth)
            if ret_status == 'profit':
                profit_agents.append(np.array([agent_id, pl]))
            elif ret_status == 'loss':
                loss_agents.append(np.array([agent_id, pl]))

        # pick top m most-profitable and top m least-loss agents => where m is the number of data-points
        if len(profit_agents) > 0:
            profit_agents = np.array(profit_agents)

            # agent ids for top "m" profitable agents
            profit_agents = profit_agents[profit_agents[:, 1].argsort()[::-1][:m], 0]
        if len(loss_agents) > 0:
            loss_agents = np.array(loss_agents)

            # agent ids for top "m" least loss agents
            loss_agents = loss_agents[loss_agents[:, 1].argsort()[:m], 0]

        self.top_agents = np.append(profit_agents, loss_agents)
        return

    def compute_top_agents3(self, ground_truth, m):
        """
            :brief Compute the top "m" highest profit agents, or least loss agents.
            If any agents were profitable, this will set self.top_agents to the top m
            of those, if there were less than m profitable agents, it will also return the
            least loss agents up to m total agents.
            :param ground_truth: ground_truth corresponding to the input training data
            :param m: an integer that specifies the number of top-performing agents that needs to estimated
        """
        profit_agents = []
        loss_agents = []
        for agent_id in range(len(self.agents)):
            pl = self.agents[agent_id].compute_profit_bool(ground_truth)
            if pl > 0:
                profit_agents.append(np.array([agent_id, pl]))
            elif pl < 0:
                loss_agents.append(np.array([agent_id, pl]))
        if len(profit_agents) > 0:
            profit_agents = np.array(profit_agents)

            # agent ids for top "m" profitable agents
            profit_agents = profit_agents[profit_agents[:, 1].argsort()[::-1], 0]
        if len(loss_agents) > 0:
            loss_agents = np.array(loss_agents)

            # agent ids for top "m" least loss agents
            loss_agents = loss_agents[loss_agents[:, 1].argsort(), 0]
        self.top_agents = np.append(profit_agents, loss_agents)[:m]
        return


    def compute_total_profits(self, ground_truth: bool):
        """
            :brief Compute the total profits made by each agent across all the markets
            :param ground_truth: ground_truth corresponding to the input training data
            :return normalized profits of all the agents in the market
        """
        profits = np.empty((0, self.config.number_of_agents))

        for agent in self.agents:
            profits = np.append(profits, agent.compute_profit(ground_truth))

        return profits / np.linalg.norm(profits)

    def output_market(self) -> dict:
        """
        Prepare a dictionary of transactions that occurred in the market that can be written out to json file
        :return: dictionary of market transactions
        """
        dic = {}
        dic['market_id'] = self.market_id
        trans = []
        for i in self.transactions:
            trans.append(i.output_transaction())
        dic['transactions'] = trans
        return dic