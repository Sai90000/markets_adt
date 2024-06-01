"""
This file is currently being used for multiple purposes:
    i)   Build or initialize agent weights
    ii)  Build or initialize markets with data and agents of a particular type -
                                                                        say, EllipticalExponentiallyRecurring agents
    iii) Train the agents in a batch-wise manner
    iv)  Update agent-weights using a Genetic Algorithm

Moving ahead, we could split this huge file into two or more parts for modularity!
"""

from sklearn.metrics import mean_squared_error, log_loss
import numpy as np
from numpy.random import default_rng
import math
from ..agents import EllipticalExponentiallyRecurringAgent
from ..markets import Market
from multiprocessing.dummy import Pool as ThreadPool
import time



def calc_shares_by_price(p, b):
    if p >= 0.5:
        num = np.log(-p / (p - 1))
        return num // b, True
    else:
        num = np.log((1 - p) / p)
        return num // b, False


class GeneticAlgorithm:

    def __init__(self, config, init_rand_pos=None, test=False):
        """
            :param init_rand_pos: a tuple comprising of a data point index, it's label depicting agent's buying
            specialization (positive or negative type of shares), corresponding start configuration (min & max radius)
            :type init_rand_pos: list of tuples
            :param test: a boolean to determine if it is test or train phase
        """

        self.config = config
        self.feature_vectors = []  # raw data - features
        self.ground_truth = []  # expected output values
        self.markets = []  # set of markets - 1 feature per row
        self.objective_loss_history = []  # holds objective function loss history
        self.agent_weights = []  # holds wx --> includes hypersphere center, wp, b and buy/sell
        self.event_queue = []  # holds all the events
        self.global_clock = 0  # time-step - limited by the market_durations number from the config file
        self.test_mode = test
        if init_rand_pos:
            self.init_rand_pos = init_rand_pos

        self.batch_loss = 0
        self.correct_count = 0

    def mean_squared_objective(self, y_pred: list) -> float:
        """
            :brief Compute the mean squared error between the non-binary (ground-truth) y_true
             and (predictions) y_pred
            :param y_pred: predicted output values
        """
        print(self.ground_truth)
        print(y_pred)
        return mean_squared_error(self.ground_truth, y_pred)

    def log_loss_objective(self, y_pred: list) -> float:
        """
            :brief Compute the log-loss or cross-entropy loss between the binary (ground-truth) y_true
            and (predictions) y_pred
            :param y_pred: predicted output values
        """
        return log_loss(self.ground_truth, y_pred, labels=[0.0,1.0])

    def compute_objective(self):
        """
            :brief Compute the objective function.
            Automatically toggles between log-loss (binary ground truth) and MSE ("softmax" ground truth)
        """
        y_pred = []
        for market in self.markets:
            y_pred.append(market.current_price)
        z = (np.array(self.ground_truth) == 0)
        o = (np.array(self.ground_truth) == 1)
        log_loss = (o + z).all()
        if log_loss:
            return self.log_loss_objective(y_pred)
        else:
            return self.mean_squared_objective(y_pred)

    #TODO: remove if we remove ER Agent Class
    def create_er_agent(self, agent_id: int, num_features: int, init_cash: float,
                        lambda_value: float, one_data_point: np.array, build_weights: bool = False):
        """
            :brief Create ExponentiallyRecurringAgent (currently not used as we have more advanced -
                                                                            EllipticallyExponentiallyRecurringAgent)

            :param agent_id: agent id number
            :param num_features: total number of input features
            :param init_cash: initial cash ( as seen in the "config.py" file )
            :param lambda_value: parameter for exponential distribution
            :param one_data_point: data point corresponding to this agent
            :param build_weights: a boolean denoting whether to build weights from scratch or
                                                                                        use the existing agent weights
        """
        agent = ExponentiallyRecurringAgent(self.config)
        agent.id = agent_id
        agent.name = "Agent_" + str(agent_id)
        agent.p = self.config.init_price
        agent.cash = init_cash

        if not build_weights:
            agent.wx = self.agent_weights[agent_id][:-2]
            agent.wp = self.agent_weights[agent_id][-2]
            agent.b = self.agent_weights[agent_id][-1]
        else:
            agent.initialize_weights(num_features)

        agent.lambda_value = lambda_value
        agent.x = one_data_point
        return agent

    def create_eer_agent(self, agent_id: int, init_cash: float, one_data_point: np.array,
                         rng: np.random._generator.Generator,                         
                         rand_pos: np.ndarray = None, build_weights: bool = False, label = None):
        """
            :brief Create EllipticallyExponentiallyRecurringAgent

            :param agent_id: agent id number
            :param init_cash: initial cash ( as seen in the "config.py" file )
            :param one_data_point: data point corresponding to this agent
            :param rand_pos: holds start-configuration -- min and max radius for every agent
            :param build_weights: a boolean denoting whether to build weights from scratch or
                                                                                        use the existing agent weights
        """
        agent = EllipticalExponentiallyRecurringAgent(config=self.config, rng=rng)
        agent.id = agent_id
        agent.name = "Agent_" + str(agent_id)
        agent.p = self.config.init_price
        agent.cash = init_cash
        agent.x = one_data_point
        agent.y = label

        if not build_weights:
            agent.wx = self.agent_weights[agent_id][:-3]
            agent.wp = self.agent_weights[agent_id][-3]
            agent.b = self.agent_weights[agent_id][-2]
            agent.buy_positive = self.agent_weights[agent_id][-1]
        else:  # the execution should not come here
            agent.init_random_weights(rng=rng)
            agent.buy_positive = rand_pos[1]

        if not self.test_mode:
            agent.min_radius = rand_pos[2][0]
            agent.max_radius = rand_pos[2][1]
            # agent.init_position(self.feature_vectors[rand_pos[0]])

        return agent

    def build_agent_weights(self, num_agents: int, rng: np.random._generator.Generator):
        """
        :brief Build or initialize Agent Weights based on the total agents count
        :TODO CHANGE THIS METHOD TO ONLY UPDATE THE ATTRIBUTE AGENT WEIGHTS INSTEAD OF OBJECT CREATION EVERYTIME

        :param num_agents: total number of agents
        """

        # BUILD AGENT WEIGHTS:
        build_weights = True
        if self.config.agent_type == "ExponentiallyRecurring":
            data_point = np.array(self.feature_vectors[0]).reshape(1, -1)
            for agent_id in range(num_agents):
                agent = self.create_er_agent(agent_id, len(self.feature_vectors[0]), self.config.init_cash,
                                             self.config.lambda_value, data_point, build_weights)
                self.agent_weights.append(agent.wx)
                self.agent_weights[agent_id] = np.append(self.agent_weights[agent_id], agent.wp)
                self.agent_weights[agent_id] = np.append(self.agent_weights[agent_id], agent.b)

        elif self.config.agent_type == "EllipticalExponentiallyRecurring":
            id = 0
            radius_arr = rng.uniform(low=self.init_rand_pos[id][2][0], high=self.init_rand_pos[id][2][1],
                                           size=self.config.no_of_agents_per_market)

            for agent_id in range(num_agents):
                init_weights = np.zeros(shape=0)
                # initialize as a hyper-sphere => same component radius for all the dimensions
                if agent_id > 0 and agent_id % self.config.no_of_agents_per_market == 0:
                    id += 1
                    radius_arr = rng.uniform(low=self.init_rand_pos[id][2][0], high=self.init_rand_pos[id][2][1],
                                                   size=self.config.no_of_agents_per_market)

                component_radius = radius_arr[agent_id % self.config.no_of_agents_per_market]
                h_i = 0
                for i in range(2 * len(self.feature_vectors[0])):
                    if i % 2 == 1:
                        # this will be replaced with the center (data-point i)
                        init_weights = np.append(init_weights, self.feature_vectors[self.init_rand_pos[id][0]][h_i])
                        h_i += 1
                    else:
                        init_weights = np.append(init_weights, component_radius)

                wx = init_weights.reshape(-1, 1)
                wp = rng.normal(0, 1)
                b = rng.uniform(0.01, 5)

                self.agent_weights.append(wx)
                self.agent_weights[agent_id] = np.append(self.agent_weights[agent_id], wp)
                self.agent_weights[agent_id] = np.append(self.agent_weights[agent_id], b)
                self.agent_weights[agent_id] = np.append(self.agent_weights[agent_id], self.init_rand_pos[id][1])

        return

    def build_market(self, market_id: int, rng: np.random._generator.Generator, label):
        """
            :brief Build or initialize a Market with agents of a particular type, say, EllipticalExponentiallyRecurring
            :param market_id: denotes a row-id in the input features list
            :param rng: numpy random generator for reproducible code
        """
        if self.config.init_price != 0.5:
            val, pos = calc_shares_by_price(self.config.init_price, 1 / self.config.liquidity_constant)
            if pos:
                positive_shares = val
                negative_shares = 0
            else:
                positive_shares = 0
                negative_shares = val
        else:
            positive_shares = 0
            negative_shares = 0
        build_weights = False
        market = Market(config=self.config, 
                        market_id=market_id, 
                        positive_shares=positive_shares, 
                        negative_shares=negative_shares, 
                        label=label)
        
        market.market_duration = self.config.market_duration

        ellipse_center_pos = 0
        rand_pos = None
        data_point = np.array(self.feature_vectors[market_id])

        if self.config.agent_type == "EllipticalExponentiallyRecurring":
            if not self.test_mode:
                rand_pos = self.init_rand_pos[ellipse_center_pos]

        for agent_id in range(self.config.number_of_agents):
            if self.config.agent_type == "ExponentiallyRecurring":

                market.agents.append(self.create_er_agent(agent_id, len(self.feature_vectors[0]),
                                                          self.config.init_cash, self.config.lambda_value, data_point,
                                                          build_weights))

            elif self.config.agent_type == "EllipticalExponentiallyRecurring":
                if not self.test_mode:
                    if agent_id % self.config.no_of_agents_per_market == 0 and agent_id > 0:
                        ellipse_center_pos += 1
                        rand_pos = self.init_rand_pos[ellipse_center_pos]
                    market.agents.append(
                        self.create_eer_agent(agent_id, self.config.init_cash, data_point, rand_pos=rand_pos,
                                              build_weights=build_weights, rng=rng))
                else:
                    market.agents.append(
                        self.create_eer_agent(agent_id, self.config.init_cash, data_point, rand_pos=None,
                                              build_weights=build_weights, rng=rng))

        return market

    def build_markets(self, rng: np.random._generator.Generator):
        """
            :brief Build all the markets and load the agents (of a particular type) with data in multi-threaded manner
            :param rng: numpy random generator for reproducible code
        """
        pool = ThreadPool(self.config.number_of_threads)
        market_ids = range(len(self.feature_vectors))
        seeds = rng.bit_generator._seed_seq.spawn(len(market_ids))
        rngs = [default_rng(child_seed) for child_seed in seeds]
        ground_truth = self.ground_truth
        inputs = zip(market_ids, rngs, ground_truth)
        self.markets = pool.starmap(self.build_market, inputs)
        pool.close()
        pool.join()

        return

    def get_events_per_market(self, market_id: int, rng: np.random._generator.Generator):
        """
            :brief If an agent is participating in a certain market-round, then update the event-queue with that event
            :param market_id: denotes the market-id or a specific row (or entry) in the input features list
        """
        for agent_id in range(len(self.markets[market_id].agents)):
            # add to the queue if the current_participation_round == the current_global_time
            if self.markets[market_id].agents[agent_id].next_participation_round == self.global_clock:
                self.event_queue.append((market_id, agent_id))
                # update the next participation round of the agent
                self.markets[market_id].agents[agent_id].determine_next_participation(current_time=self.global_clock, config=self.config, rng=rng)
        return

    def update_event_queue(self, rng: np.random._generator.Generator):
        """
            :brief For a particular time-step in the global-clock, this function loads all the events to the
            Event Queue (in an Asynchronous fashion across all the markets).
            Event Queue holds details regarding which agent(s) participates in a certain market for a given time-step.
        """
        # for every market, get all the agents participating in this time step
        pool = ThreadPool(self.config.number_of_threads)
        market_ids = range(len(self.markets))
        
        seeds = rng.bit_generator._seed_seq.spawn(len(market_ids))
        rngs = [default_rng(child_seed) for child_seed in seeds]
        inputs = zip(market_ids, rngs)
        
        pool.starmap(self.get_events_per_market, inputs)
        
        pool.close()
        pool.join()

        return

    def run_markets(self, rng):
        """
            :brief Runs the markets asynchronously based on the event-queue
        """
        for market_id, agent_id in self.event_queue:
            self.markets[market_id].run_market(agent_id, self.global_clock, rng)
        return

    def get_all_top_agents(self) -> np.array:
        """
            :brief Get top performing-agent "counts" from every market
            :TODO Running this using threads might speed up the process
        """
        all_top_agents = []
        for market in self.markets:
            all_top_agents = all_top_agents + list(market.top_agents)

        return np.unique(np.array(all_top_agents).flatten(), return_counts=True)

    def crossover(self, agent1, agent2, rng):
        """
            :brief Change strategies by crossing over with another agent.
                   Note that we are not swapping/changing any agent's centers ;
                        We are only swapping their radii "wx" ; weight on market-price "wp", and
                                                                                    scaling-factor "b" at random

            :param agent1: the first agent
            :param agent2: the second agent
        """
        r = rng.uniform()

        if r < self.config.value_r_lower_range:
            pos = rng.integers(0, len(agent1.wx) / 2)
            temp = agent1.wx[2 * pos]
            agent1.wx[2 * pos] = agent2.wx[2 * pos]
            agent2.wx[2 * pos] = temp
        elif self.config.value_r_lower_range <= r < self.config.value_r_higher_range:
            temp = agent1.wp
            agent1.wp = agent2.wp
            agent2.wp = temp
        else:
            temp = agent1.b
            agent1.b = agent2.b
            agent2.b = temp

        return agent1, agent2

    def update_agent_weights(self, sigma: bool, rng: np.random._generator.Generator):
        """
            :brief After every generation, update the agent pool by retaining only the best-performing agents
                   in every market and generation of new agents through mutation and cross-over

                Latest Agent Optimization Technique implemented on - July 5, 2020

            :TODO This function has a lot of scope for optimizations (with respect to logic, research and implementation)
                  One possible logic based improvement would be prune the total number of agents in the system.
                  Currently, even with every update on the agent-pool, we are retaining the total number of agents
                  in the system (it is kept as a constant).

            :param sigma: parameter to control weights (or radii) update's range.
                          "sigma" is estimated based on the total-batch loss.
        """

        sorted_top_agent_ids, top_agent_votes = self.get_all_top_agents()

        # build new-agent pool based on sorted_top_agent_ids
        # pick top "self.config.retain_top_k_agents_per_market" agents for every agent centered at a certain data-point.
        # say, "self.config.retain_top_k_agents_per_market" = 3 and "self.config.no_of_agents_per_market" = 5
        # Replicate 3 agents for every center h_i and for remaining 2, mutate + cross_over
        new_agent_pool = []
        bucket_id = 0
        temp_agents = []

        for agent_id, retain_vote in zip(sorted_top_agent_ids, top_agent_votes):
            if agent_id in list(range(bucket_id * self.config.no_of_agents_per_market,
                                      (
                                              bucket_id * self.config.no_of_agents_per_market) + self.config.no_of_agents_per_market)):
                temp_agents.append(np.array([agent_id, retain_vote]))
            else:
                while (bucket_id < len(self.feature_vectors)) and \
                        (agent_id not in list(range(bucket_id * self.config.no_of_agents_per_market,
                                                    (bucket_id * self.config.no_of_agents_per_market) +
                                                    self.config.no_of_agents_per_market))):
                    # if at all you don't find any agents centered for this h_i, then append the existing agents.
                    # ideally, the code should not come here.
                    if len(temp_agents) == 0:
                        print("NO REPLICATING AGENTS IN MARKET -", bucket_id)
                        start_id = bucket_id * self.config.no_of_agents_per_market
                        i = start_id
                        while i < start_id + self.config.retain_top_k_agents_per_market:
                            # add first 3 agent_ids and their retain_vote as 1 (more like equal probabilities to avoid divide by 0)
                            temp_agents.append(np.array([i, 1]))
                            i += 1

                    # create a new_agent_pool for h_i
                    if len(temp_agents):
                        temp_agents = np.array(temp_agents)
                        best_agents = temp_agents[
                            temp_agents[:, 1].argsort()[::-1][:self.config.retain_top_k_agents_per_market]].astype(
                            'int32')
                        id = 0
                        for id in list(best_agents[:, 0].astype('int32')):
                            new_agent_pool.append(self.agent_weights[id])

                        # mutate & cross-over to get evolved agents centered at h_i
                        # add them to new_agent_pool
                        i = 0
                        new_weights = []
                        new_agent1 = None
                        new_agent2 = None
                        # mutate & cross-over to get remaining n-k agents for that center h_i
                        while i < self.config.no_of_agents_per_market - best_agents.shape[0]:
                            # weighted random choice based on retain_vote_count
                            if best_agents.shape[0] > 1:
                                agent_pos1, agent_pos2 = rng.choice(best_agents[:, 0], 2, False,
                                                                          best_agents[:, 1] / best_agents[:,
                                                                                              1].sum()).astype('int32')
                            else:
                                agent_pos1, agent_pos2 = best_agents[:, 0][0], best_agents[:, 0][0]

                            if self.config.agent_type == "EllipticalExponentiallyRecurring":
                                new_agent1 = EllipticalExponentiallyRecurringAgent(config=self.config, rng=rng)
                                new_agent2 = EllipticalExponentiallyRecurringAgent(config=self.config, rng=rng)

                                new_agent1.min_radius, new_agent1.max_radius = self.markets[0].agents[
                                                                                   agent_pos1].min_radius, \
                                                                               self.markets[0].agents[
                                                                                   agent_pos1].max_radius

                                new_agent2.min_radius, new_agent2.max_radius = self.markets[0].agents[
                                                                                   agent_pos2].min_radius, \
                                                                               self.markets[0].agents[
                                                                                   agent_pos2].max_radius

                            new_agent1.wx = self.agent_weights[agent_pos1][:-3]
                            new_agent1.wp = self.agent_weights[agent_pos1][-3]
                            new_agent1.b = self.agent_weights[agent_pos1][-2]

                            new_agent2.wx = self.agent_weights[agent_pos2][:-3]
                            new_agent2.wp = self.agent_weights[agent_pos2][-3]
                            new_agent2.b = self.agent_weights[agent_pos2][-2]

                            new_agent1, new_agent2 = self.crossover(new_agent1, new_agent2, rng=rng)
                            new_agent1.mutate(sigma=sigma, rng=rng)
                            new_agent2.mutate(sigma=sigma, rng=rng)

                            temp_weight = new_agent1.wx
                            temp_weight = np.append(temp_weight, new_agent1.wp)
                            temp_weight = np.append(temp_weight, new_agent1.b)
                            temp_weight = np.append(temp_weight, self.agent_weights[agent_pos1][-1])

                            new_weights.append(temp_weight)

                            temp_weight = new_agent2.wx
                            temp_weight = np.append(temp_weight, new_agent2.wp)
                            temp_weight = np.append(temp_weight, new_agent2.b)
                            temp_weight = np.append(temp_weight, self.agent_weights[agent_pos2][-1])

                            new_weights.append(temp_weight)
                            i += 2

                        for i in range(self.config.no_of_agents_per_market - best_agents.shape[0]):
                            new_agent_pool.append(new_weights[i])

                        del new_weights
                        del new_agent1
                        del new_agent2

                    temp_agents = []
                    bucket_id += 1

                temp_agents = [np.array([agent_id, retain_vote])]

        # TODO: CODE REWRITTEN. SHOULD USE A FUNCTION But ----> Might be slower due to parameters
        while bucket_id < len(self.feature_vectors):
            if len(temp_agents) == 0:
                print("NO REPLICATING AGENTS IN MARKET -", bucket_id)
                start_id = bucket_id * self.config.no_of_agents_per_market
                i = start_id
                while i < start_id + self.config.retain_top_k_agents_per_market:
                    # add first 3 agent_ids and their retain_vote as 1
                    temp_agents.append(np.array([i, 1]))
                    i += 1

            # create a new_agent_pool for h_i
            if len(temp_agents):
                temp_agents = np.array(temp_agents)
                best_agents = temp_agents[
                    temp_agents[:, 1].argsort()[::-1][:self.config.retain_top_k_agents_per_market]].astype('int32')

                for id in list(best_agents[:, 0].astype('int32')):
                    new_agent_pool.append(self.agent_weights[id])

                # mutate & cross-over to get evolved agents centered at h_i
                # add them to new_agent_pool
                i = 0
                new_weights = []
                new_agent1 = None
                new_agent2 = None
                # mutate & cross-over to get remaining n-k agents for that center h_i
                while i < self.config.no_of_agents_per_market - best_agents.shape[0]:
                    # weighted random choice based on retain_vote_count
                    if best_agents.shape[0] > 1:
                        agent_pos1, agent_pos2 = rng.choice(best_agents[:, 0], 2, False,
                                                                  best_agents[:, 1] / best_agents[:, 1].sum()).astype(
                            'int32')
                    else:
                        agent_pos1, agent_pos2 = best_agents[:, 0][0], best_agents[:, 0][0]

                    if self.config.agent_type == "EllipticalExponentiallyRecurring":
                        new_agent1 = EllipticalExponentiallyRecurringAgent(config=self.config, rng=rng)
                        new_agent2 = EllipticalExponentiallyRecurringAgent(config=self.config, rng=rng)

                        new_agent1.min_radius, new_agent1.max_radius = self.markets[0].agents[
                                                                           agent_pos1].min_radius, \
                                                                       self.markets[0].agents[
                                                                           agent_pos1].max_radius

                        new_agent2.min_radius, new_agent2.max_radius = self.markets[0].agents[
                                                                           agent_pos2].min_radius, \
                                                                       self.markets[0].agents[
                                                                           agent_pos2].max_radius

                    new_agent1.wx = self.agent_weights[agent_pos1][:-3]
                    new_agent1.wp = self.agent_weights[agent_pos1][-3]
                    new_agent1.b = self.agent_weights[agent_pos1][-2]

                    new_agent2.wx = self.agent_weights[agent_pos2][:-3]
                    new_agent2.wp = self.agent_weights[agent_pos2][-3]
                    new_agent2.b = self.agent_weights[agent_pos2][-2]

                    new_agent1, new_agent2 = self.crossover(new_agent1, new_agent2, rng=rng)
                    new_agent1.mutate(sigma=sigma, rng=rng)
                    new_agent2.mutate(sigma=sigma, rng=rng)

                    temp_weight = new_agent1.wx
                    temp_weight = np.append(temp_weight, new_agent1.wp)
                    temp_weight = np.append(temp_weight, new_agent1.b)
                    temp_weight = np.append(temp_weight, self.agent_weights[agent_pos1][-1])

                    new_weights.append(temp_weight)

                    temp_weight = new_agent2.wx
                    temp_weight = np.append(temp_weight, new_agent2.wp)
                    temp_weight = np.append(temp_weight, new_agent2.b)
                    temp_weight = np.append(temp_weight, self.agent_weights[agent_pos2][-1])

                    new_weights.append(temp_weight)
                    i += 2

                for i in range(self.config.no_of_agents_per_market - best_agents.shape[0]):
                    new_agent_pool.append(new_weights[i])

                del new_weights
                del new_agent1
                del new_agent2

            bucket_id += 1

        if len(new_agent_pool) == len(self.agent_weights):
            self.agent_weights = new_agent_pool
        else:
            "WEIGHT UPDATE LOGIC MISTAKE!!!!!!!!!!!"
            exit()

        return

    def run_on_batch(self, num_agents: int):
        """
            :brief Train the agents in the market (in a batch-wise manner) using a genetic algorithm
            :param num_agents: the total number of agents
        """
        if len(self.feature_vectors) == 0:
            return -1  # No Data Error

        if len(self.feature_vectors) != len(self.ground_truth):
            return -2  # Number of Labels and Raw Data are not consistent

        if len(self.agent_weights) == 0:
            print("Building Agent weights")
            self.build_agent_weights(num_agents)

        print("Building Markets")
        self.build_markets(num_agents)

        print("Running Markets ...")

        start = time.time()
        while self.global_clock < self.config.market_duration:
            self.update_event_queue()
            self.run_markets(0)
            self.event_queue = []
            self.global_clock += 1

        end = time.time()
        print("Time Taken: ", end - start)

        current_obj = self.compute_objective()
        # print("Objective: ", current_obj)

        self.batch_loss = current_obj

        # Update Sigma
        sigma = 2 * math.sqrt(current_obj)

        # Other method to update SIGMA:
        # alpha = np.log(0.01) / num_generations
        # tau = 1 / math.sqrt(len(self.feature_vectors[0]))
        # sigma = tau * math.exp(- alpha * generation_count)

        print("Sigma: ", sigma)
        # GET ALL THE TOP AGENTS
        for market_id in range(len(self.markets)):
            boolean_ground_truth = self.ground_truth[market_id] >= 0.5
            self.markets[market_id].compute_top_agents(boolean_ground_truth, len(self.feature_vectors))

        # Update Agent Weights using Latest Agent Optimization Technique - July 5th 2020
        self.update_agent_weights(sigma)
        self.markets = []
        self.global_clock = 0
        self.feature_vectors = []
        self.ground_truth = []
        self.init_rand_pos = []

        return

    def run_on_batch3(self, rng: np.random._generator.Generator, num_agents: int, settings=None, generation=None):
        """
            :brief Train the agents in the market (in a batch-wise manner) using a genetic algorithm
            :param num_agents: the total number of agents
        """
        if len(self.feature_vectors) == 0:
            return -1  # No Data Error

        if len(self.feature_vectors) != len(self.ground_truth):
            return -2  # Number of Labels and Raw Data are not consistent

        if len(self.agent_weights) == 0:
            print("Building Agent weights")
            self.build_agent_weights(num_agents=num_agents, rng=rng)

        print("Building Markets")
        self.build_markets(rng=rng)

        print("Running Markets ...")

        start = time.time()
        while self.global_clock < self.config.market_duration:
            self.update_event_queue(rng=rng)
            self.run_markets(rng=rng)
            self.event_queue = []
            self.global_clock += 1

        end = time.time()
        print("Time Taken: ", end - start)

        current_obj = self.compute_objective()
        # print("Objective: ", current_obj)

        self.batch_loss = current_obj

        # Update Sigma
        sigma = 2 * math.sqrt(current_obj)

        # Other method to update SIGMA:
        # alpha = np.log(0.01) / num_generations
        # tau = 1 / math.sqrt(len(self.feature_vectors[0]))
        # sigma = tau * math.exp(- alpha * generation_count)

        print("Sigma: ", sigma)
        # GET ALL THE TOP AGENTS
        for market_id in range(len(self.markets)):
            boolean_ground_truth = self.ground_truth[market_id] >= 0.5
            if settings == None:
                self.markets[market_id].compute_top_agents3(boolean_ground_truth, 12)
            else:
                self.markets[market_id].compute_top_agents3(boolean_ground_truth, settings.top_agents_n)

        # Update Agent Weights using Latest Agent Optimization Technique - July 5th 2020
        self.update_agent_weights(sigma=sigma, rng=rng)
        self.markets = []
        self.global_clock = 0
        self.feature_vectors = []
        self.ground_truth = []
        self.init_rand_pos = []

        return
