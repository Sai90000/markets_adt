import numpy as np
from scipy.special import expit
from .Agent import Agent
import math

class EllipticalExponentiallyRecurringAgent(Agent):
    def __init__(self, config, rng, buy_positive=False):
        super().__init__(config, rng)
        self.buy_positive = buy_positive
        self.min_radius = 0
        self.max_radius = 0


    def compute_estimate(self, config) -> float:
        """
        Compute the estimate of the true price of the asset using the ellipsoid
        :param config: config object created from the .ini file
        :return: the agent's estimate of the value of the share
        """
        if len(self.wx) != 2*len(self.x):
            print("Error weights must have dimension two times feature dimension")
            return -1

        summation = 0
        for i in range(len(self.x)):
            summation += (1.0/(self.wx[2*i] ** 2)) * pow(self.x[i] - self.wx[2*i+1], 2)

        summation = 1 - summation
        # the following term b denotes the random scale-factor alpha
        summation *= self.b
        summation += self.wp * (self.p - config.init_price)
        self.estimate = expit(summation)
        return self.estimate

    #
    def mutate(self, sigma: float, rng: np.random._generator.Generator) -> None:
        """
        Use a random normal to mutate the strategy. de
        :param sigma: termines how far the strategy can wiggle.
        :return: None
        """
        for i in range(len(self.wx)):
            if i % 2 == 1:
                self.wx[i] = self.wx[i]

            else:
                self.wx[i] += rng.uniform(self.min_radius - self.wx[i], self.max_radius - self.wx[i]) * sigma
        self.wp += rng.normal(0, 1) * sigma
        l = -self.b + 0.01
        h = self.b
        if h < l:
            delta = rng.uniform(h, l) * sigma
        else:
            delta = rng.uniform(l, h) * sigma
        self.b += delta # rng.uniform(-self.b + 0.01, self.b) * sigma

    def init_position(self, random_data_point: list):
        """
        Randomly initialize the weights for position of the agent.
        :param random_data_point - a data point generated at random
        :return None
        """
        for i in range(len(random_data_point)):
            self.wx[2*i+1] = random_data_point[i]

    def determine_next_participation(self, current_time, config, rng: np.random._generator.Generator):
        """
        Calculate the next time step the agent will have the opportunity ot bid by
        random sampling of an exponential distribution
        :param current_time: time step that just occurred with this agent getting an opportunity
        :param config: config object created from config.ini file
        :return: None
        """
        self.next_participation_round = current_time + math.ceil(rng.exponential(scale=(1/config.lambda_value)))
