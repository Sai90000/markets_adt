import numpy as np
import math
from math import exp, log
from .EllipticalExponentiallyRecurringAgent import EllipticalExponentiallyRecurringAgent

class AuxAgent(EllipticalExponentiallyRecurringAgent):
    '''
    Simple wrapper class for the Elliptically exponential agents for 
    implementing auxillary agents that have correctness parameter.
    This parameter decides how frequent the auxillary agent gives us correct answers.
    These can be a crude proxy to humans who are probablistic.     
    '''
    def __init__(self, config, rng, correctness=0.5, buy_positive=False, label=None):
        super().__init__(config, rng, buy_positive)
        self.correctness = correctness
        self.y = label

    def would_buy_positive(self, 
                           trans_price: float,
                           rng: np.random._generator.Generator,
                           liquidity_constant,
                           **kwargs
                           ) -> bool:
        '''
        Parameters
        ----------
        rng: numpy random number generator object to be passed for experiment reproduciblity 
        '''


        p = [1 - self.correctness, self.correctness]
        choice = rng.choice([0,1], p=p)
        val, pos = self.calc_shares_by_price(trans_price, 1 / liquidity_constant)

        if pos:
            positive_shares = val
            negative_shares = 0
        else:
            positive_shares = 0
            negative_shares = val
        if self.cash > 0:
            c1 = (liquidity_constant * log(exp(positive_shares / liquidity_constant) + \
                                        exp(negative_shares / liquidity_constant)))
            n1 = (liquidity_constant * log(exp((self.cash + c1)/liquidity_constant) - \
                                            exp(negative_shares/liquidity_constant))) - positive_shares
            
            num_shares = math.floor(n1)
            self.y==1
        else:
            return False, 0, 0
        if choice == 1:
            return self.y==1, num_shares, self.cash
        else:
            
            return self.y==0, num_shares, self.cash
    
    def would_buy_negative(self, 
                           trans_price: float,
                           rng: np.random._generator.Generator,
                           liquidity_constant,
                           **kwargs
                           ) -> bool:
        '''
        Parameters
        ----------
        rng: numpy random number generator object to be passed for experiment reproduciblity 
        '''
        p = [1 - self.correctness, self.correctness]
        choice = rng.choice([0,1], p=p)
        
        val, pos = self.calc_shares_by_price(trans_price, 1 / liquidity_constant)
        
        if pos:
            positive_shares = val
            negative_shares = 0
        else:
            positive_shares = 0
            negative_shares = val

        if self.cash > 0:
            c1 = (liquidity_constant * log(exp(positive_shares / liquidity_constant) + \
                                        exp(negative_shares / liquidity_constant)))
            n1 = (liquidity_constant * log(exp((self.cash + c1)/liquidity_constant) - \
                                            exp(positive_shares/liquidity_constant))) - negative_shares
            val, pos =  self.calc_shares_by_price(1-self.estimate, 1 / liquidity_constant)

            num_shares = math.floor(n1)
        else:
            return False, 0, 0
        if choice == 1:
            return self.y==0, num_shares, self.cash
        else:
            return self.y==1, num_shares, self.cash
            
    def compute_estimate(self, config):
        # dummy function to mask the og compute
                
        self.buy_positive = 1
        self.estimate = 1

        return self.estimate