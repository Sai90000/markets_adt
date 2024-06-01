import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import csv
from ..training import GeneticAlgorithm
import os
import json
from ..agents.AuxAgents import AuxAgent

class BaseMarketModel:

    def __init__(self, config, train=True) -> None:
        """
        Initialize the market for training or prediction and create the relevant directories to save output in
        :param config: config object created from config.ini file
        :param train: When true, will attempt to create a folder in config.market_model_location to save the
        model training. When false, will attempt to create a folder in config.output_folder_location to save predictions
        in. In either case if a folder already exists in that location it will throw an error.
        """
        self.config = config
        self.price_history = {}
        print(train)
        if train:
            try:
                print('*'*20)
                print(config.market_model_location)
                
                os.makedirs(config.market_model_location)
            except FileExistsError:
                # raise Exception(
                #     f'ERROR OUTPUT FOLDER {config.market_model_location} EXISTS PLEASE CHOOSE DIFFERENT FOLDER TO AVOID'
                #     f' OVERWRITING EXISTING MARKET')
                pass
            self.config.save_config(f'{self.config.market_model_location}config.ini')
        else:
            try:
                os.makedirs(config.output_folder_location)
            except FileExistsError:
                raise Exception(
                    f'ERROR OUTPUT FOLDER {config.output_folder_location} EXISTS PLEASE CHOOSE DIFFERENT FOLDER TO '
                    f'AVOID OVERWRITING EXISTING OUTPUT')

    def fit(self, rng: np.random._generator.Generator, X: np.ndarray, y: np.array, start_config, cv_split_folder: str = None, logger=None):
        """

        :brief: fit a market to the data - run the evolutionary algorithm to get agent weights for the
        market and save them.
        :param X: training data
        :param y: training labels
        :param start_config: min/max sizes for elliptical agents
        :param cv_split_folder: in case of cross-validation, give sub-folder to save results into
        :param logger: to log output
        :param rng: random number generator
        """
        writer = pd.ExcelWriter(f'{self.config.market_model_location}training.xlsx')
        ga = GeneticAlgorithm(config=self.config)
        train_count = X.shape[0]
        losses_per_instance = []
        # train the market for "number_generations" generations as specified in "config.ini" file
        for generation in range(self.config.number_generations):
            gen_start_time = time.time()
            if logger is not None:
                logger.info(f'Beginning Generation {generation}')
            training_loss = 0
            temp_weights_list = [0] * X.shape[0] * self.config.no_of_agents_per_market
            if generation == self.config.number_generations - 1:
                gen_file = self.config.market_model_location + 'FINAL_AGENT_WEIGHTS.csv'
                os.mkdir(self.config.market_model_location + 'previous_generation_weights')
            else:
                if cv_split_folder is None:
                    gen_file = self.config.market_model_location + "final_agent_weights_" + str(generation) + ".csv"

                else:
                    gen_file = f'{cv_split_folder}final_agent_weights_{str(generation)}.csv'

            print("--------------Running Generation", generation, "----------------------------")
            weights = None

            # shuffle the entire dataset
            shuffle_indicies = np.arange(train_count)
            rng.shuffle(shuffle_indicies)
            n_X, n_y, n_start_config = X[shuffle_indicies], y[shuffle_indicies], start_config[shuffle_indicies]
            
            if generation > 0:
                # get latest weights (from previous generation) ;
                weight_indicies = []
                for index in shuffle_indicies:
                    for i in range(index * self.config.no_of_agents_per_market,
                                   (index * self.config.no_of_agents_per_market) + self.config.no_of_agents_per_market):
                        weight_indicies.append(i)

                weight_indicies = np.array(weight_indicies)
                weights = pd.read_csv(self.config.market_model_location + "final_agent_weights_" + str(
                    generation - 1) + ".csv", header=None).values[weight_indicies]
            if generation == self.config.number_generations - 1:
                for file in os.listdir(self.config.market_model_location):
                    if 'agent_weights' in file:
                        os.replace(self.config.market_model_location + file,
                                   self.config.market_model_location + 'previous_generation_weights/' + file)

            for batch in tqdm(range(0, train_count, self.config.batch_size)):
                batch_start_time = time.time()
                if logger is not None:
                    logger.info(f'Beginning Batch {batch}')
                new_batch = range(batch, min(train_count, batch + self.config.batch_size))

                self.config.number_of_claims = len(new_batch)
                self.config.number_of_agents = self.config.no_of_agents_per_market * self.config.number_of_claims

                # shuffle data if needed:
                new_X = list(n_X[new_batch])
                new_y = list(n_y[new_batch])
                new_start_config = list(n_start_config[new_batch])

                if self.config.agent_type == "EllipticalExponentiallyRecurring":
                    # get random_positions and its corresponding labels
                    rand_pos = list(range(0, len(new_X)))
                    rand_pos_labels = [new_y[i] >= 0.5 for i in rand_pos]
                    init_rand_pos_arg = list(tuple(zip(rand_pos, rand_pos_labels, new_start_config)))
                    ga.init_rand_pos = init_rand_pos_arg

                ga.feature_vectors = new_X
                ga.ground_truth = new_y
                if weights is not None:
                    # Next 4 lines are not needed... can write this using a mathematical formula
                    w_range = range(batch * self.config.no_of_agents_per_market,
                                    (batch * self.config.no_of_agents_per_market) +
                                    (len(new_batch) * self.config.no_of_agents_per_market))

                    ga.agent_weights = list(weights[w_range])

                ga.run_on_batch3(num_agents=self.config.number_of_agents, settings=self.config, rng=rng, generation=generation)
                training_loss += ga.batch_loss
                print("Batch Loss:", ga.batch_loss)
                # Always write the weights in-order so that they can be processed directly
                t_j = 0
                for market_id in shuffle_indicies[new_batch]:
                    for i in range(market_id * self.config.no_of_agents_per_market,
                                   (market_id * self.config.no_of_agents_per_market)
                                   + self.config.no_of_agents_per_market):
                        temp_weights_list[i] = ga.agent_weights[t_j]
                        t_j += 1

                ga.agent_weights = []
                ga.batch_acc = 0
                ga.batch_loss = 0
                ga.correct_count = 0
                if logger is not None:
                    logger.info(f'Batch took {time.time() - batch_start_time} seconds')
            print('Training Loss per instance after Generation %d: %f %%' % (generation, training_loss / train_count))
            losses_per_instance.append(training_loss / train_count)
            if logger is not None:
                logger.info(f'Generation took {time.time() - gen_start_time} seconds')
            # dump weights for this generation
            with open(gen_file, 'a', newline='') as f:
                a = csv.writer(f)
                a.writerows(temp_weights_list)
            f.close()

        pd.DataFrame(losses_per_instance).to_excel(writer, sheet_name='Losses per Instance')
        writer.close()
        return

    #TODO: haven't tested interpret functions in a while
    def level_one_interpret(self, ga, market_id):
        """
            :brief A high-level method to translate the final market score into whether the test data point is reproducible
                    or not.. based on a simple thresholding method.

            :param ga: Object belonging GeneticAlgorithm class.
            :param market_id: Market ID (under consideration)

            :return Dictionary comprising of "Level One" aspect of Market Output Interpretation.
        """
        result = {}
        decision = ""
        score = round(ga.markets[market_id].compute_price(), 5)

        if score < 0.25:
            decision = "is likely not reproducible"
        elif 0.25 <= score < 0.45:
            decision = "probably not reproducible"
        elif 0.45 <= score < 0.55:
            decision = "cannot be determined to be clearly reproducible or not"
        elif 0.55 <= score < 0.75:
            decision = "probably is reproducible"
        elif score >= 0.75:
            decision = "is likely reproducible"

        int_str_one = "The Market provided a score of " + str(score) + \
                      " suggesting that the claim - " + str(decision) + "."

        result['Market Price'] = int_str_one
        return result

    def level_two_three_five_interpret(self, ga, market_id):
        """
            :brief *) At level two, we handle high-level information associated with the market & agents output ;
                        Information such as: Total Agent Participation, Agent Participation in each class, and Outlier Handle
                *) At level three, we dive deeper into each agent to determine its total money spent on a specific asset ;
                        In a way, it determines the agent's confidence in the given test data point.
                *) At level five, we splice the ellipsoidal agent and deduce the range in which a particular component in
                the test data point falls under. Essentially, we get something like, agent 5 purchased because :-
                    'Feature "author-count" lies between 0 and 10'

            :TODO This function can be further split into levels two, three and five for modularity.

            :param ga: Object belonging GeneticAlgorithm class.
            :param market_id: Market ID (under consideration)

            :return Tuple of dictionaries comprising of "level_two", "level_three", and "level_five" aspects of
                    market interpretations respectively.
        """
        level_two = {}
        level_three = {'Agent Details': []}
        level_five = {}
        participating_agents_count = 0
        positive_agent_count = 0
        negative_agent_count = 0
        both_agent_count = 0
        most_similar_paper_ids = set()

        for agent in ga.markets[market_id].agents:

            agent_positive_shares = len(agent.positive_asset_prices)
            agent_negative_shares = len(agent.negative_asset_prices)

            if agent_positive_shares > 0 and agent_negative_shares > 0:
                both_agent_count += 1
            elif agent_positive_shares > 0:
                positive_agent_count += 1
            elif agent_negative_shares > 0:
                negative_agent_count += 1
            # check for agent-participation in the Market
            if agent_positive_shares > 0 or agent_negative_shares > 0:
                participating_agents_count += 1
                investment = self.config.init_cash - agent.cash
                most_similar_paper_ids.add(int(np.floor(agent.id / 5)))

                level_three_str1 = "Agent " + str(agent.id) + " purchased " + str(agent_positive_shares) + \
                                   " reproducible shares and " + str(
                    agent_negative_shares) + " non-reproducible shares. "
                percentage_investment = round(100 * investment / self.config.init_cash, 3)
                if investment >= 0.66 * self.config.init_cash:
                    confidence_level = 'It has High Confidence Level because of investing ' + \
                                       str(percentage_investment) + '% of its initial cash.'
                elif investment < 0.33 * self.config.init_cash:
                    confidence_level = 'It has Low Confidence Level because of investing only ' + \
                                       str(percentage_investment) + '% of its initial cash.'
                else:
                    confidence_level = 'It has Medium Confidence Level because of investing ' + \
                                       str(percentage_investment) + '% of its initial cash.'

                level_three_str = level_three_str1 + confidence_level
                level_three['Agent Details'].append(level_three_str)

                # level-five
                level_five["Agent " + str(agent.id)] = self.level_five_interpret(agent, ga.feature_vectors[market_id])

        # if no participating agents ; treat it as an outlier and find the most-closest training data-point
        if not participating_agents_count:
            inp_feat = np.array(ga.agent_weights)
            train_feat = inp_feat[0::self.config.no_of_agents_per_market, 1:-3:2]
            test_feat = np.array(ga.feature_vectors[market_id])
            dist_2 = np.sqrt(np.sum((train_feat - test_feat) ** 2, axis=1))
            nearest_distance = np.min(dist_2[np.nonzero(dist_2)])
            # nearest training data point features ; use if needed to display in the front-end ; not used for now;
            nearest_training_point_id = np.where(dist_2 == nearest_distance)[0][0]
            first_near_agent = nearest_training_point_id * self.config.no_of_agents_per_market
            last_near_agent = (
                                      nearest_training_point_id * self.config.no_of_agents_per_market) + self.config.no_of_agents_per_market
            nearby_agents = inp_feat[first_near_agent: last_near_agent]
            near_agent_weights = nearby_agents[:, 0:-3:2]
            # pick the nearest agent with max component radius
            ag_id = np.where(np.abs(near_agent_weights) == np.max(np.abs(near_agent_weights)))[0][0]
            ag_radii = np.abs(near_agent_weights[ag_id])
            ag_radii[::-1].sort()
            # compute the distance/radius from the radius
            distance_from_agent = 0
            for max_component in ag_radii:
                if nearest_distance / max_component < 1:
                    continue
                else:
                    distance_from_agent = nearest_distance / max_component
            # the radius of the nearest agent is approximately
            # distance_from_agent times away making it fall outside it's sphere of influence.
            interpret_outlier = ""
            if distance_from_agent and distance_from_agent < 500:
                interpret_outlier = "The sphere of influence of the nearest agent is approximately " + \
                                    str(distance_from_agent) + " times away."
            elif distance_from_agent and distance_from_agent > 500:
                interpret_outlier = "The sphere of influence of the nearest agent is more than 500 times away!"
            # if all the component radii are more than the nearest-distance ; then do nothing
            # if distance_from_agent == 0:
            #     interpret_outlier = ""
            level_two['Outlier Handle'] = interpret_outlier

        level_two['Total Agents'] = str(participating_agents_count) + " agents participated in this Market."
        # if positive_agent_count
        level_two['Total Positive Class Agents'] = str(
            positive_agent_count) + " agents purchased 'will reproduce' assets."
        level_two['Total Negative Class Agents'] = str(
            negative_agent_count) + " agents purchased 'will not reproduce' assets."
        level_two['Total Both Class Agents'] = str(both_agent_count) + ' agents purchased both types of assets.'
        level_two['Most Similar Paper Ids'] = list(most_similar_paper_ids)
        return level_two, level_three, level_five

    def level_four_interpret(self, ga, paper_ids):
        """
            :brief At level 4, we deduce the features corresponding to the nearest training point for a given test
            data point.

            :param ga: Object belonging to class GeneticAlgorithm.
            :param paper_ids: nearest data point "ids" corresponding to given test data point.

            :return A dictionary comprising of level-five aspect of the market interpretation.
        """
        level_four = {}
        for id in paper_ids:
            # pick all the agent weights by removing alpha, wp and buy positive/negative boolean variable
            weights_center = ga.agent_weights[self.config.no_of_agents_per_market * id][:-3]
            center = weights_center[1::2]
            nearest_data_point = {}
            for feature_id in range(len(self.config.all_features_list)):
                feat_name = self.config.all_features_list[feature_id]
                nearest_data_point[feat_name] = center[feature_id]

            key = 'Paper ID ' + str(id)
            level_four[key] = {}
            level_four[key]['Reason'] = "Agents with 'Agent ID' in range - (" + str(
                self.config.no_of_agents_per_market * id) \
                                        + "," + str(
                (self.config.no_of_agents_per_market * id) + self.config.no_of_agents_per_market) \
                                        + ") were closer to the training datapoint #" + str(id) + "."
            level_four[key]['Features'] = nearest_data_point

        return level_four

    def level_five_interpret(self, agent, X):
        """
            :brief At Level 5, we splice the ellipsoidal agent and deduce the range in which a particular component in the
            test data point falls under. Essentially, we get something like, agent 5 purchased because :-
                'Feature "author-count" lies between 0 and 10'
            in the range of (0, 10) ;

            :param agent: agent object
            :param X: input features
            :type X: numpy.ndarray

            :return A dictionary comprising of level-five aspect of the market interpretation.
        """
        X = list(X)
        level_five = []
        feat_weight = agent.wx[0::2]
        feat_center = agent.wx[1::2]
        for i in range(len(self.config.all_features_list)):
            feat_name = self.config.all_features_list[i]
            # ignore index i and compute remainder sum - Q
            h_vec = np.array(feat_center[:i] + feat_center[i + 1:])
            x_vec = np.array(X[:i] + X[i + 1:])
            r_vec = np.array(feat_weight[:i] + feat_weight[i + 1:])

            # Q = (x - h) ** 2 / r ** 2
            temp = feat_weight[i] * np.sqrt(1 - np.sum(np.power(x_vec - h_vec, 2) / np.power(r_vec, 2)))
            upper_bound = feat_center[i] + temp
            lower_bound = max(feat_center[i] - temp, 0)

            level_five_str = "Feature " + str(feat_name) + " lies between " + str(lower_bound) + " and " + str(
                upper_bound)
            level_five.append(level_five_str)

        return level_five

    def predict(self, X: np.ndarray, agent_weight_location: str, rng: np.random._generator.Generator,
                interpret_results: bool = False, logger=None, test_labels: list = []) -> (list, dict, dict):
        """
            :brief: Tests the (market) algorithm (in a batch-wise manner) and returns the output market predictions.
            Corresponding input weights are to be specified from the "config.py" file through the parameter -
            "agent_weights_file_location"

            :param X: input or test features
            :param agent_weight_location: to load the market from
            :param interpret_results:


            :return: predictions for the input X, market interpretations, price-history
        """

        y_pred = []
        interpret = []
        ga = GeneticAlgorithm(self.config, test=True)
        ga.ground_truth = test_labels
        test_count = X.shape[0]
        no_participants = 0
        temp_i = 0
        ga.agent_weights = pd.read_csv(agent_weight_location + "FINAL_AGENT_WEIGHTS.csv", header=None)
        print(f'Weights size: {ga.agent_weights.shape}')
        ga.agent_weights = ga.agent_weights.values.tolist()
        self.config.number_of_agents = len(ga.agent_weights)

        print(f'X size: {X.shape}')

        # initialize price_history list
        for i in range(test_count):
            self.price_history["Market " + str(i)] = {0: self.config.init_price}

        batch_start_time = time.time()

        self.config.number_of_claims = X.shape[0]

        ga.feature_vectors = X
        ga.build_markets(rng=rng)

        if self.config.percentage_aux_agents:
            agent_percentage =  self.config.percentage_aux_agents/100

            assert (self.config.expert + self.config.proficient + self.config.competent + self.config.beginner + self.config.novice) == 1 

            for market in ga.markets:
                N = len(market.agents)
                aux_agents_percentage = agent_percentage
                num_aux_agents =  int(aux_agents_percentage / (1 - aux_agents_percentage) * N)

                print(f"Adding {num_aux_agents} agents")

                aux_agent_data = {}
                aux_agent_data['expert'] = int(num_aux_agents*self.config.expert)
                aux_agent_data['proficient'] = int(num_aux_agents*self.config.proficient)
                aux_agent_data['competent'] = int(num_aux_agents*self.config.competent)
                aux_agent_data['beginner'] = int(num_aux_agents*self.config.beginner)
                aux_agent_data['novice'] = int(num_aux_agents*self.config.novice)

                cum_sum = N + 1
                for aa, agent_nums in aux_agent_data.items():
                    for agent_id in range(cum_sum, cum_sum + agent_nums + 1):
                        if aa == 'expert':
                            agent = AuxAgent(config=self.config, 
                                             rng=rng, 
                                             correctness=round(rng.uniform(low = 0.8, high = 1.0), 4),
                                             label=market.gt
                                             )
                    
                        elif aa == 'proficient':
                            agent = AuxAgent(config=self.config, 
                                             rng=rng, 
                                             correctness=round(rng.uniform(low = 0.6, high = 0.8), 4),
                                             label=market.gt
                                             )
                    
                        elif aa == 'competent':
                            agent = AuxAgent(config=self.config, 
                                             rng=rng, 
                                             correctness=round(rng.uniform(low = 0.4, high = 0.6), 4),
                                             label=market.gt
                                             )

                        elif aa == 'beginner':
                            agent = AuxAgent(config=self.config, 
                                             rng=rng, 
                                             correctness=round(rng.uniform(low = 0.2, high = 0.4), 4),
                                             label=market.gt
                                             )
                    
                        elif aa == 'novice':
                            agent = AuxAgent(config=self.config, 
                                             rng=rng, 
                                             correctness=round(rng.uniform(low = 0.0, high = 0.2), 4),
                                             label=market.gt
                                             )
                        
                        agent.id = agent_id
                        agent.name = f"Agent_{aa}_" + str(agent_id)
                        agent.p = self.config.init_price
                        agent.cash = self.config.init_cash
                        market.agents.append(agent)
                    
                    cum_sum = cum_sum + agent_nums
                 
        start = time.time()
        while ga.global_clock < self.config.market_duration:
            ga.update_event_queue(rng=rng)
            ga.run_markets(rng=rng)
            ga.event_queue = []
            if ga.global_clock % 10 == 0:
                print("Finished", ga.global_clock, "time durations.")

            ga.global_clock += 1

        end = time.time()
        print("Time Taken: ", end - start)

        for market_id in tqdm(range(len(ga.feature_vectors))):
            if ga.markets[market_id].current_price == self.config.init_price:
                no_participants += 1
            if interpret_results:
                market_interpret = {'Level One': self.level_one_interpret(ga, market_id)}
                market_interpret['Level Two'], market_interpret[
                    'Level Three'], level_five = self.level_two_three_five_interpret(
                    ga, market_id)
                paper_ids = market_interpret['Level Two']['Most Similar Paper Ids']
                market_interpret['Level Four'] = self.level_four_interpret(ga, paper_ids)
                market_interpret['Level Five'] = level_five

                interpret.append(market_interpret)

            temp_i += 1
            y_pred.append(ga.markets[market_id].current_price)
        for i in ga.markets:
            mkt = i.output_market()
            with open(f"{self.config.output_folder_location}market_{mkt['market_id']}_action_predict.json",
                      'w') as marketfile:
                mkt_srl = json.dumps(mkt, default=str)
                json.dump(mkt_srl, marketfile)
        ga.markets = []
        ga.global_clock = 0
        ga.feature_vectors = []

        print("Non-participants: ", no_participants)
        if logger is not None:
            logger.info(f'Batch took {time.time() - batch_start_time} seconds')
            logger.info(f'Non-participants: {no_participants}')
        print("Total Non-participants: ", no_participants)
        if logger is not None:
            logger.info(f'Total Non-participants: {no_participants}')
        return y_pred, interpret, self.price_history
