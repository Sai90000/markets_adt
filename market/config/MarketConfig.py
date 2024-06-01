import configparser

class MarketConfig():
    #TODO: should this init contain the defaults for the config
    def __init__(self):
        self.msg = "hi"

        # claim count - updates automatically based on data passed
        self.number_of_claims = 0

        # tweak crossover function in Agent.cpp
        self.value_r_lower_range = 0.333
        self.value_r_higher_range = 0.667
        self.sigma = 1.0
        self.percentage_aux_agents = 0
        # Seed
        self.random_seed = 120
        self.batch_size = 32

        # Parallelization
        self.number_of_threads = 4

        self.agent_type = "EllipticalExponentiallyRecurring"
        self.lambda_value = 0.4  # rate
        self.init_cash = 20
        self.init_price = 0.5


        # Market
        self.market_duration = 100
        self.liquidity_constant = 100
        self.percent = 0.02

        # Genetic Algorithm
        self.number_generations = 1
        self.no_of_agents_per_market = 5
        self.retain_top_k_agents_per_market = 3
        self.number_of_agents = self.no_of_agents_per_market * self.number_of_claims
        self.top_agents_n = 12

        # IO Details - placeholder strings provided through config.ini file
        self.output_folder_location = ""
        self.input_feature_file_location = ""

        self.input_feature_file_location = ""
        self.parent_dir = ""        


    def load_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)

        # agent
        self.value_r_lower_range = float(config['agent']['value_r_lower_range'])
        self.value_r_higher_range = float(config['agent']['value_r_higher_range'])
        self.sigma = float(config['agent']['sigma'])
        self.agent_type = config['agent']['agent_type']
        self.init_cash = float(config['agent']['init_cash'])
        self.lambda_value = float(config['agent']['lambda_value'])
        self.percentage_aux_agents = float(config['agent']['percentage_aux_agents'])

        # seed
        self.random_seed = int(config['seed']['random_seed'])
        self.batch_size = int(config['seed']['batch_size'])

        # parallel
        self.number_of_threads = int(config['parallel']['number_of_threads'])

        # market
        self.market_duration = int(config['market']['market_duration'])
        self.liquidity_constant = int(config['market']['liquidity_constant'])
        self.init_price = float(config['market']['init_price'])
        self.percent = float(config['market']['percent'])

        # ga
        self.number_generations = int(config['ga']['number_generations'])
        self.no_of_agents_per_market = int(config['ga']['no_of_agents_per_market'])
        self.top_agents_n = int(config['ga']['top_agents_n'])
        self.retain_top_k_agents_per_market = int(config['ga']['retain_top_k_agents_per_market'])
        self.number_of_agents = self.no_of_agents_per_market * self.number_of_claims

        # files
        self.parent_dir = config['files']['parent_dir']
        self.output_folder_location = self.parent_dir + config['files']['output_folder_location']
        self.training_feature_file_location = config['files']['training_feature_file_location']
        self.testing_feature_file_location = config['files']['testing_feature_file_location']
        self.intermediate_file_location = self.parent_dir + config['files']['intermediate_file_location']
        
        # aux
        self.expert = float(config['aux']['expert'])
        self.proficient = float(config['aux']['proficient'])
        self.competent = float(config['aux']['competent'])
        self.beginner = float(config['aux']['beginner'])
        self.novice = float(config['aux']['novice'])
        
        if 'market_model_location' in config['files'].keys():
            self.market_model_location = self.parent_dir + config['files']['market_model_location']
        print('CONFIG FILE LOADED')

    def save_config(self, filename):
        config = configparser.ConfigParser()

        # agent
        config.add_section(('agent'))
        config.set('agent','value_r_lower_range', str(self.value_r_lower_range))
        config.set('agent','value_r_higher_range',str(self.value_r_higher_range))
        config.set('agent','sigma', str(self.sigma))
        config.set('agent','agent_type',str(self.agent_type))
        config.set('agent','init_cash', str(self.init_cash))
        config.set('agent','lambda_value', str(self.lambda_value))
        config.set('agent','percentage_aux_agents', str(self.percentage_aux_agents))

        # seed
        config.add_section('seed')
        config.set('seed','random_seed', str(self.random_seed))
        config.set('seed','batch_size', str(self.batch_size))

        # parallel
        config.add_section('parallel')
        config.set('parallel','number_of_threads', str(self.number_of_threads))

        # market
        config.add_section('market')
        config.set('market','market_duration', str(self.market_duration))
        config.set('market','liquidity_constant', str(self.liquidity_constant))
        config.set('market','init_price', str(self.init_price))
        config.set('market', 'percent', str(self.percent))
        # ga
        config.add_section('ga')
        config.set('ga','number_generations', str(self.number_generations))
        config.set('ga', 'no_of_agents_per_market', str(self.no_of_agents_per_market))
        config.set('ga','retain_top_k_agents_per_market', str(self.retain_top_k_agents_per_market))
        config.set('ga', 'top_agents_n', str(self.top_agents_n))
        # files
        config.add_section('files')
        config.set('files', 'parent_dir', str(self.parent_dir))
        config.set('files','output_folder_location', str(self.output_folder_location.replace(self.parent_dir, '')))
        config.set('files','training_feature_file_location', str(self.training_feature_file_location))
        config.set('files','testing_feature_file_location', str(self.testing_feature_file_location))
        config.set('files','intermediate_file_location', str(self.intermediate_file_location.replace(self.parent_dir, '')))
        config.set('files', 'market_model_location', str(self.market_model_location.replace(self.parent_dir, '')))
        
        # aux
        config.add_section('aux')
        config.set('aux','expert', str(self.expert))
        config.set('aux','proficient', str(self.proficient))
        config.set('aux','competent', str(self.competent))
        config.set('aux','beginner', str(self.beginner))
        config.set('aux','novice', str(self.novice))

        with open(filename, 'w') as configfile:
            config.write(configfile)
        print('CONFIG FILE SAVED')
