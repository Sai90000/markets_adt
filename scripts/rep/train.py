import time
start_time = time.time()
import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import os
from market.core.models import MarketRegressor
from market.config import MarketConfig
from market.utils.initial_config import initial_configuration

# Define the config input
CONFIG_FILE = 'config.ini'
config = MarketConfig()
config.load_config(CONFIG_FILE)
print('+'*20)
print(config.market_model_location)
rng = np.random.default_rng(config.random_seed)

# Load the data
data = pd.read_csv(config.training_feature_file_location)
data = data.iloc[:, 3:]

# Make the init price the mean of the training labels - comment out for specified mean from settings
config.init_price = data.iloc[:, -1].mean()
config.save_config('config.ini')

if not os.path.exists(config.intermediate_file_location):
    loc = os.path.dirname(config.intermediate_file_location) + '/'
    os.makedirs(loc)
X = initial_configuration(data, output_path=config.intermediate_file_location, config=config, normalize=True)

start_config = X[:, -2:]
y = X[:, -3]
X = X[:, :-3]
model = MarketRegressor(config)

model.fit(rng=rng, X=X, y=y, start_config=start_config)
print(time.time() - start_time)
print(time.time() - start_time)
