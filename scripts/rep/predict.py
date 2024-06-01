import sys
sys.path.append('../../')
from market.utils.initial_config import initial_configuration
from market.config import MarketConfig
import pandas as pd
from market.core.models import MarketRegressor
import numpy as np
import re
from tqdm import tqdm
# Load the config
CONFIG_FILE = 'config.ini'
config = MarketConfig()
config.load_config(CONFIG_FILE)
rng = np.random.default_rng(config.random_seed)
data = pd.read_csv(config.testing_feature_file_location)

for i in tqdm(range(3,11)):
    for seed in [32, 34, 36]:
        config.percentage_aux_agents = i*0.1
        config.random_seed = seed
        config.output_folder_location = f'../../data/aux_rep/equal_mix_{i}/{seed}/output_{i}_{seed}/'

        # Load data
        X = initial_configuration(data.iloc[:, 3:-1], output_path=config.intermediate_file_location, train=False, config=config, normalize=False)
        y_gt = data.iloc[:, -1].tolist()
        # Load the previously trained model
        model = MarketRegressor(config, train=False)
        y_pred, interpret, price_history = model.predict(X, config.market_model_location, rng=rng, test_labels=y_gt)

        writer = pd.ExcelWriter(config.output_folder_location + f"testing_output_{i}_{seed}.xlsx")
        pd.DataFrame(y_pred).to_excel(writer, sheet_name = "Predictions")
        data.to_excel(writer, sheet_name = "Features")
        writer.close()
