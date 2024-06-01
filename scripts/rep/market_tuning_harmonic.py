import sys
sys.path.append('../../')
import optuna
from dynaconf import Dynaconf
from market.parameter_tuning.optuna import Objectives, Hyperparameter, Criterion, optuna_best_trials_to_dataframe
from market.config import MarketConfig
from market.core.models import MarketRegressor
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
import statistics
import pandas as pd
import multiprocessing as mp
import argparse
from scripts.cv_example.cross_validation import train_test
import logging
from pathlib import Path
import datetime
from market.utils.initial_config import initial_configuration
import os
import numpy as np
from sklearn.model_selection import KFold


TEST = False 
SPLITS = 5

def prep_data(
        config,
        train
        ):
    if train:
        data = pd.read_csv(config.training_feature_file_location)
    else:
        data = pd.read_csv(config.testing_feature_file_location)

    data = data.iloc[:, 3:]

    X = initial_configuration(data, output_path=config.intermediate_file_location, train=train, normalize=True)
    start_config = X[:, -2:]
    y = X[:, -3]
    X = X[:, :-3]

    return X, y, start_config

def f1_calc(
        y_pred,
        y_gt
        ):

    y_final = []

    for i in y_pred:
        if i > 0.5:
            y_final.append(1)
        else:
            y_final.append(0)

    f1_class0 = f1_score(y_final, y_gt, pos_label=0)
    f1_class1 = f1_score(y_final, y_gt)
    f1_hm = statistics.harmonic_mean([f1_class0, f1_class1])

    return f1_hm

def save_preds(
            data,
            y_pred,
            y_gt,
            file_name,
            config
            ):
    writer = pd.ExcelWriter(config.output_folder_location + file_name)
    pd.DataFrame(y_pred).to_excel(writer, sheet_name = "Predictions")
    pd.DataFrame(y_gt).to_excel(writer, sheet_name="ground truth")
    pd.DataFrame(data).to_excel(writer, sheet_name = "Features")
    writer.save()
    return

def mse_calc(
            y_pred,
            y_gt
            ):
    return mean_squared_error(y_gt, y_pred)

def mae_calc(
            y_pred,
            y_gt
            ):
    return mean_absolute_error(y_gt, y_pred)

def objective(
        trail,
        criteria,
        #batch_size=32,
        #random_seed=None,
        number_of_threads=8,
        #value_r_lower_range=None,
        #value_r_higher_range=None,
        market_duration=None,
        liquidity_constant=None,
        percent=None,
        number_generations=None,
        no_of_agents_per_market=None,
        retain_top_k_agents_per_market=None,
        top_agents_n=None,
        init_cash=None,
        lambda_value=None,
        config_file='config.ini'
        ):

    try:
#    if True:
        config = MarketConfig()
        config.load_config(config_file)

        rng = np.random.default_rng(config.random_seed)
                #config.random_seed = random_seed.constant
        #config.batch_size = batch_size#.constant
        #config.value_r_lower_range = trail.suggest_uniform(
        #                                                   value_r_lower_range.name,
        #                                                   value_r_lower_range.min,
        #                                                   value_r_lower_range.max
        #                                                   )


        #config.value_r_higher_range = trail.suggest_uniform(
        #                                                   value_r_higher_range.name,
        #                                                   config.value_r_lower_range,
        #                                                   value_r_higher_range.max
        #                                                   )

        config.market_duration = trail.suggest_int(
                                                  market_duration.name,
                                                  market_duration.min,
                                                  market_duration.max
                                                  )
        config.liquidity_constant = trail.suggest_int(
                                                   liquidity_constant.name,
                                                   liquidity_constant.min,
                                                   liquidity_constant.max
                                                   )
        config.init_cash = trail.suggest_int(
                                           init_cash.name,
                                           init_cash.min,
                                           init_cash.max
                                           )
        config.lambda_value = trail.suggest_uniform(
                                           lambda_value.name,
                                           lambda_value.min,
                                           lambda_value.max
                                           )
        config.percent = trail.suggest_uniform(
                                           percent.name,
                                           percent.min,
                                           percent.max
                                           )
        config.number_generations = trail.suggest_int(
                                                   number_generations.name,
                                                   number_generations.min,
                                                   number_generations.max
                                                   )
        config.no_of_agents_per_market = trail.suggest_int(
                                                           no_of_agents_per_market.name,
                                                           no_of_agents_per_market.min,
                                                           no_of_agents_per_market.max
                                                           )
        config.retain_top_k_agents_per_market = trail.suggest_int(
                                                                  retain_top_k_agents_per_market.name,
                                                                  no_of_agents_per_market.min,
                                                                  config.no_of_agents_per_market
                                                                  )
        config.top_agents_n = trail.suggest_int(
                                               top_agents_n.name,
                                               top_agents_n.min,
                                               top_agents_n.max
                                               )

        config.intermediate_file_location = config.intermediate_file_location.replace('trial_num', str(trail.number))
        config.market_model_location = config.market_model_location.replace('trial_num', str(trail.number))
        config.output_folder_location = config.output_folder_location.replace('trial_num', str(trail.number))

        if not os.path.exists(config.intermediate_file_location):
            loc = os.path.dirname(config.intermediate_file_location) + '/'
            os.makedirs(loc)
#vvvvvvvvvvvvvvvvvvvvvvvvvvv
        if not os.path.exists(config.market_model_location):
            os.makedirs(config.market_model_location)
        else:
            print('POSSIBLE ERROR: Already a folder in this location', config.market_model_location)
#^^^^^^^^^^^^^^^^^^^^^^^^^

        filename = config.market_model_location + 'results_log.txt'
        timing_file = config.market_model_location + 'timing_log.txt'
        formatter = logging.Formatter('%(message)s')
        formatter2 = logging.Formatter('%(asctime)s - %(message)s')
        logger = logging.getLogger('logger')
        logger2 = logging.getLogger('logger2')
        logger.setLevel(logging.INFO)
        logger2.setLevel(logging.INFO)

        # Load data, set some apart for training
        data = pd.read_csv(config.training_feature_file_location)

        if False:
            # for testing code - reduce size of dataset
            data = data.iloc[-int(np.floor(0.2*data.shape[0])):, :]
        else:
            data = data.iloc[:, :]
        data.to_csv('data.csv')
        split = 1
        kf = KFold(n_splits=SPLITS, shuffle=True, random_state=config.random_seed)
        rngs = rng.spawn(SPLITS)
        args = []
        for train_index, test_index in kf.split(data):
            args.append((train_index, test_index, split, config,
                        data, logger, logger2, rngs[split-1]))
            split += 1

        # Set up multiprocessing
        num_processors = 8
        m = mp.Manager()
        flag_dict = m.dict()
        p = mp.Pool(num_processors)
        results = p.starmap(train_test, args)
        p.close()
        p.join()
        val_acc_all = []
        mape_all = []
        tau_all = []
        pvals_all = []
        writer = pd.ExcelWriter(
            f'{config.market_model_location}/cross_validation_scores_summary.xlsx')
        k = 1
        for i in results:
            val_acc_all.append(i[0])
            mape_all.append(i[1])
            tau_all.append(i[2])
            pvals_all.append(i[3])
            print('*********************************************')
            pd.DataFrame(i[4]).transpose().to_excel(
                writer, sheet_name=f'Split_{k}')
            pred = pd.DataFrame(i[5])
            label = pd.DataFrame(i[6])
            pred_df = pd.concat([pred, label], axis=1)
            pred_df.columns = ['predicted', 'actual']
            pred_df.to_excel(writer, sheet_name=f'pred {k}')
            k += 1
        df = pd.DataFrame()
        df["RMSE"] = val_acc_all
        df["MAPE"] = mape_all
        df["Tau"] = tau_all
        df["P-Val"] = pvals_all
        df.to_excel(writer, sheet_name='Metrics')
        writer.close()
#-----------

        output = tuple([
            df['RMSE'].mean()])

        config.save_config(config.market_model_location + '/config_final.ini')
        return output

    except:
#    else:
        settings = Dynaconf(settings_files=['tune_config.yaml'])
        with open(settings.paths.experiment_location + '/failed_runs.txt', 'w+') as file:
            file.write(str(trail.number))
            file.write('\n')
        return 1

def bayesian_tune(
        experiment_parameters,
        n_trials,
        ):

    study = optuna.create_study(
            study_name='default',
            directions=experiment_parameters.get_criteria_directions()
            )

    study.optimize(
            experiment_parameters,
            n_trials=n_trials

            )
    return study
if __name__=='__main__':
    settings = Dynaconf(settings_files=['tune_config.yaml'])
    print(settings.paths.experiment_location)
    print(settings.as_dict())
    print('33333333')
    print(settings.paths)
    experiment_parameters = Objectives(
            [

                # Hyperparameter('value_r_lower_range',
                #     min=settings.agent.value_r_lower_range.min,
                #     max=settings.agent.value_r_lower_range.max),
                #
                # Hyperparameter('value_r_higher_range',
                #     min=settings.agent.value_r_higher_range.min,
                #     max=settings.agent.value_r_higher_range.max),
                #
                Hyperparameter('init_cash',
                    min=settings.agent.init_cash.min,
                    max=settings.agent.init_cash.max),

                Hyperparameter('lambda_value',
                    min=settings.agent.lambda_value.min,
                    max=settings.agent.lambda_value.max),

                Hyperparameter('market_duration',
                    min=settings.market.market_duration.min,
                    max=settings.market.market_duration.max),

                Hyperparameter('liquidity_constant',
                    min=settings.market.liquidity_constant.min,
                    max=settings.market.liquidity_constant.max),

                Hyperparameter('percent',
                    min=settings.market.percent.min,
                    max=settings.market.percent.max),

                Hyperparameter('number_generations',
                    min=settings.ga.number_generations.min,
                    max=settings.ga.number_generations.max),

                Hyperparameter('no_of_agents_per_market',
                    min=settings.ga.no_of_agents_per_market.min,
                    max=settings.ga.no_of_agents_per_market.max),

                Hyperparameter('retain_top_k_agents_per_market',
                    min=settings.ga.retain_top_k_agents_per_market.min,
                    max=settings.ga.retain_top_k_agents_per_market.max),

                Hyperparameter('top_agents_n',
                    min=settings.ga.top_agents_n.min,
                    max=settings.ga.top_agents_n.max)

                ],[
                Criterion('f1', 'maximize', range=(0,1))
                ],
                objective,
                features=None
                )

    study = bayesian_tune(
            experiment_parameters,
            n_trials=settings.n_trials
            )

    with open(settings.paths.experiment_location + '/best_trials.txt', 'w+') as file:
        file.write(f'trail number: {study.best_trial.number} with value {study.best_trial.value} best params: {study.best_trial.params}')
    #df_best = optuna_best_trials_to_dataframe(study, ['f1_train', 'f1_test'])
    #df_best.to_csv(settings.paths.experiment_location + '/best_trials.csv', index=False) ##settings.paths.experiment_location +
