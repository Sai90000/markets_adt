# Market
<p align="center">
    <a href="#Installation">Installation</a> &bull;
    <a href="#Publications">Publications</a> &bull;
    <a href="#Training">Training</a> &bull;
    <a href="#Predictions">Predictions</a> 
</p>


## Introduction
<!-- TODO - make better -->
The Market, organized as a module, is a synthetic market model that can be trained on any classification or regression 
data with the proper settings. It is optimized using a genetic algorithm. 

## Publications
* [Design and Analysis of a Synthetic Prediction Market using Dynamic Convex Sets](https://arxiv.org/pdf/2101.01787.pdf)
* [A Synthetic Prediction Market for Estimating Confidence in Published Work](https://arxiv.org/pdf/2201.06924v1.pdf)

## Installation
### 1. Pull market code
From the parent folder you want the Market in 

```

```

### 2. Virtual Environment
Using your preferred virtual environment method create an environment for the project.

* Conda: 
```
$ cd Market 
$ conda create --name <env_name> --file requirements.txt
```
* venv: 
```
$ cd Market
$ python -m virtualenv <env_name>
$ <env_name>/bin/activate
$ pip install -r requirements.txt
```

## Training 
#### 1. Data
Create a csv file with training data. The last column should be the target variable. Place any non-feature columns 
(IDs, for example) as the first columns on the left, just be sure to note the column where your features begin. Pictured
below we see that there are no non-feature columns in our heart data example and the target is the last column in the 
file.  
![Alt text](Images/data.png?raw=true "Data")

#### 2. <a name="traincode"></a> Code Setup

<u>**Template**</u>

The directory scripts/heart_example_scripts can be used as a template for the experimental setup. Making a copy of this 
for your experiment and then going through the following setup is the easiest way to train a model. 

<u>**train.py**</u>

Open train.py. Put the column where your features start in front of the ":" in the red box below, make sure to use 
zero-indexing. In the blue box below, the default is to use the mean of the data as the initial market price or
you can choose to set your own through the settings file by commenting those lines out. 
![Alt text](Images/train.png?raw=true "Train")

<u>**config.ini**</u>

In this file you will setup your experimental parameters. Here's a few notes and definitions to help, for more detail on training the market with the evolutionary algorithm please sees section 4 of our [January 2021 paper](https://arxiv.org/pdf/2101.01787.pdf).
* sigma - standard deviation of the normal distribution from which the perturbation in agent weights is sampled during the mutation.
* init_cash - the amount of cash an agent starts with. this should be considered in the context of market_duration because the amount an agent could spend is bounded by $1 * market_duration.
* lambda_value - parameter of the exponential distribution from which time between participation opportunities for the agents are determined. PDF is $$\lambda e^{-\lambda x}$$
it should be noted that $1/\lambda$ is the mean of the distribution
* market_duration - the number of time steps in a generation
* liquidity_constant - affects how much price is changed due to a transaction using PNAS. larger liquidity constant --> less price change per transaction
* init_price - the initial market price of the market. If using mean this will be updated when train.py is executed.
* percent - the percentage difference between the agent's estimated value of a share and the market price required for a transaction to occur.
* number_generations - number of generations in evolutionary algorithm. More generations increases training time.
* no_of_agents_per_market - the number of agents associated with each sample that participate in the market.
* retain_top_k_agents_per_market - the number of "best" agents associated with each sample retained after a generation. others are discarded and replaced with random mutations of these. 
* top_agents_n - the number of agents that will score points toward survival for their performance in the market of a given sample.
* output_folder_location - where to store the model, this is typically a subdirectory of data/output
* training_feature_file_location - where the data csv file is located, this is typically in the directory data/feature_pipeline
* testing_feature_file_location  - this is not needed for training a model, but is included because the configuration file parser will throw an error without it
* intermediate_file_location - after pre-processing the resulting data is stored in csv file here. this is typically in the directory data/intermediate_data

#### 3. Execution
Execute your train.py file. Make sure your environment is activated first.
```
cd path/to/Market/scripts/your_folder
python train.py
```
Then you should see output to the console looking something like this 
![Alt text](Images/console_output.png?raw=true "Console Output")

When the code finishes running, you will see a file called FINAL_AGENT_WEIGHTS.csv in the output directory you specified in the settings file. The settings file is also saved for reference and future prediction tasks that may require it. training.xlsx shows the loss for each generation in the evolutionary algorithm, if you prefer a different generation than the final for your model, you can use one of the files from the previous_generation_weights directory.

### Predictions
#### 1. Data
Create a csv file with features you want to make predictions from. You may not have a target variable in the data, but if it is it should be the last column. The features should be in the same order as they were in the training data, this is very important. If you're unsure 
Place any non-feature columns (IDs, for example) as the first columns on the left, just be sure to note the column where your features begin. Pictured
below we see that there are no non-feature columns in our heart data example and the target is the last column in the 
file. We will need to keep this in mind when we edit the code.
![Alt text](Images/data.png?raw=true "Data")
#### 2. Code Setup
<u>**Template**</u>

The directory scripts/heart_example_scripts can be used as a template for the experimental setup. 

<u>**predict.py**</u>

Open predict.py. Put the column where your features start in front of the ":" in the blue box below, make sure to use 
zero-indexing. If you have a target value in the dataset, make sure there is a -1 after the : in the blue box, as shown below.
![Alt text](Images/predict.png?raw=true "Predict")

**config.ini**

In this file you will setup your experimental parameters. If you trained the market as in the previous section, then you can keep the same config file. If not, then copy the config.ini file from the folder that contains the model you wish to use into the directory that contains your predict.py file. Set the directory of output_folder_location to where you want your output to be saved. Set the testing_feature_file_location with the path to your data file from step one. Ensure the market_model_location is indeed the location of the trained marke that you wish to use. For more information check out the [Code Setup](#traincode) section in the training the model part above. 

#### 3. Execution
Execute your predict.py file. Make sure your environment is activated first.
```
cd path/to/Market/scripts/your_folder
python predict.py
```
Then you should see output to the console looking something like pictured below, actual numbers will vary with data used.
![Alt text](Images/predict_output.png?raw=true "Predict Output")

When the code finishes running, there will be a folder populated where you told it to save the output. The testing_output.xlsx file will have the scores predicted for each sample and the features that were used for reference. They are not labeled in the predictions but are in the same order as upload. There is also a .json file for each sample that was predicted, which shows the transaction activity of the market that created the prediction. A json dictionary, it should be easy to parse for data visualizations, etc.

