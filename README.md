# Neural Networks Coursework

This code will produce the example of a trained neural network that gives the optimal r2 score. The parameters were chosen based on hyperparameter tuning. 

The Python version and library requirements are in requirements.txt

## Project Structure

#### `part1_nn_lib.py`

This script contains a mini-library for building a neural network. It evaluates the performance of a example neural network on iris data set.

#### `part2_house_value_regression.py`

This script trains the neural network based on hyperparameters found to give optimal values in perform_hyperparameter_search function. Uncomment the ... line in order to perform grid search for best hyperparameters.


#### `data_exploration.py`

This script contains the experiments of initial data analysis and identifies missing values and highly correlated columns. 

#### `plotting_hyperparam_tuning.py`

This script produces plots of r2 scores of validation of test sets for different hyperparameters values saved while performing hyperparameter tuning.

#### `part2_model.pickle`

This file contains the uploaded model with hyperparameters corresponding to the optimal performance.


## Instructions

Running the script to train an example neural network using built mini-library and evaluate it on iris dataset:

```bash
python part1_nn_lib.py
```

Configuration Options:

- Input Data File: change part1_nn_lib.py line 600 (default: iris.dat)
- \# of input dimensions to the neural network: change part1_nn_lib.py line 595 (default: 4)
- \# of neurons in each layer in the neural network: change part1_nn_lib.py line 596 (default: [16,3])
- activation fuction: change part1_nn_lib.py line 597 (default: ["relu", "identity"])
- batch size: change part1_nn_lib.py line 620 (default: 8)
- \# of epochs: change part1_nn_lib.py line 621 (default: 1000)
- learning rate: change part1_nn_lib.py line 622 (default: 0.01)
- loss function: part1_nn_lib.py line 623 (default: "cross_entropy")

Running the script to train an example neural network using PyTorch and evaluate it on housing dataset. The hyperparameters were chosen after performing hyperparameter tuning and correspond to the most optimal ones.

```bash
python part2_house_value_regression.py
```

- Input Data File: change part2_house_value_regression.py line 504 (default: housing.csv)
- Test size: change part2_house_value_regression.py line 511 (default: 0.2)
- \# of epochs: change part2_house_value_regression.py line 519 (default: 100)
- batch size: change part2_house_value_regression.py line 520 (default: 16)
- \# of layers: change part2_house_value_regression.py line 521 (default: 5)
- \# of neurons in each layer: change part2_house_value_regression.py line 522 (default: 32)
- learning rate: change part2_house_value_regression.py line 523 (default: 0.00025)
- In order to run hyperparameter tuning uncomment line 552



