import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class Regressor(torch.nn.Module):

    def __init__(self, x, nb_epoch=1000, batch_size=32, learning_rate=0.00025):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super().__init__()

        categorical_features = x.select_dtypes(include=['object']).columns
        numeric_features = x.select_dtypes(include=['int64', 'float64']).columns

        self._transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                # ('cat', OneHotEncoder(), categorical_features)
                ('cat', OrdinalEncoder(), categorical_features)
            ], remainder='passthrough'
        )

        X, _ = self._preprocessor(x, y=None, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Hyper-parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Define NN Layers
        layer_sizes = [self.input_size, 64, 64, 32, self.output_size]
        self._layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])

        # Apply Xavier Initialization
        for layer in self._layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, X):
        """
        Forward pass through the network.

        Arguments:
            x {torch.Tensor} -- Input tensor of shape (batch_size, num_input_features).

        Returns:
            torch.Tensor -- Output tensor of shape (batch_size, num_output_features).
        """

        # Apply RELU activation function to all hidden layers
        for layer in self._layers[:-1]:
            X = torch.relu(layer(X))
        # Apply final layer without activation function
        X = self._layers[-1](X)
        return X

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # Preprocess x, the feature dataframe
        numeric_cols = x.select_dtypes(include=[np.number]).columns
        x[numeric_cols] = x[numeric_cols].fillna(x[numeric_cols].mean())

        if training:
            x = self._transformer.fit_transform(x)
        else:
            x = self._transformer.transform(x)

        x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor

        if y is None:
            return x, None

        # Preprocess y, the target price dataframe
        y = torch.tensor(y.values, dtype=torch.float32)
        y = y / 100000  # Scale target variable

        # Y_train = Y_train / y_scaling  # Scale target variable

        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True) # Do not forget

        # Create a TensorDataset to pair features and labels
        train_dataset = TensorDataset(X, Y)

        # Create a DataLoader to shuffle and batch the data
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss function (Mean Squared Error)
        criterion = nn.MSELoss()

        # Define optimizer
        # optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.00025, momentum=0.9)

        # Training loop
        for epoch in range(self.nb_epoch):
            self.train()  # Set model to training mode

            batch_loss = 0
            for X_batch, Y_batch in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                predictions = self.forward(X_batch)
                loss = criterion.forward(input=predictions, target=Y_batch)
                # Backward pass
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                batch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.nb_epoch}, Loss: {batch_loss/len(train_loader)}")

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()
