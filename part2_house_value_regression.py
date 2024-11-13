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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Regressor(torch.nn.Module):

    def __init__(
            self,
            x,
            nb_epoch=1000,
            batch_size=32,
            learning_rate=0.00025,
            layers=[64, 64, 32],
        ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - batch_size {int} -- size of the batch to train the network.
            - learning_rate {float} -- learning rate for the optimizer.
            - layers {list} -- list of integers, where each integer
                corresponds to the number of neurons in a hidden layer.
        """

        # Replace this code with your own
        super().__init__()

        X, _ = self._preprocessor(x, y=None, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Hyper-parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Feature Engineering
        self._transformer = None
        self.y_scaling = 100000  # Scaling factor for the target variable

        # Define NN Layers
        layer_sizes = [self.input_size] + layers + [self.output_size]
        self._layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])

        # Apply Xavier Initialization
        for layer in self._layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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

        # Return preprocessed x and y, return None for y if it was None

        # If x is not a DataFrame, return x and y as is
        if not isinstance(x, pd.DataFrame):
            return x, y

        # Apply useful feature combinations
        x['rooms_per_household'] = x['total_rooms'] / x['households']
        x['bedrooms_per_household'] = x['total_bedrooms'] / x['households']
        x['population_per_household'] = x['population'] / x['households']
        x['income_squared'] = x['median_income'] ** 2

        # Remove extraneous features
        x = x.drop(columns=[
            'households',
            'total_rooms',
            'total_bedrooms',
            'population',
            # 'median_income',
        ])

        categorical_features = x.select_dtypes(include=['object']).columns
        numeric_features = x.select_dtypes(include=['number']).columns

        # Fill NAN values with the mean of the respective feature
        x.loc[:, numeric_features] = x[numeric_features].fillna(x[numeric_features].mean())

        if training:
            # Define the transformer
            self._transformer = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    # ('cat', OneHotEncoder(), categorical_features)
                    ('cat', OrdinalEncoder(), categorical_features)
                ], remainder='passthrough'
            )
            # Fit and transform the data
            x = self._transformer.fit_transform(x)
        else:
            x = self._transformer.transform(x)

        x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor

        if y is None:
            return x, None

        # Fill NAN values in the target variable with the mean
        y = y.fillna(y.mean())
        # Preprocess y, the target price dataframe
        y = torch.tensor(y.values, dtype=torch.float32)
        y = y / self.y_scaling  # Scale target variable

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
        self.eval()
        X, _ = self._preprocessor(x, training = False) # Do not forget

        with torch.no_grad():
            predictions = self.forward(X)

        return predictions.numpy() * self.y_scaling

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

        X, Y = self._preprocessor(x, y=y, training=False) # Do not forget
        predictions = self.predict(X)

        Y = Y.numpy() * self.y_scaling
        mse = mean_squared_error(Y, predictions)
        mae = mean_absolute_error(Y, predictions)
        r2 = r2_score(Y, predictions)
        rmse = np.sqrt(mse)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")
        print(f"Root Mean Squared Error: {rmse}")

        plt.figure()
        plt.plot(Y, predictions, '.')
        plt.show()

        return rmse


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
    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]

    # Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(X_train, nb_epoch=200, batch_size=16, layers=[128, 128, 128, 128, 128])
    regressor.fit(X_train, Y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(X_test, Y_test)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()
