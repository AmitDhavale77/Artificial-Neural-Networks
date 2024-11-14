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
from sklearn.model_selection import ParameterSampler


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the earth.
    Parameters:
        - lat1, lon1: Latitude and longitude of the first point in degrees.
        - lat2, lon2: Latitude and longitude of the second point in degrees.
    Returns:
        - Distance in kilometers between the two points.
    """

    R = 6371  # Earth radius in kilometers
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Apply Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_city_distances(df, lat_col='latitude', lon_col='longitude'):
    """
    Adds Haversine distance columns to a DataFrame for specified cities.
    Parameters:
        - df: pandas DataFrame with latitude and longitude columns.
        - lat_col: name of the latitude column in df.
        - lon_col: name of the longitude column in df.
    Returns:
        - Updated DataFrame with distance columns for each city.
    """

    # Define prominent cities in California
    cities = {
        'distance_to_SF': (37.7749, -122.4194),
        'distance_to_LA': (34.0522, -118.2437),
        'distance_to_SD': (32.7157, -117.1611),
        'distance_to_SJ': (37.3382, -121.8863)
    }

    for city_name, (city_lat, city_lon) in cities.items():
        df[city_name] = df.apply(
            lambda row: haversine_distance(row[lat_col], row[lon_col], city_lat, city_lon),
            axis=1
        )

    return df


class Regressor(torch.nn.Module):

    def __init__(
        self,
        x,
        nb_epoch=1000,
        batch_size=32,
        learning_rate=0.00025,
        num_layers=3,
        layer_size=64,
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

        X, _ = self._preprocessor(x, y=None, training=True)
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
        layer_sizes = [self.input_size] + [layer_size] * num_layers + [self.output_size]
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

    def _preprocessor(self, x, y=None, training=False):
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

        x = add_city_distances(x, lat_col='latitude', lon_col='longitude')

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

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        # Create a TensorDataset to pair features and labels
        train_dataset = TensorDataset(X, Y)

        # Create a DataLoader to shuffle and batch the data
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss function (Mean Squared Error)
        criterion = nn.MSELoss()

        # Define optimizer
        # optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.00025, momentum=0.9)

        # Store loss history
        loss_history = []

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

            cur_loss = batch_loss/len(train_loader)
            loss_history.append(cur_loss)
            print(f"Epoch {epoch+1}/{self.nb_epoch}, Loss: {cur_loss}")

        return self, loss_history

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        self.eval()
        X, _ = self._preprocessor(x, training=False)  # Do not forget

        with torch.no_grad():
            predictions = self.forward(X)

        return predictions.numpy() * self.y_scaling

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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
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


def perform_hyperparameter_search(x_train, y_train, x_val, y_val, n_iter_search=None):
    """
    Performs a hyper-parameter search for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        x_train {pd.DataFrame}: Training features
        y_train {pd.DataFrame}: Training targets
        x_val {pd.DataFrame}: Validation features
        y_val {pd.DataFrame}: Validation targets

    Returns:
        dict: The best set of hyperparameters and their corresponding validation score.
    """

    # Define hyperparameter grid
    param_grid = {
        'num_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # Number of hidden layers
        'neurons': [8, 16, 32, 64, 128],          # Neurons per layer
        'batch_size': [16, 32, 64],        # Batch sizes
        'epochs': [10, 20, 50, 100],             # Number of epochs
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01]
    }

    best_score = float('-inf')
    best_params = None

    if n_iter_search is not None:
        random_search = ParameterSampler(param_grid, n_iter=n_iter_search, random_state=42)
        param_grid['num_layers'] = random_search['num_layers']
        param_grid['neurons'] = random_search['neurons']
        param_grid['batch_size'] = random_search['batch_size']
        param_grid['epochs'] = random_search['epochs']
        param_grid['learning_rate'] = random_search['learning_rate']

    # Iterate through all possible combinations of hyperparameters
    for num_layers in param_grid['num_layers']:
        for neurons in param_grid['neurons']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    for learning_rate in param_grid['learning_rate']:
                        print(f"Training with layers={num_layers}, neurons={neurons}, batch_size={batch_size}, epochs={epochs}")

                        # Initialize the Regressor with current hyperparameters
                        regressor = Regressor(x_train, nb_epoch=epochs, batch_size=batch_size, learning_rate=learning_rate, num_layers=num_layers, layer_size=neurons)

                        # Train the model
                        regressor.fit(x_train, y_train)

                        # Evaluate the model on the validation set
                        score = regressor.score(x_val, y_val)
                        print(f"Validation R² score: {score}")

                        # Track the best score and hyperparameters
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'num_layers': num_layers,
                                'neurons': neurons,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

    print(f"\nBest Hyperparameters: {best_params} with R² score: {best_score}")
    return best_params


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
    regressor = Regressor(X_train, nb_epoch=10, batch_size=16, num_layers=3, layer_size=32)
    _, loss_history = regressor.fit(X_train, Y_train)
    plt.figure()
    plt.plot(loss_history)
    plt.show()
    save_regressor(regressor)

    # Error
    error = regressor.score(X_test, Y_test)
    print(f"\nRegressor error: {error}\n")

    print(perform_hyperparameter_search(X_train, Y_train, X_test, Y_test))


if __name__ == "__main__":
    example_main()
