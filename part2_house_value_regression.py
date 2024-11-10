import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
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
        X, y = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.forward = self._forward(X)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def _forward(self, X):

        input_tensor = X.view(-1, self.input_size)
        model = nn.Sequential(
            nn.Linear(input_tensor, 256),   # Input layer -> Hidden layer 1 (128 units)
            nn.ReLU(),                         # ReLU activation function
            nn.Linear(256, 128),                 # Hidden layer 1 -> Hidden layer 2 (64 units)
            nn.ReLU(),   
            nn.Linear(128, 64),                 # Hidden layer 1 -> Hidden layer 2 (64 units)
            nn.ReLU(),                        # ReLU activation function
            nn.Linear(64, self.output_size)     # Hidden layer 2 -> Output layer (1 unit for regression)
        )
        
        return model

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

        lb = LabelBinarizer()
        x_raw = x
        y = y

        categorical_cols = x_raw .select_dtypes(include=['object']).columns

        for col in categorical_cols:
            x_raw[col] = x_raw[col].fillna(x_raw[col].mode()[0])

        binary_cols = lb.fit_transform(x_raw[categorical_cols])

        if binary_cols.shape[1] > 1:
            for i in range(binary_cols.shape[1]):
                x_raw[f"{list(categorical_cols)}_{i}"] = binary_cols[:, i]
        else:
            x_raw[categorical_cols] = binary_cols

        columns_to_remove = [
            "Index(['ocean_proximity'], dtype='object')_0",
            "Index(['ocean_proximity'], dtype='object')_1",
            "Index(['ocean_proximity'], dtype='object')_2",
            "Index(['ocean_proximity'], dtype='object')_3",
            "Index(['ocean_proximity'], dtype='object')_4"
        ]

        x_raw.drop(columns=columns_to_remove, errors='ignore', inplace=True)

        redundant_columns = ['households', 'total_bedrooms']
        x_raw.drop(columns=redundant_columns, inplace=True, errors='ignore')


        X_filled = x_raw.apply(lambda col: col.fillna(col.mean()), axis=0)
        y_filled = y.fillna(y.mean())
        one_hot_columns = [col for col in X_filled.columns if '_0' in col or '_1' in col or '_2' in col or '_3' in col]  # Adjust this condition based on your one-hot column names
        continuous_columns = [col for col in X_filled.columns if col not in one_hot_columns]

        # Step 2: Apply Z-score normalization (standardization) to continuous columns
        scaler = StandardScaler()
        X_filled[continuous_columns] = scaler.fit_transform(X_filled[continuous_columns])

        
        # Step 1: Convert X and y into PyTorch tensors
        X_tensor = torch.tensor(X_filled.values, dtype=torch.float32)  # Convert X to tensor of float type
        y_tensor = torch.tensor(y_filled.values, dtype=torch.float32)  # Convert y to tensor of float type (or long for classification)

        x = X_tensor
        y = y_tensor
        return x, (y if isinstance(y, pd.DataFrame) else None)

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

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        learning_rate = 0.001
        # Define loss function (Mean Squared Error) and optimizer (Adam)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(self.nb_epoch):
            # Forward pass
            outputs = self.forward(X)
            loss = criterion(outputs, Y)
            
            # Backward pass
            optimizer.zero_grad()   # Zero the gradients
            loss.backward()         # Backpropagate the loss
            optimizer.step()        # Update the model parameters
            
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{self.nb_epoch}], Loss: {loss.item()}")

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

