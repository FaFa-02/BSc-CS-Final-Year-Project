"""Module initialising program application with appropriate user functions"""
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from ridge_regression import RidgeRegressionClassifier


BG_COLOUR = "#fff"

class Menu:
    """Class representing the main menu window"""

    # Read Boston Housing dataset
    col_names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_data = pd.read_csv("Boston_Housing_Dataset/housing.csv", delimiter=r"\s+", names=col_names)

    # Seperates features and labels
    boston_features = (boston_data.drop('MEDV', axis=1)).to_numpy()
    boston_labels = (boston_data['MEDV']).to_numpy()

    def __init__(self, parent):
        self.parent = parent
        self.menu = tk.Frame(self.parent, width=500, height=600, bg=BG_COLOUR)
        self.menu.grid(row=0, column=0)
        self.menu.pack_propagate(False)

        # menu frame widgets
        tk.Label(self.menu,
            text="Ridge & Lasso Regression Menu",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Opens the data visualisation window when pressed
        tk.Button(self.menu,
            text="Data Visualisation",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_data_vis_page(self)
            ).pack()

        # Opens POC menu window when pressed
        tk.Button(self.menu,
            text="Proof of Concept",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_poc(self)
            ).pack()

        # Opens ridge regression menu window when pressed
        tk.Button(self.menu,
            text="Ridge Regression Model",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_ridge(self)
            ).pack()

        # Opens new POC window
        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

        # Opens new ridge regression model window
        def load_ridge(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = RidgePage(self.newWindow) 

        # Opens new data visualisation window
        def load_data_vis_page(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = DataVisPage(self.newWindow)

class DataVisPage():
    """Class representing the data visualisation program window"""
    def __init__(self, parent):
        self.parent = parent
        self.data_vis_page = tk.Frame(self.parent, width=500, height=300, bg=BG_COLOUR)
        self.data_vis_page.grid(row=0, column=0)
        self.data_vis_page.pack_propagate(False)

        # Button that displays data visualisation for Boston housing dataset when pressed
        tk.Button(self.data_vis_page,
                text="Eignvaleus for Boston Housing Dataset",
                font=("TkMenuFont", 14),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:load_data_vis_boston(self)
                ).pack()

        # Generates and displays visualisation of data
        def load_data_vis_boston(self):
            # Compute eigenvalues of Boston dataset, first create symmetric matrix
            XTX = np.dot(np.transpose(Menu.boston_features), Menu.boston_features)
            boston_eignvals = np.linalg.eigvals(XTX)
            print(boston_eignvals)

            # Plot eigenvalues against their indexes
            np.arange(1,boston_eignvals.size)
            plt.plot(np.arange(1,boston_eignvals.size+1), boston_eignvals)
            plt.xlabel("Component Number")
            plt.ylabel("EigenValues")
            plt.title("Scree Plot")
            plt.show()


class Poc:
    """Class representing the proof of concept program window"""
    def __init__(self, parent):
        self.parent = parent
        self.poc = tk.Frame(self.parent, width=500, height=300, bg=BG_COLOUR)
        self.poc.grid(row=0, column=0)
        self.poc.pack_propagate(False)

        # Model parameters
        self.alpha = 0

        # Importing linnerud dataset, seperate dataset into its features and labels(waist)
        linnerud = load_linnerud()

        X = linnerud['data']
        y = linnerud['target'][:,1]

        # Split dataset into training and test sets in preparation for the Ridge Regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

        tk.Label(self.poc,
            text="Proof of concept Ridge Regression Program",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Button that displays data visualisation when pressed
        tk.Button(self.poc,
            text="Data Visualisation",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_data_vis(self)
            ).pack()

        # Text field to get user inputed value for alpha
        alpha_input = tk.Text(self.poc,
                bg="#EEDFCC",
                height=1,
                width=5
                )
        alpha_input.pack()
        alpha_input.insert(tk.END, 0)

        # Button to update alpha value with user inputed data
        tk.Button(self.poc,
            text="Select alpha",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:update_alpha(self)
            ).pack()

        # Button to run model on test data and output results
        tk.Button(self.poc,
            text="Predict test set",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:predict_poc(self, self.alpha)
            ).pack()

        # Generates and displays visualisation of data
        def load_data_vis(self):
            fig, ax = plt.subplots(3, figsize=(15, 15))
            plt.suptitle("Linnerud_pairplot")

            x = np.linspace(-2, 2, 100)
            for i in range(3):
                print(i)
                ax[i].scatter(X[:,i], y, s=100)
                ax[i].set_xticks(())
                ax[i].set_yticks(())
                ax[i].set_xlabel(linnerud['feature_names'][i])
                ax[i].set_ylabel(linnerud['target_names'][1])

            plt.show()

        # Takes value from text field and updates alpha variable with it
        def update_alpha(self):
            self.alpha = float(alpha_input.get("1.0", "end-1c"))

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_poc(self, a):
            ridge = RidgeRegressionClassifier(a)
            ridge.fit(X_train, y_train)

            y_hat = ridge.predict(X_test)
            ridge.score(X_test, y_test)

class RidgePage:
    """Class representing the ridge regression model program window"""
    def __init__(self, parent):
        self.parent = parent
        self.ridge_page = tk.Frame(self.parent, width=500, height=300, bg=BG_COLOUR)
        self.ridge_page.grid(row=0, column=0)
        self.ridge_page.pack_propagate(False)

        # Split dataset into training and test sets in preparation for the Ridge Regression model
        X_train, X_test, y_train, y_test = train_test_split(Menu.boston_features, Menu.boston_labels, random_state=0)

        # Model parameters
        self.alpha = 0

        tk.Label(self.ridge_page,
            text="Ridge Regression on Boston Housing Problem",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Text field to get user inputed value for alpha
        alpha_input = tk.Text(self.ridge_page,
                bg="#EEDFCC",
                height=1,
                width=5
                )
        alpha_input.pack()
        alpha_input.insert(tk.END, 0)

        # Button to update alpha value with user inputed data
        tk.Button(self.ridge_page,
            text="Select alpha",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:update_alpha(self)
            ).pack()

        # Button to run model on test data and output results
        tk.Button(self.ridge_page,
            text="Predict test set",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:predict_ridge(self, self.alpha, "House Prices")
            ).pack()

        # Takes value from text field and updates alpha variable with it
        def update_alpha(self):
            self.alpha = float(alpha_input.get("1.0", "end-1c"))

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_ridge(self, a, label_name):
            # Initialized and fit training data to Ridge Regression model
            ridge = RidgeRegressionClassifier(a)
            ridge.fit(X_train, y_train)

            # Predict values and output their score and plot predicted vs true points
            ridge.score(X_test, y_test, label_name)

def main():
    """Class representing the root window"""
    # Initialises the application on menu page
    root = tk.Tk()
    app = Menu(root)
    root.title("Model Menu")
    root.eval("tk::PlaceWindow . center")

    # Run appplication
    root.mainloop()

if __name__ == '__main__':
    main()
