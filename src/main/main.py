"""Module initialising program application with appropriate user functions"""
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import seaborn as sns
import random
from ridge_regression import RidgeRegressionClassifier


BG_COLOUR = "#fff"

RAND_STATES = [102388621, 320893374, 564240724, 173580668, 142586193]

class Menu:
    """Class representing the main menu window"""

    # Read Boston Housing dataset and seperate features and labels
    col_names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_data = pd.read_csv("Datasets/boston_housing.csv", delimiter=r"\s+", names=col_names)
    boston_data_adjusted = boston_data[boston_data.MEDV != 50.00]
    boston_features = (boston_data_adjusted.drop('MEDV', axis=1)).to_numpy()
    boston_labels = (boston_data_adjusted['MEDV']).to_numpy()

    # Read student dataset and seperate features and labels
    student_data = pd.read_csv("Datasets/student-por.csv", delimiter=';')
    student_data = student_data.drop(['school', 'reason'], axis=1)
    student_data_adjusted = pd.get_dummies(student_data, drop_first=True)
    student_features = (student_data.drop('G3', axis=1)).to_numpy()
    student_labels = (student_data['G3']).to_numpy()

    # Read conductor dataset and seperate features and labels
    conductivity_data = pd.read_csv("Datasets/con_train.csv")
    conductivity_features = (conductivity_data.drop('critical_temp', axis=1)).to_numpy()
    conductivity_labels = (conductivity_data['critical_temp']).to_numpy()


    data_list = [[boston_data, boston_data_adjusted, boston_features, boston_labels],
                 [student_data, student_data, student_features, student_labels],
                 [conductivity_data, conductivity_data, conductivity_features,conductivity_labels]]

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print("head:",student_data_adjusted.head(5))
    print("shape:",student_data_adjusted.shape)

    def __init__(self, parent):
        self.parent = parent
        self.menu = tk.Frame(self.parent, width=500, height=600, bg=BG_COLOUR)
        self.menu.grid(row=0, column=0)
        self.menu.pack_propagate(False)

        # Title for data vis menus
        tk.Label(self.menu,
            text="Data Visualisation Menu",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Opens the data visualisation menu window
        tk.Button(self.menu,
            text="Data Visualisation",
            font=("TkMenuFont", 14),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_data_vis_page(self)
            ).pack()

        # Title for regression menus
        tk.Label(self.menu,
            text="Ridge & KNN Regression Menu",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
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
        
        # Opens POC menu window when pressed
        tk.Button(self.menu,
            text="Proof of Concept",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_poc(self)
            ).pack()

        # Opens new data visualisation window
        def load_data_vis_page(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = DataVisPage(self.newWindow)

        # Opens new POC window
        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

        # Opens new ridge regression model window
        def load_ridge(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = RidgePage(self.newWindow) 

class DataVisPage():
    """Class representing the data visualisation program window"""
    def __init__(self, parent):
        self.parent = parent
        self.data_vis_page = tk.Frame(self.parent, width=500, height=300, bg=BG_COLOUR)
        self.data_vis_page.grid(row=0, column=0)
        self.data_vis_page.pack_propagate(False)

        # Title for choosing dataset
        tk.Label(self.data_vis_page,
            text="Select Dataset to Analyse",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        var = tk.IntVar()
        # Radio buttons for choosing dataset
        tk.Radiobutton(self.data_vis_page, text='Boston Housing Dataset', variable=var, value=0).pack(anchor=tk.W)
        tk.Radiobutton(self.data_vis_page, text='Dataset', variable=var, value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.data_vis_page, text='Million Song Dataset', variable=var, value=2).pack(anchor=tk.W)

        # Title for data vis options
        tk.Label(self.data_vis_page,
            text="Visualisation Methods",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Button that displays describe function for selected dataset when pressed
        tk.Button(self.data_vis_page,
                text="Describe Function",
                font=("TkMenuFont", 14),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:data_describe(self, (Menu.data_list)[var.get()][0])
                ).pack()

        # Button that displays correlation matrix for selected dataset when pressed
        tk.Button(self.data_vis_page,
                text="Correlation Matrix",
                font=("TkMenuFont", 14),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:corr_matrix(self, (Menu.data_list)[var.get()][1])
                ).pack()

        # Button that displays data visualisation for selected dataset when pressed
        tk.Button(self.data_vis_page,
                text="Eignvaleus & Condition Indices",
                font=("TkMenuFont", 14),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:load_data_vis(self, (Menu.data_list)[var.get()][2])
                ).pack()

        def data_describe(self, dataset):
            with open("output.txt", "w") as text_file:
                text_file.write(dataset.describe(include='all').to_string())

        def corr_matrix(self, dataset):
            plt.figure(figsize=(20, 10))
            sns.heatmap(dataset.corr().abs(),  annot=True)
            print(dataset.shape)
            plt.show()

        # Computes eigenvalues of a given dataset
        def comp_eigenvals(feature_set):
            # Create symmetric matrix brfore compting eigenvalues
            for i in range((feature_set.T).shape[0]):
                col = feature_set.T[i]
                feature_set.T[i] = col - col.mean()
            XTX = np.matmul(np.transpose(feature_set), feature_set)
            cov_m = XTX / (feature_set.shape[0] - 1)
            eigenvals = np.sqrt(np.linalg.eigvals(cov_m))

            #feature_set_std = StandardScaler().fit_transform(feature_set)

            #covariance_matrix = np.cov(np.transpose(feature_set_std))
            #eigenvals = np.linalg.eigvals(covariance_matrix)

            return np.round(eigenvals, decimals=4)

        # Computes condition indicies of a dataset given its eigenvalues
        def comp_ci(eigenvals):
            ci = np.arange(1,eigenvals.size+1)

            for i in range(eigenvals.size):
                ci[i] = np.sqrt(np.max(eigenvals) / eigenvals[i])

            return ci

        # Generates and displays visualisation of data
        def load_data_vis(self, feature_set):
            # Compute eigenvalues and condition indices for dataset
            eignvals = comp_eigenvals(feature_set)
            ci = comp_ci(eignvals)

            # Dataframe containing eigenvalues and condition indicies
            vis_df = pd.DataFrame({
                "Component": [x for x in range(1, eignvals.size+1)],
                "Eigenvalues": eignvals,
                "Condition Indicies": ci
            })

            # Plot eigenvalues against their indexes
            np.arange(1,eignvals.size)
            plt.plot(np.arange(1,eignvals.size+1), eignvals, '-o')
            plt.xlabel("Component Number")
            plt.ylabel("EigenValues")
            plt.title("Scree Plot")

            display(vis_df)
            vis_df.to_excel("Eigenvalues_and_Condition_indicies.xlsx", index=None)

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

        # Model parameters
        self.alpha = 0

        # Title for choosing dataset
        tk.Label(self.ridge_page,
            text="Select Dataset to Analyse",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Radio buttons for choosing dataset
        var = tk.IntVar()
        tk.Radiobutton(self.ridge_page, text='Boston Housing Dataset', variable=var, value=0).pack(anchor=tk.W)
        tk.Radiobutton(self.ridge_page, text='Dataset', variable=var, value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.ridge_page, text='Million Song Dataset', variable=var, value=2).pack(anchor=tk.W)

        tk.Label(self.ridge_page,
            text="Set Ridge Regression Parameters",
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
            command=lambda:predict_ridge(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3], 0)
            ).pack()

        # Button to run model on test data with 5 random states and output mean score
        tk.Button(self.ridge_page,
            text="Predict test set with errors",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:predict_ridge_errors(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3])
            ).pack()

        # Takes value from text field and updates alpha variable with it
        def update_alpha(self):
            self.alpha = float(alpha_input.get("1.0", "end-1c"))

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_ridge(self, data_features, data_labels, rnd_state, graph=True):

            # Split dataset into training and test sets in preparation for the Ridge Regression model
            X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, random_state=rnd_state)

            # Apply normalisation to dataset before predicting
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            X_train_scaled = std_scaler.transform(X_train)
            X_test_scaled = std_scaler.transform(X_test)

            # Initialized and fit training data to Ridge Regression model
            ridge = RidgeRegressionClassifier(self.alpha)
            ridge.fit(X_train_scaled, y_train)

            # Predict values and output their score and plot predicted vs true points
            return ridge.score(X_test_scaled, y_test, graph)

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_ridge_errors(self, data_features, data_labels):
            acc_scores_arr = []

            for i in RAND_STATES:
                acc_scores_arr.append(predict_ridge(self, data_features, data_labels, i, None))

            print("mean score:",np.mean(acc_scores_arr))
            print("std error:",( np.std(acc_scores_arr) / np.sqrt(len(acc_scores_arr))))

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
