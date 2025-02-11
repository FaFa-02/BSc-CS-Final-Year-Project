"""Module initialising program application with appropriate user functions"""
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from IPython.display import display
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from ridge_regression import RidgeRegression
from k_nearest_neighbors import KNearestNeighbors

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
    student_data_adjusted = pd.get_dummies(student_data, drop_first=True, dtype=int)
    student_features = (student_data_adjusted.drop('G3', axis=1)).to_numpy()
    student_labels = (student_data_adjusted['G3']).to_numpy()

    # Read crime dataset and seperate features and labels
    crime_data = pd.read_csv("Datasets/crimedata.csv", encoding='latin-1', na_values=["?"])
    # Drop information only columns and other target variables
    crime_data = crime_data.drop(crime_data.columns[list(range(0,5)) + list(range(129, 145)) + [146]], axis=1)
    # Remove police entries
    crime_data_adjusted = crime_data.drop(crime_data.columns[list(range(98, 115)) + list(range(118, 124))], axis=1)
    # Remove rows with null variables
    crime_data_adjusted = crime_data_adjusted.dropna(axis=0)
    crime_features = (crime_data_adjusted.drop('ViolentCrimesPerPop', axis=1)).to_numpy()
    crime_labels = (crime_data_adjusted['ViolentCrimesPerPop']).to_numpy()

    data_list = [[boston_data, boston_data_adjusted, boston_features, boston_labels],
                 [student_data, student_data_adjusted, student_features, student_labels],
                 [crime_data, crime_data_adjusted, crime_features, crime_labels]]

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

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
        
        # Opens KNN regression menu window when pressed
        tk.Button(self.menu,
            text="KNN Regression Model",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_knn(self)
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

        # Opens new ridge regression model window
        def load_ridge(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = RidgePage(self.newWindow) 

        # Opens new ridge regression model window
        def load_knn(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = KNNPage(self.newWindow)

        # Opens new POC window
        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

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
        tk.Radiobutton(self.data_vis_page, text='Student Dataset', variable=var, value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.data_vis_page, text='Crime Dataset', variable=var, value=2).pack(anchor=tk.W)

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

        # Toggle to show or not values in correlation matrix
        var2 = tk.BooleanVar()
        tk.Checkbutton(self.data_vis_page, 
                text='Display Correlation Values', 
                variable=var2, 
                onvalue=True, 
                offvalue=False
                ).pack()

        # Button that displays correlation matrix for selected dataset when pressed
        tk.Button(self.data_vis_page,
                text="Correlation Matrix",
                font=("TkMenuFont", 14),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:corr_matrix(self, (Menu.data_list)[var.get()][1], var2.get())
                ).pack()

        # Button that displays eigenvalues for selected dataset when pressed
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

        def corr_matrix(self, dataset, display_vals):
            sns.set(font_scale=2)
            plt.figure(figsize=(20, 10))
            if display_vals == True:
                corr = dataset.corr().abs()
                mask_corr = corr[(corr >= 0.70) | (corr <= -0.70)]
                sns.heatmap(mask_corr,  annot=True, fmt=".2f")
            else:
                sns.heatmap(dataset.corr().abs())
            plt.show()

        # Computes eigenvalues of a given dataset
        def comp_eigenvals(feature_set):
            # Create symmetric matrix brfore compting eigenvalues
            for i in range((feature_set.T).shape[0]):
                col = feature_set.T[i]
                feature_set.T[i] = col - col.mean()
            XTX = np.matmul(np.transpose(feature_set), feature_set)
            cov_m = XTX / (feature_set.shape[0] - 1)

            # Compute eigenvals and square them if positive
            eigenvals = np.linalg.eigvalsh(cov_m)
            for i in range(eigenvals.shape[0]):
                if eigenvals[i] >= 0:
                    eigenvals[i] = np.sqrt(eigenvals[i])

            # Sort eigenvalues with largest first
            sorted_eigenvals = np.sort(eigenvals)

            return sorted_eigenvals[::-1]

        # Computes condition indicies of a dataset given its eigenvalues
        def comp_ci(eigenvals):
            ci = []

            for i in range(eigenvals.size):
                # Check for zero and negative eigenvals
                if eigenvals[i] > 0:
                    ci.append(np.round(np.sqrt(np.max(eigenvals) / eigenvals[i])))
                else:
                    ci.append("N/A")

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
            vis_df['Eigenvalues'] = vis_df['Eigenvalues'].apply(lambda x: '{:.2e}'.format(x))
            
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
            ridge = RidgeRegression(a)
            ridge.fit(X_train, y_train)

            y_hat = ridge.predict(X_test)
            ridge.score(X_test, y_test)

class RidgePage:
    """Class representing the ridge regression model program window"""
    def __init__(self, parent):
        self.parent = parent
        self.ridge_page = tk.Frame(self.parent, width=500, height=400, bg=BG_COLOUR)
        self.ridge_page.grid(row=0, column=0)
        self.ridge_page.pack_propagate(False)

        # Model parameters
        self.alpha = 0
        self.train_size = 0.75

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
        tk.Radiobutton(self.ridge_page, text='Student Dataset', variable=var, value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.ridge_page, text='Crime Dataset', variable=var, value=2).pack(anchor=tk.W)

        tk.Label(self.ridge_page,
            text="Set ratio of training vs test size",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Takes value from scale and updates train_size variable with it
        def update_train_size(v):
            self.train_size = float(v)

        # Scale to define size of training set relative to test set
        tk.Scale(self.ridge_page, from_=0.00, to=1.00, resolution=0.05, orient="horizontal", command=update_train_size).pack()

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

        # Button to run cross val and find optimal parameters
        tk.Button(self.ridge_page,
            text="Optimal parameters (cross-val)",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:hyperparam_tunning(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3])
            ).pack()

        tk.Label(self.ridge_page,
            text="Predict and Score Model",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
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

        output_score = tk.Label(self.ridge_page,
            text="",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 10)
            )
        output_score.pack()

        output_std = tk.Label(self.ridge_page,
            text="",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 10)
            )
        output_std.pack()

        # Takes value from text field and updates alpha variable with it
        def update_alpha(self):
            self.alpha = float(alpha_input.get("1.0", "end-1c"))

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_ridge(self, data_features, data_labels, rnd_state, graph=True, opt_alpha=None):

            # Split dataset into training and test sets in preparation for the Ridge Regression model
            X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size, random_state=rnd_state)

            # Apply normalisation to dataset before predicting
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            X_train_scaled = std_scaler.transform(X_train)
            X_test_scaled = std_scaler.transform(X_test)

            # Initialized and fit training data to Ridge Regression model
            if opt_alpha is None:
                ridge = RidgeRegression(self.alpha)
            else:
                ridge = RidgeRegression(opt_alpha)
            
            ridge.fit(X_train_scaled, y_train)

            # Predict values and output their score and plot predicted vs true points
            return ridge.score(X_test_scaled, y_test, graph)

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_ridge_errors(self, data_features, data_labels):
            acc_scores_arr = []

            # Find optimal alpha, build and score model with random states
            for i in RAND_STATES:
                alpha = hyperparam_tunning(self, data_features, data_labels, i)
                acc_scores_arr.append(predict_ridge(self, data_features, data_labels, i, None, opt_alpha=alpha))

            output_score.config(text="mean score: " + str(np.mean(acc_scores_arr)))
            output_std.config(text="std error: " + str( np.std(acc_scores_arr) / np.sqrt(len(acc_scores_arr))))

            print("mean score:",np.mean(acc_scores_arr))
            print("std error:",( np.std(acc_scores_arr) / np.sqrt(len(acc_scores_arr))))

        # Returns optimal hyperparameter for knn model by using cross validation
        def hyperparam_tunning(self, data_features, data_labels, rnd=None):

            if rnd is None:
                X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size, random_state=rnd)

            best_score = 0
            for alpha in [1e-3, 1e-2, 1e-1, 1, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 1000]:
                # For each possible parameter train a ridge model
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                # Perform cross-validation
                scores = cross_val_score(pipe, X_train, y_train, cv=5)
                score = np.mean(scores)
                # Store best score from cross-val of all possibilities
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            # Rebuild a model on full training set and check performance
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_train, y_train)
            test_score = ridge.score(X_test, y_test)
            print("best CV score:", best_score)
            print("best alpha value:", best_alpha)
            print("test score on test set using best parameters:", test_score)

            return best_alpha

class KNNPage:
    """Class representing the KNN model program window"""
    def __init__(self, parent):
        self.parent = parent
        self.knn_page = tk.Frame(self.parent, width=500, height=400, bg=BG_COLOUR)
        self.knn_page.grid(row=0, column=0)
        self.knn_page.pack_propagate(False)

        # Model parameters
        self.n = 3
        self.train_size = 0.75

        # Title for choosing dataset
        tk.Label(self.knn_page,
            text="Select Dataset to Analyse",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Radio buttons for choosing dataset
        var = tk.IntVar()
        tk.Radiobutton(self.knn_page, text='Boston Housing Dataset', variable=var, value=0).pack(anchor=tk.W)
        tk.Radiobutton(self.knn_page, text='Student Dataset', variable=var, value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.knn_page, text='Crime Dataset', variable=var, value=2).pack(anchor=tk.W)

        tk.Label(self.knn_page,
            text="Set ratio of training vs test size",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()
        
        # Takes value from scale and updates train_size variable with it
        def update_train_size(v):
            self.train_size = float(v)

        # Scale to define size of training set relative to test set
        tk.Scale(self.knn_page, 
            from_=0.00, to=1.00, 
            resolution=0.05, 
            orient="horizontal", 
            command=update_train_size
            ).pack()

        tk.Label(self.knn_page,
            text="Set KNN Regression Parameters",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Text field to get user inputed value for n
        n_input = tk.Text(self.knn_page,
                bg="#EEDFCC",
                height=1,
                width=5
                )
        n_input.pack()
        n_input.insert(tk.END, 3)

        # Button to update n value with user inputed data
        tk.Button(self.knn_page,
            text="Select n neighbours",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:update_n(self)
            ).pack()

        # Button to run cross val and find optimal parameters
        tk.Button(self.knn_page,
            text="Optimal parameters (cross-val)",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:hyperparam_tunning(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3])
            ).pack()

        tk.Label(self.knn_page,
            text="Predict and Score Model",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

        # Button to run model on test data and output results
        tk.Button(self.knn_page,
            text="Predict test set",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:predict_knn(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3], 0)
            ).pack()

        # Button to run model on test data with 5 random states and output mean score
        tk.Button(self.knn_page,
            text="Predict test set with errors",
            font=("TkMenuFont", 8),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:predict_knn_errors(self, (Menu.data_list)[var.get()][2], (Menu.data_list)[var.get()][3])
            ).pack()

        output_score = tk.Label(self.knn_page,
            text="",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 10)
            )
        output_score.pack()

        output_std = tk.Label(self.knn_page,
            text="",
            bg=BG_COLOUR,
            fg="black",
            font=("TkMenuFont", 10)
            )
        output_std.pack()

        # Takes value from text field and updates n variable with it
        def update_n(self):
            self.n = int(n_input.get("1.0", "end-1c"))

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_knn(self, data_features, data_labels, rnd_state, graph=True, opt_n=None):

            # Split dataset into training and test sets in preparation for the Ridge Regression model
            X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size,random_state=rnd_state)
            
            # Apply normalisation to dataset before predicting
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            X_train_scaled = std_scaler.transform(X_train)
            X_test_scaled = std_scaler.transform(X_test)

            # Initialized and fit training data to KNN Regression model
            if opt_n is None:
                knn = KNearestNeighbors(self.n)
            else:
                knn = KNearestNeighbors(opt_n)
            knn.fit(X_train_scaled, y_train)

            # Predict values and output their score and plot predicted vs true points
            return knn.score(X_test_scaled, y_test, graph)

        # Instantiates and trains model to dataset, then executes on test set and output results
        def predict_knn_errors(self, data_features, data_labels):
            acc_scores_arr = []

            for i in RAND_STATES:
                n = hyperparam_tunning(self, data_features, data_labels, i)
                acc_scores_arr.append(predict_knn(self, data_features, data_labels, i, None, opt_n=n))
            
            output_score.config(text="mean score: " + str(np.mean(acc_scores_arr)))
            output_std.config(text="std error: " + str( np.std(acc_scores_arr) / np.sqrt(len(acc_scores_arr))))
            
            print("mean score:",np.mean(acc_scores_arr))
            print("std error:",( np.std(acc_scores_arr) / np.sqrt(len(acc_scores_arr))))

        # Returns optimal hyperparameter for knn model by using cross validation
        def hyperparam_tunning(self, data_features, data_labels, rnd=None):

            if rnd is None:
                X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, train_size=self.train_size, random_state=rnd)

            best_score = 0
            for n in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
                # For each possible parameter train a knn model
                pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n))
                # Perform cross-validation
                scores = cross_val_score(pipe, X_train, y_train, cv=5)
                score = np.mean(scores)
                # Store best score from cross-val of all possibilities
                if score > best_score:
                    best_score = score
                    best_n = n
            # Rebuild a model on full training set and check performance
            knn = KNeighborsRegressor(n_neighbors=best_n)
            knn.fit(X_train, y_train)
            test_score = knn.score(X_test, y_test)
            print("best CV score:", best_score)
            print("best n value:", best_n)
            print("test score on test set using best parameters:", test_score)

            return best_n

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
