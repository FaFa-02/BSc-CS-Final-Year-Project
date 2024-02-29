"""Module initialising program application with appropriate user functions"""
import tkinter as tk
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from ridge_regression import RidgeRegressionClassifier
import matplotlib.pyplot as plt

BG_COLOUR = "#fff"

class Menu:
    """Class representing the main menu window"""
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

        # Opens POC menu window when pressed
        tk.Button(self.menu,
            text="Proof of Concept",
            font=("TkMenuFont", 20),
            bg=BG_COLOUR,
            fg="black",
            cursor="hand2",
            command=lambda:load_poc(self)
            ).pack()

        #opens new POC window
        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

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

            """
            # Plots prediction versus true labels
            fig, ax = plt.subplots(3, figsize=(15, 15))
            plt.suptitle("Linnerud_pairplot")

            x = np.linspace(-2, 2, 100)
            for i in range(3):
                print(i)
                ax[i].scatter(X[:,i], y, s=100)
                ax[i].scatter(X_test[:,i], y_hat)
                ax[i].set_xticks(())
                ax[i].set_yticks(())
                ax[i].set_xlabel(linnerud['feature_names'][i])
                ax[i].set_ylabel(linnerud['target_names'][1])

            plt.show()
            """



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
