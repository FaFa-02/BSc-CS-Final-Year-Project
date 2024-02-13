"""Module initialising program application with appropriate user functions"""
import tkinter as tk

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

        tk.Button(self.menu,
                text="Proof of Concept",
                font=("TkMenuFont", 20),
                bg=BG_COLOUR,
                fg="black",
                cursor="hand2",
                command=lambda:load_poc(self)
            ).pack()

        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

class Poc:
    """Class representing the proof of concept program window"""
    def __init__(self, parent):
        self.parent = parent
        self.poc = tk.Frame(self.parent, width=500, height=600, bg=BG_COLOUR)
        self.poc.grid(row=0, column=0)
        self.poc.pack_propagate(False)

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
