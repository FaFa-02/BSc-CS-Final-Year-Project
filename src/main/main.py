import tkinter as tk

bg_colour = "#fff"

class Menu:
    def __init__(self, parent):
        self.parent = parent
        self.menu = tk.Frame(self.parent, width=500, height=600, bg=bg_colour)
        self.menu.grid(row=0, column=0)
        self.menu.pack_propagate(False)

        # menu frame widgets
        tk.Label(self.menu,
                text="Ridge & Lasso Regression Menu",
                bg=bg_colour,
                fg="black",
                font=("TkMenuFont", 14)
                ).pack()

        tk.Button(self.menu,
                text="Proof of Concept",
                font=("TkMenuFont", 20),
                bg=bg_colour,
                fg="black",
                cursor="hand2",
                command=lambda:load_poc(self)
            ).pack()

        def load_poc(self):
            self.newWindow = tk.Toplevel(self.parent)
            self.app = Poc(self.newWindow)

class Poc:
    def __init__(self, parent):
        self.parent = parent
        self.poc = tk.Frame(self.parent, width=500, height=600, bg=bg_colour)
        self.poc.grid(row=0, column=0)
        self.poc.pack_propagate(False)

def main():
    # Initialises the application on menu page
    root = tk.Tk()
    app = Menu(root)
    root.title("Model Menu")
    root.eval("tk::PlaceWindow . center")

    # Run appplication
    root.mainloop()

if __name__ == '__main__':
    main()
