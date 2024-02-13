import tkinter as tk

bg_colour = "#fff"

def load_menu():
    menu.pack_propagate(False)

    # menu frame widgets
    tk.Label(menu,
            text="Ridge & Lasso Regression Menu",
            bg=bg_colour,
            fg="black",
            font=("TkMenuFont", 14)
            ).pack()

    tk.Button(menu,
            text="Proof of Concept",
            font=("TkMenuFont", 20),
            bg=bg_colour,
            fg="black",
            cursor="hand2",
            command=lambda:load_POC()
            ).pack()

# Loads POC menu frame (only test for now)
def load_POC():
    print("test")

# Initialises the application on menu page
root = tk.Tk()
root.title("Model Menu")
root.eval("tk::PlaceWindow . center")

# Create frame widget
menu = tk.Frame(root, width=500, height=600, bg=bg_colour)
poc = tk.Frame(root, bg=bg_colour)

for frame in (menu, poc):
    frame.grid(row=0, column=0)

load_menu()
# Run appplication
root.mainloop()
