import tkinter as tk

bg_colour = "#fff"

# Loads POC menu frame (only test for now)
def load_POC():
    print("test")

# Initialises the application on menu page
root = tk.Tk()
root.title("Model Menu")
root.eval("tk::PlaceWindow . center")

# Create frame widget
frame1 = tk.Frame(root, width=500, height=600, bg=bg_colour)
frame1.grid(row=0, column=0)
frame1.pack_propagate(False)

# frame1 widgets
tk.Label(frame1,
         text="Ridge & Lasso Regression Menu",
         bg=bg_colour,
         fg="black",
         font=("TkMenuFont", 14)
         ).pack()

tk.Button(frame1,
          text="Proof of Concept",
          font=("TkMenuFont", 20),
          bg=bg_colour,
          fg="black",
          cursor="hand2",
          command=lambda:load_POC()
          ).pack()

root.mainloop()
