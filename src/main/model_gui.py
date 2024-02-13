import tkinter as tk

root = tk.Tk()
root.title("Model Menu")
root.eval("tk::PlaceWindow . center")

frame1 = tk.Frame(root, width=500, height=600, bg="#fff")
frame1.grid(row=0, column=0)

root.mainloop()
