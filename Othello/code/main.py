from train import train_model
from play import ReversiApp
import tkinter as tk

def main():
    print("1. Train AI")
    print("2. Play against AI")
    choice = input("Enter your choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        root = tk.Tk()
        app = ReversiApp(root)
        root.mainloop()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()