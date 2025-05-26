import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os

GRID_SIZE = 28   # dimensiunea grilei (28x28)
CELL_SIZE = 15  # dimensiunea unei celule

class HopfieldApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg="#f2f2f2")
        self.patterns = []       # lista cu cifrele antrenate (vectori + etichete)
        self.grid_data = np.full((GRID_SIZE, GRID_SIZE), -1)

        self.build_ui()          # construieste interfata grafica
        self.clear_grid()        

    def build_ui(self):
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)     # aici am activat desenul cu mouse-ul

        tk.Button(self.root, text="Load Train Set", command=self.load_train_set, bg="#64b5f6", fg="white").grid(row=0, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Clear", command=self.clear_grid, bg="#e57373", fg="white").grid(row=1, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Recognize", command=self.recognize_digit, bg="#81c784", fg="white").grid(row=2, column=1, sticky="ew", padx=10)


        self.result_label = tk.Label(self.root, text="Predicted: ", font=("Helvetica", 20, "bold"), bg="#f2f2f2")
        self.result_label.grid(row=3, column=1, pady=20)

    def clear_grid(self):
        self.grid_data = np.full((GRID_SIZE, GRID_SIZE), -1)  # initializare: toate celulele = -1 (alb)
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.create_rectangle(
                    j*CELL_SIZE, i*CELL_SIZE, (j+1)*CELL_SIZE, (i+1)*CELL_SIZE,
                    fill="white", outline="gray", tags=f"cell_{i}_{j}"
                )

    def paint(self, event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid_data[y][x] = 1
            self.canvas.itemconfig(f"cell_{y}_{x}", fill="black")

    def load_train_set(self):
        dir_path = filedialog.askdirectory(title="Select folder with 28x28 digit PNGs")
        if not dir_path:
            return

        self.patterns.clear()
        for file in os.listdir(dir_path):
            if file.lower().endswith(".png"):
                path = os.path.join(dir_path, file)
                vector = self.image_to_vector(path)
                if vector is not None:
                    label = file[0]  
                    self.patterns.append((vector, label))

        if not self.patterns:
            messagebox.showwarning("Warning", "No valid PNG files found!")
            return

        messagebox.showinfo("Load", f"Loaded {len(self.patterns)} images")

    def image_to_vector(self, path):
        try:
            img = Image.open(path).convert("L")
            img = ImageOps.invert(img).resize((GRID_SIZE, GRID_SIZE))
            data = np.array(img)
            return np.where(data > 128, 1, -1).flatten()
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
     
    def recognize_digit(self):
        current_matrix = self.grid_data.copy()        
        current_vector = current_matrix.flatten()     #am citit ca cica pt retea ar fi mai ok sa tinem in vector decat matrice

        print("Matrice 28x28:")
        print(current_matrix)
        print("Vector 784 elemente:")
        print(current_vector)

  
if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldApp(root)
    root.mainloop()