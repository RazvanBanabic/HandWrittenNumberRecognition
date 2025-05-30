import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image
import numpy as np
import os
from HopfieldNetwork import HopfieldNetwork
from scipy.ndimage import center_of_mass, shift, binary_dilation

GRID_SIZE = 28
CELL_SIZE = 15

def center_image(data):
    if np.sum(data) == 0:
        return data
    mass = center_of_mass(data)
    shift_y = data.shape[0] // 2 - mass[0]
    shift_x = data.shape[1] // 2 - mass[1]
    return shift(data, shift=(shift_y, shift_x), cval=0, mode='constant')

class HopfieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Digit Recognition")
        self.patterns = []
        self.grid_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.hopfield = HopfieldNetwork(GRID_SIZE * GRID_SIZE)
        self.build_ui()
        self.clear_grid()

    def build_ui(self):
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=7, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        tk.Button(self.root, text="Load Train Set", command=self.load_train_set, bg="#64b5f6", fg="white").grid(row=0, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Clear", command=self.clear_grid, bg="#e57373", fg="white").grid(row=1, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Recognize", command=self.recognize_digit, bg="#81c784", fg="white").grid(row=2, column=1, sticky="ew", padx=10)

        self.result_label = tk.Label(self.root, text="Predicted: None", font=("Helvetica", 16))
        self.result_label.grid(row=5, column=1, pady=10)

    def clear_grid(self):
        self.grid_data = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.create_rectangle(
                    j*CELL_SIZE, i*CELL_SIZE, (j+1)*CELL_SIZE, (i+1)*CELL_SIZE,
                    fill="white", outline="lightgray", tags=f"cell_{i}_{j}"
                )

    def paint(self, event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    self.grid_data[ny][nx] = 1
                    self.canvas.itemconfig(f"cell_{ny}_{nx}", fill="black")

    def load_train_set(self):
        dir_path = filedialog.askdirectory(title="Select folder with 28x28 digit PNGs")
        if not dir_path:
            return
        self.patterns.clear()
        vectors = []
        for file in os.listdir(dir_path):
            if file.lower().endswith(".png"):
                vector = self.image_to_vector(os.path.join(dir_path, file))
                if vector is not None:
                    label = file.split('_')[0]
                    self.patterns.append((vector, label))
                    vectors.append(vector)
        if self.patterns:
            self.hopfield.train(vectors)
            messagebox.showinfo("Load", f"Loaded {len(self.patterns)} images")
        else:
            messagebox.showwarning("Warning", "No valid PNG files found!")

    def image_to_vector(self, path):
        try:
            img = Image.open(path).convert("L").resize((GRID_SIZE, GRID_SIZE))
            data = np.asarray(img)
            threshold = np.percentile(data, 75)
            binary = np.where(data < threshold, 1, 0)
            binary = center_image(binary)
            sparsity = np.sum(binary == 1) / binary.size
            if sparsity < 0.20:
                threshold = np.percentile(data, 60)
                binary = np.where(data < threshold, 1, 0)
                binary = center_image(binary)
            if np.sum(binary == 1) / binary.size < 0.15:
                binary = binary_dilation(binary, iterations=1)
            return np.where(binary == 1, 1, -1).flatten()
        except:
            return None

    def recognize_digit(self):
        if not self.patterns:
            messagebox.showwarning("Warning", "Please load training patterns first!")
            return
        binary = np.where(self.grid_data == 1, 1, 0)
        binary = center_image(binary)
        if np.sum(binary) / binary.size < 0.15:
            binary = binary_dilation(binary, iterations=1)
        input_vec = np.where(binary == 1, 1, -1).flatten()
        recalled = self.hopfield.recall(input_vec.copy(), max_steps=10)
        label = self.find_best_match(recalled)
        self.result_label.config(text=f"Predicted: {label}")

    def find_best_match(self, vector):
        best_label, best_score = None, -np.inf
        for ref_vec, label in self.patterns:
            cosine = np.dot(ref_vec, vector) / (np.linalg.norm(ref_vec) * np.linalg.norm(vector))
            hamming = np.sum(ref_vec == vector) / len(vector)
            score = 0.6 * cosine + 0.4 * hamming
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldApp(root)
    root.mainloop()