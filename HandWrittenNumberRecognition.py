import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
from HopfieldNetwork import HopfieldNetwork
from scipy.ndimage import center_of_mass, shift
from sklearn.preprocessing import StandardScaler

GRID_SIZE = 28   # dimensiunea grilei (28x28)
CELL_SIZE = 15  # dimensiunea unei celule

def center_image(data):
    """Center the digit in the binary image array (0 and 1)."""
    if np.sum(data) == 0:  # Empty image
        return data
    mass = center_of_mass(data)
    shift_y = data.shape[0] // 2 - mass[0]
    shift_x = data.shape[1] // 2 - mass[1]
    return shift(data, shift=(shift_y, shift_x), cval=0, mode='constant')

def enhance_pattern_contrast(binary_pattern):
    """Enhance pattern to make it more distinguishable"""
    # Apply morphological operations to strengthen the pattern
    from scipy.ndimage import binary_dilation, binary_erosion
    
    # Slightly dilate to fill gaps
    enhanced = binary_dilation(binary_pattern, iterations=1)
    
    # Then erode to maintain original thickness
    enhanced = binary_erosion(enhanced, iterations=1)
    
    return enhanced

def balance_pattern(pattern_2d):
    """Balance the pattern to have roughly equal +1 and -1 values"""
    # Count current black pixels
    black_pixels = np.sum(pattern_2d == 1)
    total_pixels = pattern_2d.size
    
    # Target: 25-40% black pixels for good discrimination
    target_ratio = 0.30
    target_black = int(total_pixels * target_ratio)
    
    if black_pixels < target_black * 0.5:  # Too sparse
        # More aggressive thresholding to get more black pixels
        return pattern_2d  # Keep as is for now
    elif black_pixels > target_black * 2:  # Too dense
        # Thin the pattern by keeping only strongest pixels
        # This is a simplified approach - you might want more sophisticated thinning
        return pattern_2d
    
    return pattern_2d

class HopfieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Improved Hopfield Digit Recognition")
        self.root.configure(bg="#f2f2f2")
        self.patterns = []       # lista cu cifrele antrenate (vectori + etichete)
        self.grid_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.hopfield = HopfieldNetwork(GRID_SIZE * GRID_SIZE)
        self.build_ui()          # construieste interfata grafica
        self.clear_grid()        

    def build_ui(self):
        # Main canvas for drawing
        self.canvas = tk.Canvas(self.root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=7, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)  # Single click painting

        # Control buttons
        tk.Button(self.root, text="Load Train Set", command=self.load_train_set, bg="#64b5f6", fg="white").grid(row=0, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Clear", command=self.clear_grid, bg="#e57373", fg="white").grid(row=1, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Recognize", command=self.recognize_digit, bg="#81c784", fg="white").grid(row=2, column=1, sticky="ew", padx=10)
        tk.Button(self.root, text="Save Drawing", command=self.save_drawing, bg="#ffd54f", fg="black").grid(row=3, column=1, sticky="ew", padx=10)

        # Results display
        self.result_label = tk.Label(self.root, text="Predicted: None", font=("Helvetica", 16, "bold"), bg="#f2f2f2")
        self.result_label.grid(row=5, column=1, pady=10)

    def clear_grid(self):
        self.grid_data = np.full((GRID_SIZE, GRID_SIZE), -1)  # initializare: toate celulele = -1 (alb)
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.create_rectangle(
                    j*CELL_SIZE, i*CELL_SIZE, (j+1)*CELL_SIZE, (i+1)*CELL_SIZE,
                    fill="white", outline="lightgray", tags=f"cell_{i}_{j}"
                )

    def paint(self, event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        # Paint a 2x2 area for better visibility
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
        loaded_vectors = []
        
        for file in os.listdir(dir_path):
            if file.lower().endswith(".png"):
                path = os.path.join(dir_path, file)
                vector = self.image_to_vector(path)
                if vector is not None:
                    label = file.split('_')[0]
                    self.patterns.append((vector, label))
                    loaded_vectors.append(vector)
                    print(f"Loaded {label}: sparsity = {np.sum(vector == 1) / len(vector) * 100:.1f}%")

        if not self.patterns:
            messagebox.showwarning("Warning", "No valid PNG files found!")
            return

        # Train the network
        print(f"\nTraining Hopfield network with {len(loaded_vectors)} patterns...")
        self.hopfield.train(loaded_vectors)
        
        # Verify training
        print("\nVerifying training (should be perfect recall):")
        for i, (vec, label) in enumerate(self.patterns):
            recalled = self.hopfield.recall(vec.copy(), max_steps=10)
            errors = np.sum(vec != recalled)
            print(f"Pattern {label}: {errors} errors in recall")

        self.info_label.config(text=f"Patterns loaded: {len(self.patterns)}")
        messagebox.showinfo("Load", f"Loaded {len(self.patterns)} images")

    def save_drawing(self):
        label = simpledialog.askstring("Save Digit", "Enter the digit label (e.g., 3, 3_drawn1):")
        if label is None or not label or not any(c.isdigit() for c in label):
            messagebox.showerror("Invalid Input", "Please enter a label that includes a digit (e.g., 3, 5_drawn).")
            return

        # Create TrainingSet directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), "TrainingSet")
        os.makedirs(save_dir, exist_ok=True)

        # Build a unique filename
        base_filename = f"{label}.png"
        filepath = os.path.join(save_dir, base_filename)
        count = 1
        while os.path.exists(filepath):
            filepath = os.path.join(save_dir, f"{label}_drawn{count}.png")
            count += 1

        # Convert -1, 1 grid to image (255=white, 0=black)
        image_array = np.where(self.grid_data == 1, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_array, mode='L')
        img.save(filepath)

        messagebox.showinfo("Saved", f"Digit saved as {os.path.basename(filepath)} in 'TrainingSet'")
    
    def image_to_vector(self, path):
        try:
            # Load and preprocess image
            img = Image.open(path).convert("L").resize((GRID_SIZE, GRID_SIZE))
            data = np.asarray(img)
            
            # Use a more aggressive threshold to get more black pixels
            # For digit images, we want to capture the stroke structure
            threshold = np.percentile(data, 75)  # Take 25% darkest pixels as foreground
            
            # Create binary image
            binary = np.where(data < threshold, 1, 0)
            
            # Only apply minimal enhancement - too much processing makes patterns similar
            # Skip morphological operations that reduce distinctiveness
            
            # Center the pattern
            binary_centered = center_image(binary)
            
            # Force better sparsity for Hopfield networks
            current_sparsity = np.sum(binary_centered == 1) / binary_centered.size
            
            # If too sparse, make threshold more aggressive
            if current_sparsity < 0.20:  # Less than 20%
                threshold = np.percentile(data, 60)  # Take 40% darkest pixels
                binary_centered = np.where(data < threshold, 1, 0)
                binary_centered = center_image(binary_centered)
            
            # If still too sparse, dilate slightly
            current_sparsity = np.sum(binary_centered == 1) / binary_centered.size
            if current_sparsity < 0.15:  # Still too sparse
                from scipy.ndimage import binary_dilation
                binary_centered = binary_dilation(binary_centered, iterations=1)
            
            # Convert to bipolar (-1, 1) - this is crucial for Hopfield networks
            bipolar = np.where(binary_centered == 1, 1, -1).flatten()
            
            return bipolar
        
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None
        
    def recognize_digit(self):
        if not self.patterns:
            messagebox.showwarning("Warning", "Please load training patterns first!")
            return
            
        # Get current drawing and preprocess it the same way as training data
        current_matrix = self.grid_data.copy()
        
        # Convert from -1/1 to 0/1 for processing
        binary = np.where(current_matrix == 1, 1, 0)
        
        # Apply same preprocessing as training - but keep it minimal
        binary = center_image(binary)
        
        # Check sparsity and adjust if needed
        current_sparsity = np.sum(binary == 1) / binary.size
        if current_sparsity < 0.15:  # Too sparse
            from scipy.ndimage import binary_dilation
            binary = binary_dilation(binary, iterations=1)
            
        # Convert to bipolar
        current_vector = np.where(binary == 1, 1, -1).flatten()
        
        print(f"\n=== Recognition Process ===")
        final_sparsity = np.sum(binary == 1) / binary.size * 100
        print(f"Input sparsity: {final_sparsity:.1f}% black pixels")
        
        # Show input for debugging
        input_img_array = (binary * 255).astype(np.uint8)
        input_img = Image.fromarray(input_img_array, mode='L')
        print("Showing input image...")
        input_img.show()
        
        # Calculate initial energy
        initial_energy = self.hopfield.energy(current_vector)
        print(f"Initial energy: {initial_energy:.2f}")
        
        # Recall with monitoring
        print("Starting recall process...")
        recalled_vector = self.hopfield.recall(current_vector.copy(), max_steps=10)
        final_energy = self.hopfield.energy(recalled_vector)
        print(f"Final energy: {final_energy:.2f}")
        print(f"Energy improvement: {initial_energy - final_energy:.2f}")
        
        # Check if recall actually changed anything
        changes = np.sum(current_vector != recalled_vector)
        print(f"Pixels changed during recall: {changes}")
        
        # Find best match using improved scoring
        best_label, best_score = self.find_best_match(recalled_vector)
        
        self.result_label.config(text=f"Predicted: {best_label}")
        print(f"\n=== Result: {best_label} (score: {best_score:.3f}) ===")
        
        # Also compare directly with input (before recall)
        print("\nDirect comparison with input:")
        best_direct_label, best_direct_score = self.find_best_match(current_vector)
        print(f"Direct match: {best_direct_label} (score: {best_direct_score:.3f})")

    def find_best_match(self, recalled_vector):
        """Find the best matching pattern using multiple similarity metrics"""
        best_label, best_score = None, -np.inf
        
        print(f"\nPattern matching:")
        for vec, label in self.patterns:
            # Normalized dot product (cosine similarity)
            norm_recalled = np.linalg.norm(recalled_vector)
            norm_pattern = np.linalg.norm(vec)
            
            if norm_recalled > 0 and norm_pattern > 0:
                cosine_sim = np.dot(vec, recalled_vector) / (norm_pattern * norm_recalled)
            else:
                cosine_sim = 0
            
            # Hamming similarity (percentage of matching bits)
            hamming_sim = np.sum(vec == recalled_vector) / len(vec)
            
            # Raw dot product
            dot_product = np.dot(vec, recalled_vector)
            
            # Combined score (weighted average)
            combined_score = 0.6 * cosine_sim + 0.4 * hamming_sim
            
            print(f"  {label}: cosine={cosine_sim:.3f}, hamming={hamming_sim:.3f}, combined={combined_score:.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_label = label
        
        return best_label, best_score

if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldApp(root)
    root.mainloop()