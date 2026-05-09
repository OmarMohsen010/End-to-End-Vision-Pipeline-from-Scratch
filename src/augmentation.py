import numpy as np
import matplotlib.pyplot as plt
import random
from minicv.transforms import rotate, translate
from minicv.filtering import gaussian_filter
from minicv.utils import clip_pixels


def save_augmentation_panel(X: np.ndarray, output_filename: str = "augmentation_panel.png") -> None:
    """
    Generates and saves a 2x3 grid visualizing 5 augmentations applied to a single base image.

    Parameters:
    -----------
    X : np.ndarray
        The preprocessed training dataset array containing images.
    output_filename : str, optional
        The path/filename where the panel will be saved (default is 'augmentation_panel.png').

    Returns:
    --------
    None

    Raises:
    -------
    TypeError
        If the input X is not a NumPy array.
    ValueError
        If the input array does not have 4 dimensions (N, H, W, C) or is empty.

    Notes:
    ------
    - Expected input ranges/dtypes: X should be normalized float64 data in the [0, 1] range.
    - Requires minicv transformations: rotate, translate, gaussian_filter, clip_pixels.
    """
    # --- Input Validation (Requirement 9.2) ---
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected X to be a NumPy array, got {type(X)}.")
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, H, W, C), but got shape {X.shape}.")
    if len(X) == 0:
        raise ValueError("Input array X is empty.")

    # --- Extract Base Image ---
    random_indices = random.randrange(len(X))
    X_normal = X[random_indices]

    # --- Apply Transformations ---
    X_rotated = rotate(X_normal, angle=17)
    X_translated = translate(X_normal, tx=5, ty=10)
    X_gaussian = gaussian_filter(X_normal, kernel_size=5, sigma=1.0)
    
    # Brightness adjustment with proper clipping bounds
    X_bright = X_normal.astype(np.float32) + 30.0
    X_bright = clip_pixels(X_bright, low=0.0, high=255.0)
    X_bright = X_bright.astype(np.uint8) 
    
    # Horizontal flip (NumPy vectorization)
    X_fliped = X_normal[:, ::-1, :]

    # --- Prepare Plotting Data ---
    images = [X_normal, X_rotated, X_translated, X_gaussian, X_bright, X_fliped]
    titles = ["Original", "Rotated", "Translated", 
              "Gaussian Blur", "Brightness", "Horizontal Flip"]

    # --- Plotting Engine ---
    # Create the 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # Flatten the 2x3 axes matrix into a 1D list of 6 items for easy looping
    axs_flat = axs.ravel()

    for idx, ax in enumerate(axs_flat):
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.axis('off') # Hide the tick marks for a cleaner look

    # Adjust layout to prevent overlapping text
    plt.tight_layout()
    
    # Save the figure with high resolution for the report
    plt.savefig(output_filename, dpi=300)
    plt.close() # Free up memory
    
    print(f"Visualization panel saved successfully as '{output_filename}'.")



def run_targeted_augmentation(x_path: str, y_path: str, num_to_augment: int = 500):
    """
    Randomly selects a subset of training images, applies one of five random augmentations 
    to each, and appends them to the dataset to increase feature variance without doubling RAM usage.

    Parameters:
    -----------
    x_path : str
        The file path to the saved NumPy array of preprocessed training images (e.g., 'X_train.npy').
    y_path : str
        The file path to the saved NumPy array of training labels (e.g., 'Y_train.npy').
    num_to_augment : int, optional
        The exact number of images to randomly sample and augment (default is 500).

    Returns:
    --------
    None
        The function saves the final concatenated arrays directly to disk as 'X_train_augmented.npy' 
        and 'Y_train_augmented.npy' to preserve memory.

    Raises:
    -------
    FileNotFoundError
        If the specified .npy files cannot be located at the provided paths.
    ValueError
        If `num_to_augment` is greater than the total number of available images in the loaded dataset.

    Notes:
    ------
    - Expected input ranges/dtypes: Images in the loaded X array must be normalized float data in the [0.0, 1.0] range.
    - The brightness augmentation internally handles the temporary conversion to float32/uint8 
      required for the 0-255 scale math, ensuring no floating-point clipping errors occur.
    """
    

    print("Loading original training data...")
    X = np.load(x_path)
    Y = np.load(y_path)
    
    total_images = len(X)
    
    # We will convert the numpy arrays to lists to easily append the new images
    X_list = list(X)
    Y_list = list(Y)
    
    print(f"Selecting {num_to_augment} random images to augment...")
    # Randomly pick indices from the dataset
    augment_indices = random.sample(range(total_images), num_to_augment)
    
    for i, idx in enumerate(augment_indices):
        img = X[idx]
        label = Y[idx]
        
        # Pick a random transform (0 to 4)
        choice = random.randint(0, 4)
        
        if choice == 0:
            aug_img = rotate(img, angle=15)
        elif choice == 1:
            aug_img = translate(img, tx=10, ty=10)
        elif choice == 2:
            aug_img = gaussian_filter(img, kernel_size=3, sigma=1.0)
        elif choice == 3:
            aug_img = img.astype(np.float32) + 40.0
            aug_img = clip_pixels(aug_img, low=0.0, high=255.0)
            aug_img = aug_img.astype(np.uint8)
        elif choice == 4:
            aug_img = img[:, ::-1, :]
            
        # Append only the new augmented image to our lists
        X_list.append(aug_img)
        Y_list.append(label)
        
        if i > 0 and i % 100 == 0:
            print(f"Augmented {i} / {num_to_augment} images...")

    print("Stacking lists back into NumPy arrays...")
    X_final = np.array(X_list)
    Y_final = np.array(Y_list)
    
    print(f"Done, Original Shape: {X.shape} | Final Shape: {X_final.shape}")

    print("Shuffling the dataset to ensure random distribution...")
    
    # Generate a randomly ordered list of indices (from 0 to total_length)
    shuffle_indices = np.random.permutation(len(X_final))
    
    # Apply that exact same random order to BOTH arrays
    X_final = X_final[shuffle_indices]
    Y_final = Y_final[shuffle_indices]

    
    # Save the new dataset
    np.save("X_train_augmented.npy", X_final)
    np.save("Y_train_augmented.npy", Y_final)
    print("Saved 'X_train_augmented.npy' and 'Y_train_augmented.npy'")

if __name__ == "__main__":
    X = np.load('X_train.npy')
    save_augmentation_panel(X,"augmentation_panel2.png")
