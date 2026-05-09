import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
DATASET_DIR = r"..\dataset"
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

def load_dataset_metadata(dataset_dir: str) -> pd.DataFrame:
    """
    Scans the dataset directory and compiles image paths and labels into a DataFrame.

    Parameters:
    -----------
    dataset_dir : str
        The relative or absolute path to the root dataset directory containing class folders.

    Returns:
    --------
    pd.DataFrame
        A dataframe containing 'image_path', 'class_name', and integer 'label' for all discovered images.

    Raises:
    -------
    FileNotFoundError
        If the specified dataset_dir does not exist or is inaccessible.

    Notes:
    ------
    - Expected input `dataset_dir` is a valid string path.
    - Output DataFrame dtypes: 'image_path' (string), 'class_name' (string), 'label' (int).
    - Folders inside the root directory are sorted alphabetically to ensure consistent integer label mapping.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    data = []
    class_names = sorted(os.listdir(dataset_dir))
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir): 
            continue
            
        images = os.listdir(class_dir)
        for img in images:
            img_path = os.path.join(class_dir, img)
            data.append({"image_path": img_path, "class_name": class_name, "label": label})
            
    return pd.DataFrame(data)


def plot_class_distribution(class_counts: pd.Series, title: str, output_filename: str) -> None:
    """
    Generates and saves a bar chart visualizing the number of images per class.

    Parameters:
    -----------
    class_counts : pd.Series
        A pandas Series where the index represents class names and values represent image counts.
    title : str
        The title to display at the top of the chart.
    output_filename : str
        The filename (including extension) used to save the generated plot.

    Returns:
    --------
    None

    Raises:
    -------
    TypeError
        If class_counts is not a pandas Series or if title/output_filename are not strings.
    ValueError
        If class_counts is empty.

    Notes:
    ------
    - Expected `class_counts` dtype is numeric (integers).
    - The plot is saved directly to the disk without pausing script execution.
    """
    if not isinstance(class_counts, pd.Series):
        raise TypeError("class_counts must be a pandas Series.")
    if class_counts.empty:
        raise ValueError("Cannot plot an empty class distribution.")

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.title(title)
    plt.xlabel("Animal Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close() # Close figure to free memory


def stratified_split(df: pd.DataFrame, split_ratios: tuple) -> tuple:
    """
    Performs a stratified split of the dataset into training, validation, and test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The downsampled dataframe containing the dataset metadata.
    split_ratios : tuple
        A tuple of three floats representing the ratio for (train, validation, test).

    Returns:
    --------
    tuple
        A tuple containing three pandas DataFrames: (train_df, val_df, test_df).

    Raises:
    -------
    ValueError
        If the sum of split_ratios does not equal 1.0, or if df is missing required columns.
    TypeError
        If split_ratios is not a tuple.

    Notes:
    ------
    - Expected `split_ratios` elements must be floats between 0.0 and 1.0.
    - Requires the input DataFrame to contain a 'class_name' column for grouping.
    - Rows are shuffled internally using a fixed random_state for reproducibility.
    """
    if sum(split_ratios) != 1.0:
        raise ValueError("The sum of split_ratios must equal exactly 1.0.")
    if 'class_name' not in df.columns:
        raise ValueError("DataFrame must contain a 'class_name' column for stratified splitting.")

    train_data, val_data, test_data = [], [], []

    for class_name, group in df.groupby('class_name'):
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_total = len(group)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_data.append(group.iloc[:n_train])
        val_data.append(group.iloc[n_train:n_train+n_val])
        test_data.append(group.iloc[n_train+n_val:])

    train_df = pd.concat(train_data).sample(frac=1, random_state=42) 
    val_df = pd.concat(val_data).sample(frac=1, random_state=42)
    test_df = pd.concat(test_data).sample(frac=1, random_state=42)
    
    return train_df, val_df, test_df


def main():
    """Main execution pipeline for dataset preparation."""
    print("Starting dataset preparation pipeline...")

    # 1. Load Data
    df = load_dataset_metadata(DATASET_DIR)
    
    # 2. Plot Initial Distribution
    initial_counts = df['class_name'].value_counts()
    plot_class_distribution(initial_counts, "Class Distribution Before Downsampling", "class_distribution_before.png")

    # 3. Downsample Logic
    min_class_count = initial_counts.min()
    print(f"Downsampling all classes to match the smallest class: {min_class_count} images.")
    downsampled_df = df.groupby('class_name').sample(n=min_class_count, random_state=42).reset_index(drop=True)

    # 4. Plot Downsampled Distribution
    after_counts = downsampled_df['class_name'].value_counts()
    plot_class_distribution(after_counts, "Class Distribution After Downsampling", "class_distribution_after.png")

    # 5. Split Dataset
    train_df, val_df, test_df = stratified_split(downsampled_df, SPLIT_RATIOS)

    # 6. Save Annotations
    train_df.to_csv("train_annotations.csv", index=False)
    val_df.to_csv("val_annotations.csv", index=False)
    test_df.to_csv("test_annotations.csv", index=False)

    print(f"Pipeline finished successfully!")
    print(f"Total Sets -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()