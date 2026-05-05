import os

# Assign the directory's absolute path
DATASET_DIR = r"..\dataset"

def rename_dataset_files(base_dir):
    # Loop through each animal class folder
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir,class_name)

        # Skip if it's not a directory
        if not os.path.isdir(class_dir):
            continue

        print(f"Renaming files in: {class_name}...")

        # Get all images in the folder and sort them
        images = sorted(os.listdir(class_dir))
        
        for idx,filename in enumerate(images):
            # Extract the original file extension (e.g jpg, .png)
            _ , extenstion = os.path.splitext(filename)
            # Create the new name: classname_001.jpg, classname_002.jpg, etc.
            # The :04d ensures it pads with zeros up to 4 digits.
            new_name = f"{class_name}_{idx + 1:04d}{extenstion}"

            old_path = os.path.join(class_dir,filename)
            new_path = os.path.join(class_dir,new_name)

            # Rename the file
            os.rename(old_path,new_path)

    print("Renaming complete!")

# Run the function
rename_dataset_files(DATASET_DIR)   
