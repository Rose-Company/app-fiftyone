import fiftyone as fo
import os
from pathlib import Path

def load_images_to_fiftyone(image_dir, dataset_name="my_images"):
    """
    Load images from a directory into FiftyOne.
    
    Args:
        image_dir (str): Path to the directory containing images
        dataset_name (str): Name of the dataset to create/use in FiftyOne
    """
    print(f"Loading images from: {image_dir}")
    
    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Create or load dataset
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        print(f"Loaded existing dataset: {dataset_name}")
    else:
        dataset = fo.Dataset(dataset_name)
        print(f"Created new dataset: {dataset_name}")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(image_dir).rglob(f"*{ext}")))
        image_files.extend(list(Path(image_dir).rglob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images")
    
    # Add images to dataset
    samples = []
    for image_path in image_files:
        # Create sample
        sample = fo.Sample(filepath=str(image_path))
        
        # Add metadata
        sample["filename"] = image_path.name
        sample["directory"] = str(image_path.parent)
        
        samples.append(sample)
    
    # Add samples to dataset
    dataset.add_samples(samples)
    
    print(f"Successfully added {len(samples)} images to dataset '{dataset_name}'")
    print("\nYou can now view your dataset in the FiftyOne app by running:")
    print("fiftyone app launch")
    
    return dataset

if __name__ == "__main__":
    # Example usage
    image_directory = "/app/data/dataset"  # Replace with your image directory
    dataset_name = "my_images_17/6"  # Replace with your desired dataset name
    
    dataset = load_images_to_fiftyone(image_directory, dataset_name) 