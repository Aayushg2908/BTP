"""
Download and organize Google Speech Commands Dataset v0.02
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_NAME = "speech_commands_v0.02"


def download_file(url, dest_path):
    """Download a file with progress bar."""
    def reporthook(count, block_size, total_size):
        pbar.update(block_size)
    
    pbar = tqdm(total=None, unit='B', unit_scale=True, desc="Downloading")
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    pbar.close()


def extract_tar(tar_path, extract_to):
    """Extract tar.gz file."""
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Use filter='data' for Python 3.12+ to handle deprecation warning
        # For older versions, use extractall without filter
        try:
            tar.extractall(extract_to, filter='data')
        except TypeError:
            # Fallback for Python < 3.12
            tar.extractall(extract_to)
    print("Extraction complete.")


def organize_dataset(data_dir, dataset_dir):
    """
    Organize dataset into train/validation/test splits.
    Uses the validation_list.txt and testing_list.txt if available,
    otherwise uses a simple split.
    Preserves label subdirectories (backward/, bed/, etc.)
    """
    dataset_path = Path(data_dir) / dataset_dir
    
    # Ensure dataset directory exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    
    # Check if validation and test lists exist
    validation_list_path = dataset_path / "validation_list.txt"
    testing_list_path = dataset_path / "testing_list.txt"
    
    # Create split directories (with parents=True to ensure parent exists)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validation"
    test_dir = dataset_path / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files (including those in subdirectories)
    # Exclude files in train/validation/test directories and _background_noise_
    audio_files = []
    for audio_file in dataset_path.rglob("*.wav"):
        # Skip if already in organized splits or background noise
        parts = audio_file.parts
        if 'train' not in parts and 'validation' not in parts and 'test' not in parts:
            if '_background_noise_' not in parts:
                audio_files.append(audio_file)
    
    if validation_list_path.exists() and testing_list_path.exists():
        # Use provided splits
        with open(validation_list_path, 'r') as f:
            val_files = set(f.read().strip().split('\n'))
        with open(testing_list_path, 'r') as f:
            test_files = set(f.read().strip().split('\n'))
        
        for audio_file in tqdm(audio_files, desc="Organizing dataset"):
            # Get relative path from dataset root (e.g., "backward/00176480_nohash_0.wav")
            rel_path = audio_file.relative_to(dataset_path)
            rel_path_str = str(rel_path).replace('\\', '/')
            
            # Determine split
            if rel_path_str in test_files:
                split_dir = test_dir
            elif rel_path_str in val_files:
                split_dir = val_dir
            else:
                split_dir = train_dir
            
            # Preserve label subdirectory structure
            # rel_path has format: "label/filename.wav"
            label_dir = split_dir / audio_file.parent.name
            label_dir.mkdir(parents=True, exist_ok=True)
            
            dest = label_dir / audio_file.name
            shutil.copy2(audio_file, dest)
    else:
        # Simple 80/10/10 split per label
        import random
        random.seed(42)
        
        # Group files by label (subdirectory)
        files_by_label = {}
        for audio_file in audio_files:
            label = audio_file.parent.name
            if label not in files_by_label:
                files_by_label[label] = []
            files_by_label[label].append(audio_file)
        
        # Split each label's files
        for label, files in tqdm(files_by_label.items(), desc="Organizing dataset"):
            random.shuffle(files)
            n_test = len(files) // 10
            n_val = len(files) // 10
            
            for i, audio_file in enumerate(files):
                if i < n_test:
                    split_dir = test_dir
                elif i < n_test + n_val:
                    split_dir = val_dir
                else:
                    split_dir = train_dir
                
                # Preserve label subdirectory
                label_dir = split_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                dest = label_dir / audio_file.name
                shutil.copy2(audio_file, dest)
    
    print(f"Dataset organized:")
    print(f"  Train: {len(list(train_dir.rglob('*.wav')))} files")
    print(f"  Validation: {len(list(val_dir.rglob('*.wav')))} files")
    print(f"  Test: {len(list(test_dir.rglob('*.wav')))} files")


def main():
    """Main download and setup function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    tar_path = data_dir / "speech_commands_v0.02.tar.gz"
    dataset_path = data_dir / DATASET_NAME
    
    # Check if dataset files are already in data_dir (from previous extraction)
    # This happens when tar extracts directly to data_dir instead of subdirectory
    files_in_data_dir = list(data_dir.glob("*.wav")) + list(data_dir.glob("validation_list.txt"))
    if files_in_data_dir and not dataset_path.exists():
        print(f"Found dataset files in {data_dir}, moving to {dataset_path}...")
        dataset_path.mkdir(parents=True, exist_ok=True)
        # Move all files and directories from data_dir to dataset_path
        # Collect items first to avoid iteration issues
        items_to_move = [item for item in data_dir.iterdir() 
                        if item.name != DATASET_NAME and item.name != tar_path.name]
        for item in items_to_move:
            dest = dataset_path / item.name
            if dest.exists():
                print(f"Warning: {dest} already exists, skipping {item.name}")
                continue
            if item.is_dir():
                shutil.move(str(item), str(dest))
            else:
                shutil.move(str(item), str(dest))
        print("Files moved successfully.")
    
    # Check if dataset already exists in correct location
    if dataset_path.exists():
        # Check for audio files (either directly or in subdirectories)
        audio_files = list(dataset_path.rglob("*.wav"))
        if len(audio_files) > 0:
            print(f"Dataset already exists at {dataset_path} with {len(audio_files)} audio files")
            # Check if already organized
            if (dataset_path / "train").exists():
                print("Dataset already organized into train/validation/test splits")
                return
        else:
            print(f"Dataset directory exists but no audio files found. Re-extracting...")
            # Remove empty directory and re-extract
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
    
    # Download dataset
    if not tar_path.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        download_file(DATASET_URL, tar_path)
    else:
        print(f"Tar file already exists at {tar_path}")
    
    # Extract dataset
    # The tar file extracts files directly to the extraction directory
    if not dataset_path.exists():
        # Create dataset directory first
        dataset_path.mkdir(parents=True, exist_ok=True)
        # Extract directly to dataset_path
        extract_tar(tar_path, dataset_path)
    
    # Verify dataset was extracted (check for audio files)
    audio_files = list(dataset_path.rglob("*.wav"))
    if len(audio_files) == 0:
        raise FileNotFoundError(f"Dataset extraction failed. No audio files found in {dataset_path}")
    else:
        print(f"Found {len(audio_files)} audio files in dataset")
    
    # Organize into splits
    if not (dataset_path / "train").exists():
        organize_dataset(data_dir, DATASET_NAME)
    else:
        print("Dataset already organized into train/validation/test splits")


if __name__ == "__main__":
    main()

