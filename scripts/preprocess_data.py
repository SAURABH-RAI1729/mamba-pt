"""
Data preprocessing for TrackML dataset
"""
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import List
import warnings
warnings.filterwarnings('ignore')


def read_event(event_path: Path):
    """Read a single event from TrackML format"""
    event_prefix = event_path.stem.replace('-hits', '')  # Remove -hits suffix
    
    # Read CSV files
    hits = pd.read_csv(event_path.parent / f"{event_prefix}-hits.csv")
    truth = pd.read_csv(event_path.parent / f"{event_prefix}-truth.csv")
    particles = pd.read_csv(event_path.parent / f"{event_prefix}-particles.csv")
    
    # Merge dataframes
    hits = hits.merge(truth, on='hit_id')
    
    # Filter hits from pixel detector (volumes 7, 8, 9)
    hits = hits[hits['volume_id'].isin([7, 8, 9])]
    
    # Group by particle to form tracks
    tracks = []
    momenta = []
    
    for particle_id, group in hits.groupby('particle_id'):
        if particle_id == 0:  # Skip noise hits
            continue
            
        # Sort hits by radius (approximate track order)
        group = group.sort_values('hit_id')
        
        # Extract module IDs
        modules = group[['volume_id', 'layer_id', 'module_id']].values
        
        # Get particle momentum
        particle_data = particles[particles['particle_id'] == particle_id].iloc[0]
        momentum = np.sqrt(
            particle_data['px']**2 + 
            particle_data['py']**2 + 
            particle_data['pz']**2
        )
        
        tracks.append(modules)
        momenta.append(momentum)
    
    return tracks, momenta


def process_event_file(args):
    """Process a single event file"""
    event_path, output_dir = args
    
    try:
        tracks, momenta = read_event(event_path)
        
        # Save as temporary file
        event_id = event_path.stem.split('-')[0]
        output_path = output_dir / f"{event_id}_temp.h5"
        
        with h5py.File(output_path, 'w') as f:
            event_group = f.create_group(event_id)
            
            # Pad tracks to same length for batching
            max_length = max(len(track) for track in tracks)
            padded_tracks = np.zeros((len(tracks), max_length, 3), dtype=np.int32)
            
            for i, track in enumerate(tracks):
                padded_tracks[i, :len(track)] = track
            
            event_group.create_dataset('tracks', data=padded_tracks)
            event_group.create_dataset('momenta', data=momenta)
            
        return output_path
    except Exception as e:
        print(f"Error processing {event_path}: {e}")
        return None


def merge_temp_files(temp_files: List[Path], output_path: Path, split_name: str):
    """Merge temporary files into final dataset"""
    print(f"Merging {len(temp_files)} files for {split_name} split...")
    
    with h5py.File(output_path, 'w') as output_file:
        for temp_file in tqdm(temp_files, desc=f"Merging {split_name}"):
            if temp_file is None:
                continue
                
            with h5py.File(temp_file, 'r') as input_file:
                for event_id in input_file.keys():
                    input_file.copy(event_id, output_file)
            
            # Remove temporary file
            temp_file.unlink()


def preprocess_trackml_data(
    data_dir: Path,
    output_dir: Path,
    num_workers: int = 8,
    train_split: float = 0.8,
    val_split: float = 0.1,
):
    """
    Preprocess TrackML dataset
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all event files
    event_files = []
    for i in range(1, 6):  # train_1 through train_5
        train_dir = data_dir / f"train_{i}"
        if train_dir.exists():
            event_files.extend(list(train_dir.glob("event*-hits.csv")))
    
    print(f"Found {len(event_files)} event files")
    
    # Shuffle and split
    np.random.shuffle(event_files)
    
    n_train = int(len(event_files) * train_split)
    n_val = int(len(event_files) * val_split)
    
    train_files = event_files[:n_train]
    val_files = event_files[n_train:n_train + n_val]
    test_files = event_files[n_train + n_val:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
    }
    
    # Process each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} files)...")
        
        temp_dir = output_dir / f"{split_name}_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            args = [(f, temp_dir) for f in files]
            temp_files = list(tqdm(
                executor.map(process_event_file, args),
                total=len(files),
                desc=f"Processing {split_name}"
            ))
        
        # Filter out None values (failed processing)
        temp_files = [f for f in temp_files if f is not None]
        
        # Merge temporary files
        output_path = output_dir / f"{split_name}_tracks.h5"
        merge_temp_files(temp_files, output_path, split_name)
        
        # Clean up temp directory
        temp_dir.rmdir()
        
        print(f"Saved {split_name} split to {output_path}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TrackML dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw TrackML data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save preprocessed data")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    args = parser.parse_args()
    
    preprocess_trackml_data(
        Path(args.data_dir),
        Path(args.output_dir),
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
    )
