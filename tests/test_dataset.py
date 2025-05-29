"""
Test suite for dataset and data processing
"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.data.trackml_dataset import TrackMLDataset, TrackMLDataModule
from src.data.preprocessing import (
    calculate_radius,
    calculate_phi,
    validate_track,
    filter_tracks_by_quality,
    compute_dataset_statistics
)
from src.models.tokenizer import DetectorTokenizer


class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    def test_calculate_radius(self):
        """Test radius calculation"""
        assert np.isclose(calculate_radius(3, 4), 5)
        assert np.isclose(calculate_radius(0, 5), 5)
        assert np.isclose(calculate_radius(5, 0), 5)
    
    def test_calculate_phi(self):
        """Test phi angle calculation"""
        assert np.isclose(calculate_phi(1, 0), 0)
        assert np.isclose(calculate_phi(0, 1), np.pi/2)
        assert np.isclose(calculate_phi(-1, 0), np.pi)
    
    def test_validate_track(self):
        """Test track validation"""
        # Valid track
        track = np.array([[7, 2, 100], [7, 4, 200], [8, 6, 300]])
        assert validate_track(track) == True
        
        # Invalid track (duplicate modules)
        track = np.array([[7, 2, 100], [7, 2, 100], [8, 6, 300]])
        assert validate_track(track) == False
        
        # Invalid track (large layer gap)
        track = np.array([[7, 2, 100], [7, 8, 200], [8, 14, 300]])
        assert validate_track(track) == False
    
    def test_filter_tracks_by_quality(self):
        """Test track quality filtering"""
        tracks = [
            np.array([[7, 2, 100], [7, 4, 200]]),
            np.array([[8, 2, 100], [8, 4, 200]]),
        ]
        momenta = [
            np.array([0.3, 0.4, 1.0, 0.5, 1.1]),  # pt = 0.5 GeV
            np.array([0.1, 0.1, 1.0, 0.14, 1.01]),  # pt = 0.14 GeV (below cut)
        ]
        
        filtered_tracks, filtered_momenta = filter_tracks_by_quality(
            tracks, momenta, min_pt=0.3
        )
        
        assert len(filtered_tracks) == 1
        assert len(filtered_momenta) == 1
        assert filtered_momenta[0][3] >= 0.3  # pt >= min_pt
    
    def test_compute_dataset_statistics(self):
        """Test dataset statistics computation"""
        tracks = [
            np.array([[7, 2, 100], [7, 4, 200]]),
            np.array([[8, 2, 100], [8, 4, 200], [8, 6, 300]]),
        ]
        momenta = [
            np.array([0.3, 0.4, 1.0, 0.5, 1.1]),
            np.array([0.6, 0.8, 1.5, 1.0, 1.8]),
        ]
        
        stats = compute_dataset_statistics(tracks, momenta)
        
        assert stats['n_tracks'] == 2
        assert stats['mean_track_length'] == 2.5
        assert stats['mean_pt'] == 0.75
        assert stats['track_length_distribution'][2] == 1
        assert stats['track_length_distribution'][3] == 1


class TestTrackMLDataset:
    """Test TrackML dataset implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = DetectorTokenizer(vocab_size=1000)
        
        # Build vocabulary
        modules = [(7, i, j) for i in range(10) for j in range(10)]
        self.tokenizer.build_vocab(modules)
        
        # Create dummy data
        self.create_dummy_data()
        
    def create_dummy_data(self):
        """Create dummy HDF5 data for testing"""
        import h5py
        
        # Create dummy tracks
        tracks = np.array([
            [[7, 2, 10], [7, 4, 20], [7, 6, 30], [0, 0, 0]],  # Padded
            [[8, 2, 10], [8, 4, 20], [8, 6, 30], [8, 8, 40]],
        ])
        
        momenta = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ])
        
        # Save as HDF5
        h5_path = Path(self.temp_dir) / "train_tracks.h5"
        with h5py.File(h5_path, 'w') as f:
            event_group = f.create_group('event1')
            event_group.create_dataset('tracks', data=tracks)
            event_group.create_dataset('momenta', data=momenta)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        config = {
            'data': {
                'track_length_range': [3, 8],
                'volumes': [7, 8, 9],
                'max_tracks_per_event': 1000,
            }
        }
        
        dataset = TrackMLDataset(
            Path(self.temp_dir),
            self.tokenizer,
            config,
            split="train"
        )
        
        assert len(dataset) > 0
        
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        config = {
            'data': {
                'track_length_range': [3, 8],
                'volumes': [7, 8, 9],
                'max_tracks_per_event': 1000,
            }
        }
        
        dataset = TrackMLDataset(
            Path(self.temp_dir),
            self.tokenizer,
            config,
            split="train"
        )
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'momentum' in item
        assert 'track_idx' in item
        
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['momentum'], torch.Tensor)
    
    def test_data_module(self):
        """Test data module"""
        config = {
            'data': {
                'data_dir': str(self.temp_dir),
                'track_length_range': [3, 8],
                'volumes': [7, 8, 9],
                'max_tracks_per_event': 1000,
                'batch_size': 2,
                'num_workers': 0,
            }
        }
        
        # Create dummy files for all splits
        for split in ['train', 'val', 'test']:
            h5_path = Path(self.temp_dir) / f"{split}_tracks.h5"
            with h5py.File(h5_path, 'w') as f:
                event_group = f.create_group('event1')
                event_group.create_dataset('tracks', data=np.random.rand(2, 4, 3))
                event_group.create_dataset('momenta', data=np.random.rand(2, 3))
        
        data_module = TrackMLDataModule(
            config['data']['data_dir'],
            self.tokenizer,
            config,
            batch_size=2,
            num_workers=0
        )
        
        data_module.setup()
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        assert len(train_loader) > 0
        
        batch = next(iter(train_loader))
        assert batch['input_ids'].shape[0] == 2  # batch size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
