from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from scipy.io import loadmat
import pandas as pd
import numpy as np
import warnings
import torch
import os
import pickle
import json
import random
from scipy import signal
from scipy.signal import butter, filtfilt

from perturbations.time_invariant import TimeInvariantPerturbationLayer

warnings.filterwarnings("ignore", category=UserWarning)

class CardiacArrhythmiaDataset(Dataset):
    """
    Cardiac Arrhythmia dataset for ECG classification.
    
    Based on the reference code approach:
    - Multiple trimming methods: max, min, center
    - Data augmentation: noise addition and time shifting
    - Flexible preprocessing options
    """
    
    # Label mapping based on REFERENCE.csv
    LABEL_MAPPING = {
        'N': 0,  # Normal sinus rhythm
        'A': 1,  # Atrial fibrillation  
        'O': 2,  # Alternative rhythm
        '~': 3   # Others/Noisy
    }
    
    def __init__(
        self,
        root: str,
        split: str,  # "train" | "val" | "test"
        sample_rate: int = 100,  # ECG sampling rate (Hz)
        trim_method: str = "center",  # "max", "min", "center"
        apply_preprocessing: bool = True,
        augment: bool = False,
        augment_factor: int = 10,  # Number of augmented samples per original
        # Time-invariant perturbation parameters
        time_invariant_augment: bool = False,
        time_invariant_prob: float = 0.5,
        perturbation_types: Optional[List[str]] = None,
        kernel_sizes: Optional[List[int]] = None,
        epsilon_range: Tuple[float, float] = (0.1, 0.5),
        force_reprocess: bool = False,
    ):
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.trim_method = trim_method
        self.apply_preprocessing = apply_preprocessing
        self.augment = augment
        self.augment_factor = augment_factor
        
        # Time-invariant perturbation parameters (only for training)
        self.time_invariant_augment = time_invariant_augment and split == "train"
        self.time_invariant_prob = time_invariant_prob
        self.perturbation_types = perturbation_types or ['shift', 'lowpass', 'echo', 'highpass', 'gaussian']
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.epsilon_range = epsilon_range
        
        # Set up paths
        self.ecg_dir = os.path.join(root, 'CardiacArrhythmia')
        self.reference_file = os.path.join(self.ecg_dir, 'REFERENCE.csv')
        
        # Cache file paths
        self.cache_dir = os.path.join(self.ecg_dir, 'cache')
        cache_name = f'ecg_data_sr{sample_rate}_trim{trim_method}_prep{int(apply_preprocessing)}_aug{int(augment)}.pkl'
        self.cache_file = os.path.join(self.cache_dir, cache_name)
        self.metadata_file = os.path.join(self.cache_dir, cache_name.replace('.pkl', '.json'))
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load reference labels
        self.reference_df = self._load_reference()
        
        # Create label mapping
        self.label_to_index = self.LABEL_MAPPING
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        
        # Load and preprocess data (from cache if available)
        self.ecg_segments, self.labels = self._load_and_preprocess_data(force_reprocess)
        
    def _load_reference(self) -> pd.DataFrame:
        """Load the REFERENCE.csv file with labels"""
        if not os.path.exists(self.reference_file):
            raise FileNotFoundError(f"Reference file not found: {self.reference_file}")
        
        # Read CSV with custom column names
        df = pd.read_csv(self.reference_file, header=None, names=['record_id', 'label'])
        return df
    
    def _load_and_preprocess_data(self, force_reprocess: bool = False) -> Tuple[List[torch.Tensor], List[int]]:
        """Load ECG data and create segments (from cache if available)"""
        # Check if cache exists and is valid
        if not force_reprocess and self._cache_exists():
            print(f"Loading preprocessed ECG data from cache: {self.cache_file}")
            return self._load_from_cache()
        
        print("Processing ECG data and creating cache...")
        
        # Process data normally
        ecg_segments, labels = self._process_raw_data()
        
        # Save to cache
        self._save_to_cache(ecg_segments, labels)
        
        return ecg_segments, labels
    
    def _cache_exists(self) -> bool:
        """Check if cache files exist and are valid"""
        if not os.path.exists(self.cache_file) or not os.path.exists(self.metadata_file):
            return False
        
        # Check if metadata is valid
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if parameters match
            if (metadata.get('sample_rate') != self.sample_rate or
                metadata.get('trim_method') != self.trim_method or
                metadata.get('apply_preprocessing') != self.apply_preprocessing or
                metadata.get('augment') != self.augment):
                return False
            
            # Check if cache file is not empty
            if os.path.getsize(self.cache_file) == 0:
                return False
                
            return True
        except:
            return False
    
    def _load_from_cache(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Load preprocessed data from cache"""
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Extract data for current split
            split_data = cached_data[self.split]
            ecg_segments = split_data['segments']
            labels = split_data['labels']
            
            print(f"Loaded {len(ecg_segments)} {self.split} segments from cache")
            return ecg_segments, labels
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Falling back to reprocessing...")
            return self._process_raw_data()
    
    def _save_to_cache(self, ecg_segments: List[torch.Tensor], labels: List[int]):
        """Save preprocessed data to cache"""
        try:
            # Create splits
            train_segments, train_labels = self._create_splits(ecg_segments, labels, "train")
            val_segments, val_labels = self._create_splits(ecg_segments, labels, "val")
            test_segments, test_labels = self._create_splits(ecg_segments, labels, "test")
            
            # Prepare cache data
            cache_data = {
                'train': {'segments': train_segments, 'labels': train_labels},
                'val': {'segments': val_segments, 'labels': val_labels},
                'test': {'segments': test_segments, 'labels': test_labels}
            }
            
            # Save to pickle file
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save metadata
            metadata = {
                'sample_rate': self.sample_rate,
                'trim_method': self.trim_method,
                'apply_preprocessing': self.apply_preprocessing,
                'augment': self.augment,
                'num_records': len(self._get_valid_records()),
                'total_segments': len(ecg_segments),
                'train_samples': len(train_segments),
                'val_samples': len(val_segments),
                'test_samples': len(test_segments),
                'class_distribution': self._get_class_distribution(labels),
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved preprocessed ECG data to cache: {self.cache_file}")
            print(f"Cache metadata: {self.metadata_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _process_raw_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Process raw ECG data and create segments"""
        # Get valid records
        labeled_records = self._get_valid_records()
        
        print(f"Found {len(labeled_records)} labeled ECG records")
        
        # Load raw ECG data
        ecg_data = []
        ecg_labels = []
        
        for i, record_id in enumerate(labeled_records):
            if (i + 1) % 100 == 0:
                print(f"Loading record {i + 1}/{len(labeled_records)}: {record_id}")
            
            try:
                # Load ECG data
                ecg_signal = self._load_ecg_data(record_id)
                
                # Get label
                label_row = self.reference_df[self.reference_df['record_id'] == record_id]
                if label_row.empty:
                    continue
                
                label = label_row.iloc[0]['label']
                label_idx = self.label_to_index[label]
                
                ecg_data.append(ecg_signal)
                ecg_labels.append(label_idx)
                    
            except Exception as e:
                print(f"Error processing record {record_id}: {e}")
                continue
        
        print(f"Loaded {len(ecg_data)} ECG records")
        
        # Apply trimming method
        ecg_data = self._apply_trimming(ecg_data)
        
        # Apply preprocessing if enabled
        if self.apply_preprocessing:
            ecg_data = self._apply_preprocessing(ecg_data)
        
        # Apply augmentation if enabled
        if self.augment and self.split == "train":
            ecg_data, ecg_labels = self._apply_augmentation(ecg_data, ecg_labels)
        
        # Convert to tensors
        ecg_segments = [torch.from_numpy(data).float() for data in ecg_data]
        
        print(f"Created {len(ecg_segments)} segments")
        
        return ecg_segments, ecg_labels
    
    def _get_valid_records(self) -> List[str]:
        """Get list of valid ECG record IDs"""
        if not os.path.exists(self.ecg_dir):
            raise FileNotFoundError(f"ECG data directory not found: {self.ecg_dir}")
        
        # Get all available record IDs from the directory
        available_records = set()
        for filename in os.listdir(self.ecg_dir):
            if filename.endswith('.hea'):
                record_id = filename.replace('.hea', '')
                available_records.add(record_id)
        
        # Filter to only include records that have both .hea and .mat files
        valid_records = []
        for record_id in available_records:
            mat_file = os.path.join(self.ecg_dir, f"{record_id}.mat")
            if os.path.exists(mat_file):
                valid_records.append(record_id)
        
        # Filter to only include records that have labels in reference
        labeled_records = []
        for record_id in valid_records:
            if record_id in self.reference_df['record_id'].values:
                labeled_records.append(record_id)
        
        return labeled_records
    
    def _load_ecg_data(self, record_id: str) -> np.ndarray:
        """Load ECG data from .mat file"""
        mat_file = os.path.join(self.ecg_dir, f"{record_id}.mat")
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"ECG data file not found: {mat_file}")
        
        # Load .mat file
        mat_data = loadmat(mat_file)
        
        # Find the ECG signal data (usually the largest array)
        ecg_signal = None
        for key in mat_data.keys():
            if key.startswith('__'):  # Skip metadata keys
                continue
            data = mat_data[key]
            if isinstance(data, np.ndarray) and data.ndim == 2:
                if ecg_signal is None or data.size > ecg_signal.size:
                    ecg_signal = data
        
        if ecg_signal is None:
            raise ValueError(f"No valid ECG signal found in {mat_file}")
        
        # Convert to 1D array
        if ecg_signal.shape[0] > ecg_signal.shape[1]:
            ecg_signal = ecg_signal.flatten()
        else:
            ecg_signal = ecg_signal.flatten()
        
        return ecg_signal
    
    def _apply_trimming(self, ecg_data: List[np.ndarray]) -> List[np.ndarray]:
        """Apply trimming method to ECG data"""
        # Find min and max lengths
        lengths = [len(data) for data in ecg_data]
        min_length = min(lengths)
        max_length = max(lengths)
        
        print(f"Original lengths - Min: {min_length}, Max: {max_length}")
        
        if self.trim_method == "max":
            # Pad all signals to max length with zeros at the beginning
            trimmed_data = []
            for data in ecg_data:
                diff = max_length - len(data)
                padded = np.append(np.zeros(diff), data)
                trimmed_data.append(padded)
                
        elif self.trim_method == "min":
            # Truncate all signals to min length
            trimmed_data = [data[:min_length] for data in ecg_data]
            
        elif self.trim_method == "center":
            # Extract center portion of each signal
            def clipping(x, n):
                del_width = (len(x) - n) // 2
                return x[del_width:n + del_width]
            
            trimmed_data = [clipping(data, min_length) for data in ecg_data]
            
        else:
            raise ValueError(f"Unknown trim method: {self.trim_method}")
        
        final_length = len(trimmed_data[0])
        print(f"After trimming - Length: {final_length}")
        
        return trimmed_data
    
    def _apply_preprocessing(self, ecg_data: List[np.ndarray]) -> List[np.ndarray]:
        """Apply ECG preprocessing steps"""
        processed_data = []
        
        for data in ecg_data:
            # 1. Remove baseline wander using high-pass filter
            data = self._remove_baseline_wander(data)
            
            # 2. Remove power line interference (50/60 Hz)
            data = self._remove_power_line_interference(data)
            
            # 3. Normalize to zero mean and unit variance
            data = self._normalize_signal(data)
            
            processed_data.append(data)
        
        return processed_data
    
    def _remove_baseline_wander(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove baseline wander using high-pass filter"""
        # High-pass filter with cutoff at 0.5 Hz
        nyquist = self.sample_rate / 2
        cutoff = 0.5 / nyquist
        
        if cutoff < 1.0:  # Ensure cutoff is valid
            b, a = butter(4, cutoff, btype='high')
            signal_data = filtfilt(b, a, signal_data)
        
        return signal_data
    
    def _remove_power_line_interference(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove power line interference using notch filter"""
        # Notch filter at 50 Hz (European standard)
        nyquist = self.sample_rate / 2
        if 50 < nyquist:
            # Design notch filter
            freq = 50.0 / nyquist
            q = 30.0  # Quality factor
            b, a = signal.iirnotch(freq, q)
            signal_data = signal.filtfilt(b, a, signal_data)
        
        return signal_data
    
    def _normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        # Remove DC component
        signal_data = signal_data - np.mean(signal_data)
        
        # Normalize to unit variance
        std = np.std(signal_data)
        if std > 0:
            signal_data = signal_data / std
        
        return signal_data
    
    def _apply_augmentation(self, ecg_data: List[np.ndarray], ecg_labels: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        """Apply data augmentation following the reference code approach"""
        print(f"Applying augmentation with factor {self.augment_factor}")
        
        augmented_data = ecg_data.copy()
        augmented_labels = ecg_labels.copy()
        
        for _ in range(self.augment_factor):
            for i, (data, label) in enumerate(zip(ecg_data, ecg_labels)):
                # Add noise
                noise_rate = np.random.rand() * 0.1  # Random noise rate up to 10%
                noisy_data = data + noise_rate * np.random.randn(len(data))
                
                # Time shifting
                shift_amount = np.random.randint(1, len(data) // 20)  # Shift up to 5% of signal length
                shifted_data = np.roll(noisy_data, shift_amount)
                
                augmented_data.append(shifted_data)
                augmented_labels.append(label)
        
        print(f"Augmentation complete: {len(augmented_data)} total samples")
        return augmented_data, augmented_labels
    
    def _create_splits(self, ecg_segments: List[torch.Tensor], labels: List[int], split: str) -> Tuple[List[torch.Tensor], List[int]]:
        """Create train/val/test splits"""
        # Convert to numpy arrays for easier manipulation
        segments_array = np.array(ecg_segments)
        labels_array = np.array(labels)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Get indices for each class
        class_indices = {}
        for class_idx in range(len(self.label_to_index)):
            class_indices[class_idx] = np.where(labels_array == class_idx)[0]
        
        # Create splits maintaining class distribution
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_idx, indices in class_indices.items():
            n_samples = len(indices)
            
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Calculate split sizes
            if class_idx == 0:  # Normal: 60% train, 15% val, 25% test
                n_train = int(0.60 * n_samples)
                n_val = int(0.15 * n_samples)
            elif class_idx == 1:  # AF: 50% train, 25% val, 25% test
                n_train = int(0.50 * n_samples)
                n_val = int(0.25 * n_samples)
            elif class_idx == 2:  # Other rhythm: 55% train, 20% val, 25% test
                n_train = int(0.55 * n_samples)
                n_val = int(0.20 * n_samples)
            else:  # Others: 40% train, 30% val, 30% test
                n_train = int(0.40 * n_samples)
                n_val = int(0.30 * n_samples)
            
            # Assign indices to splits
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        # Shuffle each split
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Select appropriate split
        if split == "train":
            selected_indices = train_indices
        elif split == "val":
            selected_indices = val_indices
        elif split == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Return selected segments and labels
        selected_segments = [ecg_segments[i] for i in selected_indices]
        selected_labels = [labels[i] for i in selected_indices]
        
        return selected_segments, selected_labels
    
    def _get_class_distribution(self, labels: List[int]) -> Dict[str, int]:
        """Get class distribution statistics"""
        label_counts = {}
        for label in labels:
            label_name = self.index_to_label[label]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        return label_counts
    
    def __len__(self) -> int:
        return len(self.ecg_segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get ECG segment and label
        ecg_segment = self.ecg_segments[idx]
        label = self.labels[idx]
        
        # Ensure correct shape [1, T] for consistency with other datasets
        if ecg_segment.dim() == 1:
            ecg_segment = ecg_segment.unsqueeze(0)
        
        # Apply time-invariant perturbation augmentation
        if self.time_invariant_augment and random.random() < self.time_invariant_prob:
            ecg_segment = self._apply_time_invariant_augmentation(ecg_segment)
        
        return ecg_segment, label
    
    def _apply_time_invariant_augmentation(self, ecg_segment: torch.Tensor) -> torch.Tensor:
        """Apply random time-invariant perturbation augmentation using TimeInvariantPerturbationLayer"""
        # Randomly select perturbation type and parameters
        perturbation_type = random.choice(self.perturbation_types)
        kernel_size = random.choice(self.kernel_sizes)
        epsilon = random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        
        # Create perturbation layer
        perturb_layer = TimeInvariantPerturbationLayer(
            input_signal=ecg_segment,
            perturbation_type=perturbation_type,
            kernel_size=kernel_size,
        )
        
        # Generate random perturbation parameter
        # For most perturbations, this is in [0, epsilon] range
        perturbation_param = torch.tensor([[epsilon * random.random()]])
        
        # Apply perturbation
        augmented_ecg = perturb_layer(perturbation_param)
        
        # Detach from computation graph to avoid gradient issues in DataLoader
        return augmented_ecg.detach()

def build_cardiac_arrhythmia_datasets(
    root: str,
    sample_rate: int = 100,
    trim_method: str = "center",
    apply_preprocessing: bool = True,
    augment: bool = False,
    augment_factor: int = 10,
    # Time-invariant perturbation parameters
    time_invariant_augment: bool = False,
    time_invariant_prob: float = 0.5,
    perturbation_types: Optional[List[str]] = None,
    kernel_sizes: Optional[List[int]] = None,
    epsilon_range: Tuple[float, float] = (0.1, 0.5),
    force_reprocess: bool = False,
) -> Tuple[CardiacArrhythmiaDataset, CardiacArrhythmiaDataset, CardiacArrhythmiaDataset, Dict[str, int]]:
    """
    Build Cardiac Arrhythmia datasets for train/val/test splits.
    
    Args:
        root: Root directory containing the data
        sample_rate: ECG sampling rate in Hz (default: 100 Hz)
        trim_method: Trimming method ("max", "min", "center")
        apply_preprocessing: Whether to apply ECG preprocessing
        augment: Whether to apply data augmentation
        augment_factor: Number of augmented samples per original
        time_invariant_augment: Whether to apply time-invariant perturbation augmentation
        time_invariant_prob: Probability of applying time-invariant perturbation
        perturbation_types: List of perturbation types to use
        kernel_sizes: List of kernel sizes for perturbations
        epsilon_range: Range of epsilon values for perturbation strength
        force_reprocess: Force reprocessing even if cache exists
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_mapping)
    """
    
    # Create datasets
    ds_train = CardiacArrhythmiaDataset(
        root, "train", sample_rate, trim_method, apply_preprocessing, 
        augment, augment_factor, time_invariant_augment, time_invariant_prob,
        perturbation_types, kernel_sizes, epsilon_range, force_reprocess
    )
    ds_val = CardiacArrhythmiaDataset(
        root, "val", sample_rate, trim_method, apply_preprocessing, 
        False, 0, False, 0.0, None, None, (0.0, 0.0), force_reprocess
    )
    ds_test = CardiacArrhythmiaDataset(
        root, "test", sample_rate, trim_method, apply_preprocessing, 
        False, 0, False, 0.0, None, None, (0.0, 0.0), force_reprocess
    )
    
    # Label mapping is the same for all
    label_mapping = ds_train.label_to_index
    
    return ds_train, ds_val, ds_test, label_mapping
