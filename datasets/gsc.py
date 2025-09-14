from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torchaudio
import warnings
import torch
import os
import random

from perturbations.time_invariant import TimeInvariantPerturbationLayer


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*TorchCodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)

class SpeechCommandsDigitsDataset(Dataset):
    """
    Speech Commands dataset for digit recognition (zero, one, two, three, four, five, six, seven, eight, nine)
    """
    
    # Hardcoded digit labels
    DIGIT_LABELS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    def __init__(
        self,
        root: str,
        split: str,  # "train" | "val" | "test"
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        augment: bool = False,
        augment_prob: float = 0.5,
        perturbation_types: Optional[List[str]] = None,
        kernel_sizes: Optional[List[int]] = None,
        epsilon_range: Tuple[float, float] = (0.1, 0.5),
    ):
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_s)
        
        # Augmentation parameters (only used for training)
        self.augment = augment and split == "train"
        self.augment_prob = augment_prob
        self.perturbation_types = perturbation_types or ['shift', 'lowpass', 'echo', 'highpass', 'gaussian']
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.epsilon_range = epsilon_range
        
        # Set up paths
        self.audio_dir = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
        
        # Load split lists
        split_files = {
            "train": "training_list.txt",
            "val": "validation_list.txt", 
            "test": "testing_list.txt"
        }
        
        # Create label mapping
        self.label_to_index = {label: i for i, label in enumerate(self.DIGIT_LABELS)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        
        # Load and filter files
        self.file_list = self._load_digit_files(split_files[split])
        
    def _load_digit_files(self, split_filename: str) -> List[str]:
        """Load only files that belong to digit labels"""
        # For training, we need to compute which files are training files
        if self.split == "train":
            return self._get_training_files()
        else:
            # For val and test, load from the split file
            split_path = os.path.join(self.audio_dir, split_filename)
            with open(split_path, 'r') as f:
                all_split_files = [line.strip() for line in f.readlines()]
            
            # Filter to only include digit files
            digit_files = []
            for file_path in all_split_files:
                label = os.path.dirname(file_path)
                if label in self.DIGIT_LABELS:
                    # Verify file exists
                    full_path = os.path.join(self.audio_dir, file_path)
                    if os.path.exists(full_path):
                        digit_files.append(file_path)
            
            return digit_files
    
    def _get_training_files(self) -> List[str]:
        """Get training files by excluding validation and test files from all digit files"""
        # Load validation and test files
        val_path = os.path.join(self.audio_dir, "validation_list.txt")
        test_path = os.path.join(self.audio_dir, "testing_list.txt")
        
        val_files = set()
        test_files = set()
        
        if os.path.exists(val_path):
            with open(val_path, 'r') as f:
                val_files = set(line.strip() for line in f.readlines())
        
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                test_files = set(line.strip() for line in f.readlines())
        
        excluded_files = val_files | test_files
        
        # Get all digit files
        all_digit_files = []
        for label in self.DIGIT_LABELS:
            label_dir = os.path.join(self.audio_dir, label)
            if os.path.exists(label_dir):
                for filename in os.listdir(label_dir):
                    if filename.endswith('.wav'):
                        file_path = f"{label}/{filename}"
                        # Include only if not in validation or test sets
                        if file_path not in excluded_files:
                            all_digit_files.append(file_path)
        
        return all_digit_files
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        """Fix audio length to target duration"""
        # wav: [1, T]
        T = wav.shape[-1]
        if T == self.num_samples:
            return wav
        if T > self.num_samples:
            return wav[:, :self.num_samples]
        # pad with zeros at end
        pad = self.num_samples - T
        return torch.nn.functional.pad(wav, (0, pad))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get file path and construct full path
        file_path = self.file_list[idx]
        full_path = os.path.join(self.audio_dir, file_path)
        
        # Load audio
        waveform, sr = torchaudio.load(full_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed (though GSC should be 16kHz already)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Fix length
        waveform = self._fix_length(waveform)
        
        # Apply augmentation if enabled and for training split
        if self.augment and random.random() < self.augment_prob:
            waveform = self._apply_augmentation(waveform)
        
        # Get label from directory name
        label = os.path.dirname(file_path)
        y = self.label_to_index[label]
        
        return waveform, y
    
    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation using TimeInvariantPerturbationLayer"""
        # Randomly select perturbation type and parameters
        perturbation_type = random.choice(self.perturbation_types)
        kernel_size = random.choice(self.kernel_sizes)
        epsilon = random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        
        # Create perturbation layer
        perturb_layer = TimeInvariantPerturbationLayer(
            input_signal=waveform,
            perturbation_type=perturbation_type,
            kernel_size=kernel_size,
        )
        
        # Generate random perturbation parameter
        # For most perturbations, this is in [0, epsilon] range
        perturbation_param = torch.tensor([[epsilon * random.random()]])
        
        # Apply perturbation
        augmented_waveform = perturb_layer(perturbation_param)
        
        # Detach from computation graph to avoid gradient issues in DataLoader
        return augmented_waveform.detach()
        
def build_speech_commands_datasets(
    root: str,
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    download: bool = True,
    augment: bool = False,
    augment_prob: float = 0.5,
    perturbation_types: Optional[List[str]] = None,
    kernel_sizes: Optional[List[int]] = None,
    epsilon_range: Tuple[float, float] = (0.1, 0.5),
) -> Tuple[SpeechCommandsDigitsDataset, SpeechCommandsDigitsDataset, SpeechCommandsDigitsDataset, Dict[str, int]]:
        
    # Download dataset if needed (using torchaudio)
    if download:
        try:
            import torchaudio.datasets
            _ = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=download)
        except Exception as e:
            print(f"Download failed, assuming data exists: {e}")
    
    # Create datasets
    ds_train = SpeechCommandsDigitsDataset(
        root, "train", sample_rate, duration_s, 
        augment=augment, augment_prob=augment_prob,
        perturbation_types=perturbation_types, kernel_sizes=kernel_sizes,
        epsilon_range=epsilon_range
    )
    ds_val = SpeechCommandsDigitsDataset(root, "val", sample_rate, duration_s)
    ds_test = SpeechCommandsDigitsDataset(root, "test", sample_rate, duration_s)
    
    # Label mapping is the same for all
    label_mapping = ds_train.label_to_index
    
    return ds_train, ds_val, ds_test, label_mapping