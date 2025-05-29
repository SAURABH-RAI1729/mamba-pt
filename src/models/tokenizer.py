"""
Tokenizer for detector modules
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path


class DetectorTokenizer:
    """
    Tokenizes detector modules for the MAMBA model
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        
        # Special tokens
        self.special_tokens = {
            pad_token: 0,
            mask_token: 1,
            unk_token: 2,
            cls_token: 3,
            sep_token: 4,
        }
        
        # Token to ID mapping
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Module ID to token mapping
        self.module_to_token = {}
        self.token_to_module = {}
        
        # Special token IDs
        self.pad_token_id = self.special_tokens[pad_token]
        self.mask_token_id = self.special_tokens[mask_token]
        self.unk_token_id = self.special_tokens[unk_token]
        self.cls_token_id = self.special_tokens[cls_token]
        self.sep_token_id = self.special_tokens[sep_token]
        
        self.is_built = False
        
    def build_vocab(self, module_ids: List[Tuple[int, int, int]]):
        """
        Build vocabulary from detector module IDs
        
        Args:
            module_ids: List of (volume_id, layer_id, module_id) tuples
        """
        unique_modules = list(set(module_ids))
        
        # Sort for consistency
        unique_modules.sort()
        
        # Assign tokens to modules
        next_token_id = len(self.special_tokens)
        
        for module in unique_modules[:self.vocab_size - len(self.special_tokens)]:
            module_str = f"{module[0]}_{module[1]}_{module[2]}"
            
            self.module_to_token[module] = next_token_id
            self.token_to_module[next_token_id] = module
            self.token_to_id[module_str] = next_token_id
            self.id_to_token[next_token_id] = module_str
            
            next_token_id += 1
        
        self.is_built = True
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def encode_track(
        self,
        track_modules: List[Tuple[int, int, int]],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Encode a track (sequence of detector modules) to token IDs
        
        Args:
            track_modules: List of (volume_id, layer_id, module_id) tuples
            max_length: Maximum sequence length
            add_special_tokens: Whether to add CLS/SEP tokens
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Convert modules to tokens
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.cls_token_id)
        
        for module in track_modules:
            if module in self.module_to_token:
                tokens.append(self.module_to_token[module])
            else:
                tokens.append(self.unk_token_id)
        
        if add_special_tokens:
            tokens.append(self.sep_token_id)
        
        # Truncation
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        
        # Padding
        if padding and max_length:
            padding_length = max_length - len(tokens)
            tokens.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs back to module identifiers"""
        modules = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                modules.append(self.id_to_token[token_id])
            else:
                modules.append(self.unk_token)
        return modules
    
    def decode_modules(self, token_ids: List[int]) -> List[Tuple[int, int, int]]:
        """Convert token IDs back to module tuples"""
        modules = []
        for token_id in token_ids:
            if token_id in self.token_to_module:
                modules.append(self.token_to_module[token_id])
            elif token_id in self.special_tokens.values():
                continue  # Skip special tokens
            else:
                modules.append(None)  # Unknown module
        return modules
    
    def create_masked_input(
        self,
        input_ids: List[int],
        mask_rate: float = 0.15,
        mask_token_id: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Create masked input for MDM task
        
        Args:
            input_ids: Original token IDs
            mask_rate: Fraction of tokens to mask
            mask_token_id: Token ID to use for masking
            
        Returns:
            Tuple of (masked_input_ids, masked_positions)
        """
        if mask_token_id is None:
            mask_token_id = self.mask_token_id
            
        input_ids = input_ids.copy()
        masked_positions = []
        
        # Don't mask special tokens
        special_token_ids = set(self.special_tokens.values())
        
        # Get maskable positions
        maskable_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in special_token_ids
        ]
        
        # Randomly select positions to mask
        num_to_mask = int(len(maskable_positions) * mask_rate)
        positions_to_mask = np.random.choice(
            maskable_positions,
            size=num_to_mask,
            replace=False
        )
        
        # Apply masking
        for pos in positions_to_mask:
            input_ids[pos] = mask_token_id
            masked_positions.append(pos)
        
        return input_ids, masked_positions
    
    def save(self, path: Path):
        """Save tokenizer to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'module_to_token': self.module_to_token,
            'token_to_module': self.token_to_module,
            'is_built': self.is_built,
        }
        
        with open(path / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer_data, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load tokenizer from disk"""
        path = Path(path)
        
        with open(path / 'tokenizer.pkl', 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = cls(vocab_size=tokenizer_data['vocab_size'])
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.module_to_token = tokenizer_data['module_to_token']
        tokenizer.token_to_module = tokenizer_data['token_to_module']
        tokenizer.is_built = tokenizer_data['is_built']
        
        # Restore special token IDs
        tokenizer.pad_token_id = tokenizer.special_tokens[tokenizer.pad_token]
        tokenizer.mask_token_id = tokenizer.special_tokens[tokenizer.mask_token]
        tokenizer.unk_token_id = tokenizer.special_tokens[tokenizer.unk_token]
        tokenizer.cls_token_id = tokenizer.special_tokens[tokenizer.cls_token]
        tokenizer.sep_token_id = tokenizer.special_tokens[tokenizer.sep_token]
        
        return tokenizer
