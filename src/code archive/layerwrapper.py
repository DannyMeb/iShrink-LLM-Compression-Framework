#layerwrapper.py

import torch
import torch.nn as nn

class BiasGPT:
    """
    Wrapper class for computing layer statistics
    """
    def __init__(self, weight_tensor: torch.Tensor, metric: str, device: torch.device):
        self.weight = weight_tensor
        self.device = device
        self.out_dim, self.in_dim = weight_tensor.shape
        self.type = metric
        self.nsamples = 0
        
        # Initialize statistics tensors
        self.baseline_inp = torch.zeros(self.in_dim, device=self.device)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros(self.in_dim, device=self.device)
        else:
            self.fluc_inp = torch.zeros(self.in_dim, device=self.device)
            
        print(f"\nInitialized BiasGPT wrapper:")
        print(f"Weight shape: {self.out_dim}x{self.in_dim}")
        print(f"Metric type: {self.type}")
        print(f"Device: {self.device}")

    def add_batch(self, inp: torch.Tensor):
        """
        Update statistics with a new batch of inputs
        """
        print(f"\nProcessing input batch:")
        print(f"Input shape: {inp.shape}")
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            
        batch_size = inp.shape[0]
        
        # Reshape and handle dimensions properly
        if len(inp.shape) == 3:
            # Reshape from [batch_size, seq_len, hidden_dim] to [hidden_dim, batch_size * seq_len]
            inp = inp.reshape(-1, inp.shape[-1]).t()
        else:
            inp = inp.t()
            
        inp = inp.to(self.device).float()
        
        print(f"Transformed input shape: {inp.shape}")
        print(f"Current samples: {self.nsamples}")
        
        # Store current baseline for variance computation
        old_baseline = self.baseline_inp.clone()
        
        # Update baseline mean (use proper dimensions)
        if inp.shape[0] != self.in_dim:
            # Need to handle dimension mismatch
            inp = inp.reshape(self.in_dim, -1)
        
        # Update running statistics
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        
        # Update metric-specific statistics
        if self.type == "WIFN":
            # Update weight-independent feature norm
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            norm_sq = torch.norm(inp, p=2, dim=1) ** 2
            self.scaler_inp += norm_sq / (self.nsamples + batch_size)
            
            print(f"Updated scaler stats:")
            print(f"min={self.scaler_inp.min().item():.6f}, "
                  f"max={self.scaler_inp.max().item():.6f}, "
                  f"mean={self.scaler_inp.mean().item():.6f}")
            
        else:
            # Update variance statistics
            if self.nsamples == 0:
                self.fluc_inp = torch.zeros_like(self.baseline_inp)
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                fluc = torch.sum(
                    (inp - self.baseline_inp.unsqueeze(1)) * 
                    (inp - old_baseline.unsqueeze(1)),
                    dim=1
                ) / (self.nsamples + batch_size)
                self.fluc_inp += fluc
                
            print(f"Updated fluctuation stats:")
            print(f"min={self.fluc_inp.min().item():.6f}, "
                  f"max={self.fluc_inp.max().item():.6f}, "
                  f"mean={self.fluc_inp.mean().item():.6f}")
        
        self.nsamples += batch_size

    def free(self):
        """
        Free memory and clear statistics
        """
        print(f"\nFreeing wrapper resources")
        self.baseline_inp = None
        self.weight = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        torch.cuda.empty_cache()