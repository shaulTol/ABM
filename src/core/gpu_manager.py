"""
GPU Manager Module - Intelligent GPU detection and batch size optimization

Logic Overview:
1. Detection: Query PyTorch for GPU properties (name, memory, compute capability)
2. Profiling: Match GPU against known profiles for optimization parameters
3. Batch Sizing: Calculate based on available memory and operation complexity
4. Adaptation: Catch OOM errors and reduce batch size automatically
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable
import json
import os
from pathlib import Path
import warnings
from dataclasses import dataclass
import functools

@dataclass
class GPUProfile:
    """Data class for GPU-specific optimization parameters"""
    name: str
    memory_gb: float
    compute_capability: Tuple[int, int]
    kemeny_batch_factor: float  # Multiplier for Kemeny operations
    kde_batch_factor: float      # Multiplier for KDE operations
    use_tf32: bool              # Whether to use TensorFloat-32
    use_mixed_precision: bool   # Whether to use automatic mixed precision
    memory_reserve: float = 0.2  # Reserve 20% by default

class GPUManager:
    """
    Intelligent GPU management with auto-detection and adaptive batching
    
    The manager works in three phases:
    1. Initialization: Detect GPU and load optimization profile
    2. Runtime: Calculate optimal batch sizes for operations
    3. Adaptation: Handle OOM errors gracefully
    """
    
    # Known GPU profiles with optimization parameters
    # These factors are empirically determined for each GPU model
    GPU_PROFILES = {
        "A100": GPUProfile(
            name="A100",
            memory_gb=80,  # Will be overridden by actual detection
            compute_capability=(8, 0),
            kemeny_batch_factor=50,  # Can handle 50x base batch per GB
            kde_batch_factor=100,
            use_tf32=True,
            use_mixed_precision=True
        ),
        "V100": GPUProfile(
            name="V100", 
            memory_gb=32,
            compute_capability=(7, 0),
            kemeny_batch_factor=40,
            kde_batch_factor=80,
            use_tf32=False,
            use_mixed_precision=True
        ),
        "RTX 4090": GPUProfile(
            name="RTX 4090",
            memory_gb=24,
            compute_capability=(8, 9),
            kemeny_batch_factor=45,
            kde_batch_factor=90,
            use_tf32=True,
            use_mixed_precision=True
        ),
        "RTX 3090": GPUProfile(
            name="RTX 3090",
            memory_gb=24,
            compute_capability=(8, 6),
            kemeny_batch_factor=35,
            kde_batch_factor=70,
            use_tf32=True,
            use_mixed_precision=True
        ),
        "Tesla T4": GPUProfile(
            name="Tesla T4",
            memory_gb=16,
            compute_capability=(7, 5),
            kemeny_batch_factor=25,
            kde_batch_factor=50,
            use_tf32=False,
            use_mixed_precision=True
        ),
        "P100": GPUProfile(
            name="P100",
            memory_gb=16,
            compute_capability=(6, 0),
            kemeny_batch_factor=30,
            kde_batch_factor=60,
            use_tf32=False,
            use_mixed_precision=False
        ),
        "DEFAULT": GPUProfile(
            name="Unknown GPU",
            memory_gb=8,
            compute_capability=(6, 0),
            kemeny_batch_factor=20,
            kde_batch_factor=40,
            use_tf32=False,
            use_mixed_precision=False
        )
    }
    
    def __init__(self, device: Optional[str] = None, 
                 memory_fraction: float = 0.8,
                 verbose: bool = True):
        """
        Initialize GPU Manager
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            memory_fraction: Fraction of GPU memory to use (0.8 = 80%)
            verbose: Print detection information
        """
        self.verbose = verbose
        self.memory_fraction = memory_fraction
        self.batch_size_cache = {}  # Cache successful batch sizes
        self.oom_history = {}  # Track OOM errors for each operation
        
        # Phase 1: Device Detection
        self.device = self._detect_device(device)
        self.is_cuda = self.device.type == 'cuda'
        
        # Phase 2: GPU Profiling (if CUDA available)
        if self.is_cuda:
            self.gpu_properties = self._get_gpu_properties()
            self.gpu_profile = self._match_gpu_profile()
            self._apply_optimizations()
        else:
            self.gpu_properties = None
            self.gpu_profile = None
            
        if self.verbose:
            self._print_configuration()
    
    def _detect_device(self, requested_device: Optional[str]) -> torch.device:
        """
        Smart device detection logic:
        1. If device specified, try to use it
        2. If auto/None, detect best available
        3. Fallback to CPU if CUDA unavailable
        """
        if requested_device == 'cpu':
            return torch.device('cpu')
        
        if requested_device == 'cuda' or requested_device is None:
            if torch.cuda.is_available():
                # Select best GPU (usually GPU 0, but could be configured)
                device_id = 0
                if torch.cuda.device_count() > 1:
                    # For multi-GPU, select the one with most free memory
                    device_id = self._select_best_gpu()
                return torch.device(f'cuda:{device_id}')
            elif requested_device == 'cuda':
                warnings.warn("CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
        
        # Try to parse device string (e.g., 'cuda:1')
        return torch.device(requested_device)
    
    def _select_best_gpu(self) -> int:
        """Select GPU with most free memory in multi-GPU setup"""
        max_free = 0
        best_device = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free_memory = torch.cuda.mem_get_info()[0]
            if free_memory > max_free:
                max_free = free_memory
                best_device = i
        
        return best_device
    
    def _get_gpu_properties(self) -> Dict[str, Any]:
        """
        Extract detailed GPU properties using PyTorch CUDA API
        
        Returns dict with:
        - name: GPU model name
        - total_memory_gb: Total VRAM in GB  
        - compute_capability: (major, minor) version
        - multi_processors: Number of SMs
        - cuda_cores: Estimated CUDA cores
        """
        if not self.is_cuda:
            return None
        
        device_id = self.device.index or 0
        
        # Get GPU name and clean it for matching
        gpu_name = torch.cuda.get_device_name(device_id)
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        total_memory_gb = total_memory / (1024**3)
        
        # Get compute capability
        major = torch.cuda.get_device_properties(device_id).major
        minor = torch.cuda.get_device_properties(device_id).minor
        
        # Get other properties
        props = torch.cuda.get_device_properties(device_id)
        
        return {
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'compute_capability': (major, minor),
            'multi_processors': props.multi_processor_count,
            'cuda_cores': props.multi_processor_count * self._cores_per_sm(major),
            'max_threads_per_block': props.max_threads_per_multi_processor if hasattr(props, 'max_threads_per_multi_processor') else 1024,
            'warp_size': props.warp_size if hasattr(props, 'warp_size') else 32,
            'device_id': device_id
        }
    
    def _cores_per_sm(self, major: int) -> int:
        """
        Calculate CUDA cores per SM based on compute capability
        
        This varies by GPU architecture:
        - Kepler (3.x): 192 cores/SM
        - Maxwell (5.x): 128 cores/SM  
        - Pascal (6.x): 64 cores/SM
        - Volta/Turing (7.x): 64 cores/SM
        - Ampere (8.x): 64 or 128 cores/SM
        """
        cores_per_sm_dict = {
            3: 192,  # Kepler
            5: 128,  # Maxwell
            6: 64,   # Pascal
            7: 64,   # Volta/Turing
            8: 64,   # Ampere (varies, using conservative)
            9: 128   # Hopper
        }
        return cores_per_sm_dict.get(major, 64)
    
    def _match_gpu_profile(self) -> GPUProfile:
        """
        Match detected GPU to optimization profile
        
        Logic:
        1. Try exact name match
        2. Try partial name match (e.g., "3090" in name)
        3. Use compute capability to guess
        4. Fall back to DEFAULT profile
        """
        gpu_name = self.gpu_properties['name']
        
        # Clean GPU name for matching
        gpu_name_clean = gpu_name.upper().replace("GEFORCE", "").replace("NVIDIA", "").strip()
        
        # Try exact match first
        for profile_name, profile in self.GPU_PROFILES.items():
            if profile_name.upper() in gpu_name_clean:
                # Override memory with detected value
                profile.memory_gb = self.gpu_properties['total_memory_gb']
                return profile
        
        # Try compute capability matching
        cc = self.gpu_properties['compute_capability']
        if cc[0] >= 8:  # Ampere or newer
            profile = self.GPU_PROFILES.get("RTX 3090", self.GPU_PROFILES["DEFAULT"])
        elif cc[0] == 7 and cc[1] >= 5:  # Turing
            profile = self.GPU_PROFILES.get("Tesla T4", self.GPU_PROFILES["DEFAULT"])
        elif cc[0] == 7:  # Volta
            profile = self.GPU_PROFILES.get("V100", self.GPU_PROFILES["DEFAULT"])
        else:
            profile = self.GPU_PROFILES["DEFAULT"]
        
        # Always use actual detected memory
        profile.memory_gb = self.gpu_properties['total_memory_gb']
        return profile
    
    def _apply_optimizations(self):
        """Apply GPU-specific optimizations based on profile"""
        if not self.gpu_profile:
            return
        
        # Enable TF32 for Ampere GPUs (3x faster for matmul)
        if self.gpu_profile.use_tf32 and hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set cudnn benchmarking for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def estimate_batch_size(self, 
                          operation: str,
                          data_shape: Tuple[int, ...],
                          dtype: torch.dtype = torch.float32,
                          additional_memory_gb: float = 0) -> int:
        """
        Estimate optimal batch size for an operation
        
        The calculation logic:
        1. Get available GPU memory
        2. Apply operation-specific factor from GPU profile
        3. Account for data dimensions and dtype
        4. Reserve safety margin
        5. Check cache for previously successful sizes
        
        Args:
            operation: Type of operation ('kemeny', 'kde', 'projection', 'eigenvalue')
            data_shape: Shape of data (n_samples, n_features)
            dtype: Data type (affects memory usage)
            additional_memory_gb: Extra memory needed for operation
            
        Returns:
            Recommended batch size
        """
        if not self.is_cuda:
            # CPU: Use fixed batch sizes
            return {'kemeny': 10, 'kde': 100, 'projection': 50, 'eigenvalue': 5}.get(operation, 20)
        
        # Check cache first
        cache_key = (operation, data_shape, dtype)
        if cache_key in self.batch_size_cache:
            return self.batch_size_cache[cache_key]
        
        # Get available memory
        free_memory_gb = self._get_free_memory_gb()
        usable_memory_gb = free_memory_gb * self.memory_fraction - additional_memory_gb
        
        if usable_memory_gb <= 0:
            warnings.warn(f"Insufficient GPU memory for operation {operation}")
            return 1
        
        # Get operation-specific batch factor
        if operation == 'kemeny':
            batch_factor = self.gpu_profile.kemeny_batch_factor
        elif operation == 'kde':
            batch_factor = self.gpu_profile.kde_batch_factor
        elif operation == 'projection':
            batch_factor = self.gpu_profile.kde_batch_factor * 1.5  # Projections are lighter
        elif operation == 'eigenvalue':
            batch_factor = self.gpu_profile.kemeny_batch_factor * 0.5  # Eigenvalues are heavy
        else:
            batch_factor = 30  # Default
        
        # Calculate batch size based on data dimensions
        n_samples, n_features = data_shape[:2] if len(data_shape) >= 2 else (data_shape[0], 1)
        
        # Memory usage estimation (rough heuristic)
        bytes_per_element = {'float32': 4, 'float64': 8, 'float16': 2}.get(str(dtype).split('.')[-1], 4)
        
        # Base memory per batch item (considering intermediate tensors)
        memory_per_item_mb = (n_features * bytes_per_element * 10) / (1024**2)  # 10x for intermediates
        
        # Special adjustments for operations
        if operation == 'kemeny':
            # Kemeny needs transition matrix (n_discretization x n_discretization)
            memory_per_item_mb *= 5  # Additional factor for matrix operations
        elif operation == 'eigenvalue':
            # Eigenvalue decomposition is memory intensive
            memory_per_item_mb *= 10
        
        # Calculate batch size
        batch_size = int(usable_memory_gb * 1024 * batch_factor / memory_per_item_mb)
        
        # Apply bounds
        batch_size = max(1, min(batch_size, 10000))  # Reasonable bounds
        
        # If we've had OOM with this operation, be conservative
        if operation in self.oom_history:
            reduction_factor = 0.7 ** self.oom_history[operation]
            batch_size = int(batch_size * reduction_factor)
        
        return batch_size
    
    def _get_free_memory_gb(self) -> float:
        """Get current free GPU memory in GB"""
        if not self.is_cuda:
            return 0
        
        # Clear cache first for accurate reading
        torch.cuda.empty_cache()
        
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        return free_memory / (1024**3)
    
    def adaptive_compute(self, 
                        func: Callable,
                        *args,
                        initial_batch_size: Optional[int] = None,
                        min_batch_size: int = 1,
                        reduction_factor: float = 0.5,
                        **kwargs) -> Any:
        """
        Wrapper that automatically adjusts batch size on OOM
        
        Algorithm:
        1. Try with initial batch size (or estimated)
        2. On OOM, reduce batch size by reduction_factor
        3. Retry until success or min_batch_size reached
        4. Cache successful batch size
        
        Args:
            func: Function to execute (must accept batch_size kwarg)
            initial_batch_size: Starting batch size (None = auto)
            min_batch_size: Minimum batch size before giving up
            reduction_factor: How much to reduce batch on OOM (0.5 = half)
            
        Returns:
            Function result
        """
        batch_size = initial_batch_size or 100
        
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            try:
                # Clear cache before attempt
                if self.is_cuda:
                    torch.cuda.empty_cache()
                
                # Try to run function
                result = func(*args, batch_size=batch_size, **kwargs)
                
                # Success! Cache this batch size
                if hasattr(func, '__name__'):
                    self.batch_size_cache[func.__name__] = batch_size
                
                if self.verbose and attempts > 0:
                    print(f"✓ Succeeded with batch_size={batch_size}")
                
                return result
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' in str(e).lower():
                    # OOM detected
                    attempts += 1
                    
                    # Track OOM for this operation
                    op_name = getattr(func, '__name__', 'unknown')
                    self.oom_history[op_name] = self.oom_history.get(op_name, 0) + 1
                    
                    # Reduce batch size
                    old_batch_size = batch_size
                    batch_size = max(min_batch_size, int(batch_size * reduction_factor))
                    
                    if self.verbose:
                        print(f"⚠ OOM with batch_size={old_batch_size}, reducing to {batch_size}")
                    
                    if batch_size == min_batch_size and attempts > 1:
                        print(f"✗ Failed even with minimum batch_size={min_batch_size}")
                        raise
                    
                    # Clear GPU memory
                    if self.is_cuda:
                        torch.cuda.empty_cache()
                else:
                    # Non-OOM error, re-raise
                    raise
        
        raise RuntimeError(f"Failed after {max_attempts} attempts to find working batch size")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not self.is_cuda:
            return {'device': 'cpu', 'memory_available': 'N/A'}
        
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        free, total = torch.cuda.mem_get_info(self.device)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        return {
            'device': str(self.device),
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free_gb,
            'total_gb': total_gb,
            'utilization_percent': (allocated / total_gb) * 100
        }
    
    def _print_configuration(self):
        """Print detected configuration"""
        print("\n" + "="*60)
        print("GPU MANAGER CONFIGURATION")
        print("="*60)
        
        if self.is_cuda:
            print(f"Device: {self.gpu_properties['name']}")
            print(f"Memory: {self.gpu_properties['total_memory_gb']:.1f} GB")
            print(f"Compute Capability: {self.gpu_properties['compute_capability']}")
            print(f"CUDA Cores: ~{self.gpu_properties['cuda_cores']}")
            print(f"Profile: {self.gpu_profile.name}")
            print(f"TF32: {'Enabled' if self.gpu_profile.use_tf32 else 'Disabled'}")
            print(f"Memory Reserved: {(1-self.memory_fraction)*100:.0f}%")
        else:
            print("Device: CPU")
            print("GPU acceleration: Disabled")
        
        print("="*60 + "\n")
    
    def benchmark_operation(self, operation: str, data_shape: Tuple[int, ...], 
                           n_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark an operation to find optimal batch size
        
        Returns timing and memory statistics
        """
        # This will be implemented with actual operations
        pass
    
    @property
    def config_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'device': str(self.device),
            'is_cuda': self.is_cuda,
            'gpu_properties': self.gpu_properties,
            'gpu_profile': self.gpu_profile.__dict__ if self.gpu_profile else None,
            'memory_fraction': self.memory_fraction,
            'batch_size_cache': self.batch_size_cache,
            'oom_history': self.oom_history
        }


# Convenience function for global GPU manager instance
_global_gpu_manager = None

def get_gpu_manager(reset: bool = False, **kwargs) -> GPUManager:
    """
    Get or create global GPU manager instance
    
    This ensures we only detect GPU once and reuse the configuration
    """
    global _global_gpu_manager
    
    if _global_gpu_manager is None or reset:
        _global_gpu_manager = GPUManager(**kwargs)
    
    return _global_gpu_manager
