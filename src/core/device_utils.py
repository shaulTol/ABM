"""
Device Utils Module - Memory management and device operations

This module provides:
1. Memory monitoring with alerts
2. Tensor movement between devices
3. Batch processing utilities
4. Device synchronization helpers
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional, Generator, Any, Dict
import functools
import warnings
from contextlib import contextmanager
import gc
import time


class MemoryMonitor:
    """
    Real-time GPU memory monitoring
    
    Logic:
    - Track memory usage before/after operations
    - Alert when usage exceeds thresholds
    - Provide memory cleanup utilities
    """
    
    def __init__(self, device: torch.device, threshold_percent: float = 90.0):
        """
        Initialize memory monitor
        
        Args:
            device: PyTorch device to monitor
            threshold_percent: Alert when memory usage exceeds this percentage
        """
        self.device = device
        self.threshold_percent = threshold_percent
        self.is_cuda = device.type == 'cuda'
        self.history = []
        
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not self.is_cuda:
            return {'type': 'cpu', 'usage': 0}
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        
        usage_percent = (allocated / total) * 100
        
        stats = {
            'timestamp': time.time(),
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': free / (1024**3),
            'total_gb': total / (1024**3),
            'usage_percent': usage_percent
        }
        
        # Check threshold
        if usage_percent > self.threshold_percent:
            warnings.warn(f"GPU memory usage ({usage_percent:.1f}%) exceeds threshold ({self.threshold_percent}%)")
        
        return stats
    
    @contextmanager
    def track_memory(self, operation_name: str = "operation"):
        """
        Context manager to track memory usage of an operation
        
        Usage:
            with monitor.track_memory("matrix_multiplication"):
                result = torch.matmul(a, b)
        """
        if not self.is_cuda:
            yield
            return
        
        # Get initial memory
        torch.cuda.synchronize(self.device)
        start_memory = torch.cuda.memory_allocated(self.device)
        start_time = time.time()
        
        try:
            yield
        finally:
            # Get final memory
            torch.cuda.synchronize(self.device)
            end_memory = torch.cuda.memory_allocated(self.device)
            end_time = time.time()
            
            # Calculate delta
            memory_delta_mb = (end_memory - start_memory) / (1024**2)
            time_delta = end_time - start_time
            
            # Store in history
            self.history.append({
                'operation': operation_name,
                'memory_delta_mb': memory_delta_mb,
                'time_seconds': time_delta,
                'timestamp': start_time
            })
            
            # Report if significant
            if abs(memory_delta_mb) > 100:  # More than 100MB change
                sign = "+" if memory_delta_mb > 0 else ""
                print(f"Memory: {operation_name}: {sign}{memory_delta_mb:.1f} MB ({time_delta:.3f}s)")
    
    def cleanup(self, aggressive: bool = False):
        """
        Clean up GPU memory
        
        Args:
            aggressive: If True, also runs garbage collection
        """
        if not self.is_cuda:
            return
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        if aggressive:
            # Force garbage collection
            gc.collect()
            
            # Synchronize and clear again
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
    
    def get_summary(self) -> str:
        """Get summary of memory usage history"""
        if not self.history:
            return "No operations tracked"
        
        total_memory = sum(h['memory_delta_mb'] for h in self.history)
        total_time = sum(h['time_seconds'] for h in self.history)
        
        summary = f"Operations: {len(self.history)}\n"
        summary += f"Total memory delta: {total_memory:.1f} MB\n"
        summary += f"Total time: {total_time:.3f} seconds\n"
        
        # Top memory users
        sorted_ops = sorted(self.history, key=lambda x: abs(x['memory_delta_mb']), reverse=True)
        summary += "\nTop memory operations:\n"
        for op in sorted_ops[:5]:
            summary += f"  {op['operation']}: {op['memory_delta_mb']:.1f} MB\n"
        
        return summary


def move_to_device(data: Union[torch.Tensor, List, Tuple, Dict], 
                   device: torch.device) -> Union[torch.Tensor, List, Tuple, Dict]:
    """
    Recursively move data to specified device
    
    Logic:
    - Handle single tensors
    - Recursively process containers (list, tuple, dict)
    - Preserve structure
    
    Args:
        data: Input data (tensor or container of tensors)
        device: Target device
        
    Returns:
        Data moved to device with same structure
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        # Not a tensor or container, return as is
        return data


def auto_device(func):
    """
    Decorator to automatically move inputs to correct device
    
    Logic:
    - Detect device from first tensor argument
    - Move all tensor arguments to that device
    - Return results on same device
    
    Usage:
        @auto_device
        def my_function(x, y):
            return x + y
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find first tensor to determine device
        device = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        
        if device is None:
            # Check kwargs
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
        
        if device is None:
            # No tensors found, use default
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move all arguments to device
        args = [move_to_device(arg, device) for arg in args]
        kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
        
        # Call function
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper


class BatchProcessor:
    """
    Utility for processing data in batches with automatic sizing
    
    Logic:
    - Split data into optimal batches
    - Process with progress tracking
    - Aggregate results
    """
    
    def __init__(self, gpu_manager=None):
        """
        Initialize batch processor
        
        Args:
            gpu_manager: GPUManager instance for batch size estimation
        """
        self.gpu_manager = gpu_manager
        
    def process_in_batches(self,
                          data: torch.Tensor,
                          func: callable,
                          batch_size: Optional[int] = None,
                          operation: str = "process",
                          aggregate: str = "concat",
                          show_progress: bool = True) -> torch.Tensor:
        """
        Process data in batches
        
        Logic:
        1. Determine optimal batch size
        2. Split data into batches
        3. Process each batch
        4. Aggregate results
        
        Args:
            data: Input tensor
            func: Function to apply to each batch
            batch_size: Batch size (None = auto)
            operation: Operation name for batch size estimation
            aggregate: How to combine results ('concat', 'mean', 'sum', 'list')
            show_progress: Show progress bar
            
        Returns:
            Processed results
        """
        n_samples = data.shape[0]
        
        # Determine batch size
        if batch_size is None and self.gpu_manager:
            batch_size = self.gpu_manager.estimate_batch_size(
                operation, data.shape, data.dtype
            )
        elif batch_size is None:
            batch_size = 100  # Default
        
        # Process batches
        results = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(0, n_samples, batch_size), 
                            desc=operation, 
                            total=n_batches)
        else:
            batch_iter = range(0, n_samples, batch_size)
        
        for start_idx in batch_iter:
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            # Process batch
            batch_result = func(batch)
            results.append(batch_result)
        
        # Aggregate results
        if aggregate == "concat":
            return torch.cat(results, dim=0)
        elif aggregate == "mean":
            return torch.stack(results).mean(dim=0)
        elif aggregate == "sum":
            return torch.stack(results).sum(dim=0)
        elif aggregate == "list":
            return results
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
    
    def generate_batches(self, 
                        data: torch.Tensor,
                        batch_size: Optional[int] = None,
                        shuffle: bool = False,
                        drop_last: bool = False) -> Generator[torch.Tensor, None, None]:
        """
        Generate batches from data
        
        Args:
            data: Input tensor
            batch_size: Batch size (None = auto)
            shuffle: Randomly shuffle data
            drop_last: Drop last incomplete batch
            
        Yields:
            Batch tensors
        """
        n_samples = data.shape[0]
        
        if batch_size is None and self.gpu_manager:
            batch_size = self.gpu_manager.estimate_batch_size(
                "process", data.shape, data.dtype
            )
        elif batch_size is None:
            batch_size = 100
        
        # Create indices
        indices = torch.arange(n_samples)
        if shuffle:
            indices = indices[torch.randperm(n_samples)]
        
        # Generate batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            if drop_last and end_idx - start_idx < batch_size:
                break
            
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices]


@contextmanager
def device_context(device: Union[str, torch.device]):
    """
    Context manager for temporary device switching
    
    Usage:
        with device_context('cuda:1'):
            # Operations here run on cuda:1
            tensor = torch.randn(100, 100)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == 'cuda':
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(device)
            yield device
        finally:
            torch.cuda.set_device(old_device)
    else:
        yield device


def synchronize(device: Optional[torch.device] = None):
    """
    Synchronize device operations
    
    Important for accurate timing and memory measurements
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def get_optimal_threads():
    """
    Get optimal number of CPU threads for operations
    
    Logic:
    - Use half of available cores for parallel operations
    - Leave some for system processes
    """
    import os
    
    try:
        # Get number of CPU cores
        cpu_count = os.cpu_count() or 4
        
        # Use half to 3/4 of cores
        optimal = max(1, min(cpu_count - 2, int(cpu_count * 0.75)))
        
        return optimal
    except:
        return 4  # Safe default


class TensorPool:
    """
    Pre-allocated tensor pool for reducing allocation overhead
    
    Logic:
    - Pre-allocate tensors of common sizes
    - Reuse instead of reallocating
    - Automatic cleanup when done
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.pool = {}
        self.in_use = set()
        
    def get_tensor(self, shape: Tuple[int, ...], 
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get tensor from pool or allocate new
        
        Logic:
        1. Check if tensor of this shape exists in pool
        2. If yes and not in use, return it
        3. Otherwise allocate new
        """
        key = (shape, dtype)
        
        if key in self.pool and key not in self.in_use:
            # Reuse existing tensor
            tensor = self.pool[key]
            self.in_use.add(key)
            tensor.zero_()  # Clear previous values
            return tensor
        else:
            # Allocate new tensor
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            self.pool[key] = tensor
            self.in_use.add(key)
            return tensor
    
    def release_tensor(self, tensor: torch.Tensor):
        """Release tensor back to pool"""
        key = (tensor.shape, tensor.dtype)
        self.in_use.discard(key)
    
    def clear(self):
        """Clear entire pool"""
        self.pool.clear()
        self.in_use.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


# Utility functions for common operations

def safe_inverse(matrix: torch.Tensor, 
                 regularization: float = 1e-10) -> torch.Tensor:
    """
    Compute matrix inverse with regularization for stability
    
    Logic:
    - Add small value to diagonal for numerical stability
    - Use pseudo-inverse for rank-deficient matrices
    """
    try:
        # Try regular inverse first
        return torch.linalg.inv(matrix)
    except:
        # Fall back to pseudo-inverse with regularization
        n = matrix.shape[0]
        reg_matrix = matrix + regularization * torch.eye(n, device=matrix.device)
        
        try:
            return torch.linalg.inv(reg_matrix)
        except:
            # Use SVD-based pseudo-inverse as last resort
            return torch.linalg.pinv(matrix, rcond=regularization)


def safe_cholesky(matrix: torch.Tensor, 
                  regularization: float = 1e-6) -> torch.Tensor:
    """
    Compute Cholesky decomposition with regularization
    
    Logic:
    - Ensure positive definiteness
    - Add regularization if needed
    """
    try:
        return torch.linalg.cholesky(matrix)
    except:
        # Add regularization to diagonal
        n = matrix.shape[0]
        reg_matrix = matrix + regularization * torch.eye(n, device=matrix.device)
        
        # Increase regularization if still failing
        for reg_scale in [1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                return torch.linalg.cholesky(
                    matrix + reg_scale * torch.eye(n, device=matrix.device)
                )
            except:
                continue
        
        # Last resort: eigenvalue thresholding
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=regularization)
        matrix_fixed = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        return torch.linalg.cholesky(matrix_fixed)


def estimate_memory_usage(shape: Tuple[int, ...], 
                         dtype: torch.dtype = torch.float32,
                         overhead_factor: float = 1.5) -> float:
    """
    Estimate memory usage for tensor in MB
    
    Logic:
    - Calculate base memory from shape and dtype
    - Apply overhead factor for intermediate operations
    """
    dtype_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1
    }
    
    bytes_per_element = dtype_bytes.get(dtype, 4)
    n_elements = np.prod(shape)
    
    base_memory_mb = (n_elements * bytes_per_element) / (1024**2)
    total_memory_mb = base_memory_mb * overhead_factor
    
    return total_memory_mb
