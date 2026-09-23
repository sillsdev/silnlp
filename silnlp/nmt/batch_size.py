import gc
import logging
from typing import Any, Callable, Optional, Set

import torch
from accelerate.utils.memory import should_reduce_batch_size
from tqdm.std import tqdm as std_tqdm

LOGGER = logging.getLogger(__name__)


def find_executable_batch_size(
    function: Optional[Callable] = None, starting_batch_size: int = 64, accelerator=None
):
    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        last_exception = None

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.") from last_exception
            open_bars = set(getattr(std_tqdm, "_instances", []))
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if indicates_out_of_memory(e):
                    last_exception = e
                    _close_orphaned_progress_bars(open_bars)
                    LOGGER.warning(
                        f"Reducing batch size from {batch_size} to {batch_size // 2} after exception: {e}. "
                        f"CUDA memory allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB, "
                        f"max allocated={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    accelerator.gradient_accumulation_steps = accelerator.gradient_accumulation_steps * 2
                    kwargs["args"].gradient_accumulation_steps = accelerator.gradient_accumulation_steps
                else:
                    raise

    return decorator


def _close_orphaned_progress_bars(open_bars: Set[Any]) -> None:
    for bar in list(getattr(std_tqdm, "_instances", [])):
        if bar not in open_bars:
            try:
                bar.close()
            except Exception:
                pass


def indicates_out_of_memory(exception: Exception) -> bool:
    if should_reduce_batch_size(exception):
        return True
    # Check for MIG Out of Memory error. Can remove when should_reduce_batch_size works on MIGs.
    if 'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp"' in str(exception):
        return True
    return False
