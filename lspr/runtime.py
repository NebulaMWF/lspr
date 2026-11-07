"""Optional torchrun data parallelism, device selection, and logging for the scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def shard(self, items: Sequence[Any]) -> List[Any]:
        """Strided slice of ``items`` owned by this rank."""
        return list(items)[self.rank::self.world_size]


def setup(device: str = "cuda") -> DistEnv:
    """Initialize the process group when launched with torchrun (``WORLD_SIZE > 1``)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistEnv(enabled, rank, world_size, local_rank, dev)


def all_mean(env: DistEnv, value: float) -> float:
    if not env.enabled:
        return float(value)
    t = torch.tensor([float(value)], device=env.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / env.world_size


def gather(env: DistEnv, items: List[Any]) -> List[Any]:
    """Concatenate per-rank lists on every rank."""
    if not env.enabled:
        return list(items)
    parts: List[Optional[List[Any]]] = [None] * env.world_size
    dist.all_gather_object(parts, list(items))
    return [x for part in parts for x in (part or [])]


def cleanup(env: DistEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


class Logger:
    """Print and append to ``<output_dir>/<name>`` (main rank only)."""

    def __init__(self, path: Optional[Path], enabled: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = enabled
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        if not self.enabled:
            return
        print(msg, flush=True)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(msg + "\n")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2))
