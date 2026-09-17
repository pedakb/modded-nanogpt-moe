"""FineWeb shard loading with serializable, relocatable cursor state."""

from pathlib import Path

import torch
import torch.distributed as dist

def _read_data_shard_header(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    return header, num_tokens


def _load_data_shard(file: Path, pin_memory=True):
    _, num_tokens = _read_data_shard_header(file)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _data_shard_identity(file: Path):
    header, num_tokens = _read_data_shard_header(file)
    return {
        "name": file.name,
        "size_bytes": file.stat().st_size,
        "num_tokens": num_tokens,
        "header": header.tolist(),
    }


class DistributedDataLoader:
    """Sequential shard loader with an explicit, relocatable cursor state."""
    def __init__(self, filename_pattern: str, batch_size: int, seq_len=1024,
                 data_root: str | Path | None = None, world_size: int | None = None,
                 rank: int | None = None, device: str | torch.device = "cuda"):
        self.filename_pattern = filename_pattern
        self.data_root = Path.cwd() if data_root is None else Path(data_root)
        self.files = sorted(self.data_root.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"no data shards match {filename_pattern!r} under {self.data_root}")
        distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if world_size is None and distributed else (world_size or 1)
        self.rank = dist.get_rank() if rank is None and distributed else (rank or 0)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is invalid for world_size {self.world_size}")
        if batch_size % self.world_size != 0:
            raise ValueError("batch_size must be divisible by world_size")
        self.batch_size = batch_size
        self.local_batch_size = batch_size // self.world_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.shard_identities = [_data_shard_identity(file) for file in self.files]
        self.shard_index = 0
        self.pos = 0
        self.tokens = _load_data_shard(
            self.files[self.shard_index], pin_memory=self.device.type == "cuda")

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.shard_index += 1
            if self.shard_index >= len(self.files):
                raise StopIteration("training data shards exhausted")
            self.tokens = _load_data_shard(
                self.files[self.shard_index], pin_memory=self.device.type == "cuda")
            self.pos = 0
        start = self.pos + self.rank * self.local_batch_size
        buf = self.tokens[start:][:self.local_batch_size + 1]
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)
        self.pos += self.batch_size
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def state_dict(self):
        return {
            "format_version": 1,
            "filename_pattern": self.filename_pattern,
            "shards": self.shard_identities,
            "shard_index": self.shard_index,
            "token_offset": self.pos,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "world_size": self.world_size,
            "rank": self.rank,
            "epoch": 0,
            "shuffle": False,
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError(f"unsupported data-loader state version: {state.get('format_version')}")
        expected = {
            "filename_pattern": self.filename_pattern,
            "shards": self.shard_identities,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "world_size": self.world_size,
            "rank": self.rank,
            "epoch": 0,
            "shuffle": False,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"incompatible data-loader {key}: checkpoint={state.get(key)!r}, current={value!r}")
        shard_index = int(state["shard_index"])
        token_offset = int(state["token_offset"])
        if not 0 <= shard_index < len(self.files):
            raise ValueError(f"invalid checkpoint shard_index: {shard_index}")
        if token_offset < 0 or token_offset % self.batch_size != 0:
            raise ValueError(f"invalid checkpoint token_offset: {token_offset}")
        self.shard_index = shard_index
        self.pos = token_offset
        self.tokens = _load_data_shard(
            self.files[self.shard_index], pin_memory=self.device.type == "cuda")
        if self.pos > len(self.tokens):
            raise ValueError(
                f"checkpoint token_offset {self.pos} exceeds shard length {len(self.tokens)}")


def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024,
                               data_root: str | Path | None = None):
    return DistributedDataLoader(filename_pattern, batch_size, seq_len, data_root=data_root)

