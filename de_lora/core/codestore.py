"""codestore.py — consolidated storage for the pre-computed codec codes.

The pt-BR template stores one `.npz` per utterance. On the machine this was
developed on, the data disk is a virtiofs share: every write syscall goes
through FUSE to the host, so small blocks pay the latency thousands of times
(measured 1.5 MB/s with 16 KiB writes vs 14.9 MB/s with 8 MiB writes). With
~200 h of audio that would be more than 100,000 individual files — slow to
write, slow to read, and the host keeps an O_PATH descriptor per inode.

Hence: a single stream `codes.i16` (int16, N_frames x 16, concatenated) plus
`index.jsonl` mapping idx -> (offset_frames, n_frames). Writes go through an
8 MiB buffer, reads use np.memmap.

The format is deliberately trivial: raw int16 without compression. The codes
are indices into 2048-entry codebooks, they barely compress, and
np.savez_compressed only cost CPU time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_CODEBOOKS = 16
_ITEM_BYTES = N_CODEBOOKS * 2  # int16
FLUSH_BYTES = 8 * 1024 * 1024  # virtiofs: large blocks, see the module header


class CodeWriter:
    """Appends codes to the stream. Resumable: idx values that already exist
    are reported by `has()`, and the caller skips them."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_p = self.root / "codes.i16"
        self.index_p = self.root / "index.jsonl"
        self.index: dict[str, tuple[int, int]] = {}
        self._next_frame = 0
        if self.index_p.is_file():
            for ln in self.index_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                self.index[r["idx"]] = (r["off"], r["n"])
                self._next_frame = max(self._next_frame, r["off"] + r["n"])
        # Truncate the stream to the length covered by the index: an aborted run
        # may have written bytes that never made it into the index. Otherwise all
        # subsequent offsets would be shifted.
        want = self._next_frame * _ITEM_BYTES
        if self.data_p.is_file():
            have = self.data_p.stat().st_size
            if have > want:
                with self.data_p.open("r+b") as fh:
                    fh.truncate(want)
            elif have < want:
                raise RuntimeError(
                    f"codes.i16 is shorter ({have} B) than the index requires ({want} B) — "
                    "index and stream do not match")
        self._fh = self.data_p.open("ab", buffering=0)
        self._idx_fh = self.index_p.open("a", encoding="utf-8")
        self._buf = bytearray()
        self._pending: list[str] = []

    def has(self, idx: str) -> bool:
        return idx in self.index

    def append(self, idx: str, codes: np.ndarray) -> int:
        if codes.ndim != 2 or codes.shape[1] != N_CODEBOOKS:
            raise ValueError(f"expected (T,{N_CODEBOOKS}), got {codes.shape}")
        codes = np.ascontiguousarray(codes, dtype=np.int16)
        n = int(codes.shape[0])
        off = self._next_frame
        self._buf += codes.tobytes()
        self._pending.append(json.dumps({"idx": idx, "off": off, "n": n}))
        self.index[idx] = (off, n)
        self._next_frame += n
        if len(self._buf) >= FLUSH_BYTES:
            self.flush()
        return n

    def flush(self) -> None:
        # Order matters: first the data on disk, then the index. The other way
        # round, a crash could leave an index that points at bytes never written.
        if self._buf:
            self._fh.write(bytes(self._buf))
            self._buf.clear()
        if self._pending:
            self._idx_fh.write("\n".join(self._pending) + "\n")
            self._idx_fh.flush()
            self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()
        self._idx_fh.close()

    def __enter__(self) -> "CodeWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CodeReader:
    """Reads codes via np.memmap — no syscall per item in the normal case."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index: dict[str, tuple[int, int]] = {}
        index_p = self.root / "index.jsonl"
        if index_p.is_file():
            for ln in index_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                self.index[r["idx"]] = (r["off"], r["n"])
        self._mm: np.ndarray | None = None

    def _map(self) -> np.ndarray:
        if self._mm is None:
            self._mm = np.memmap(self.root / "codes.i16", dtype=np.int16, mode="r").reshape(-1, N_CODEBOOKS)
        return self._mm

    def __contains__(self, idx: str) -> bool:
        return idx in self.index

    def __len__(self) -> int:
        return len(self.index)

    def get(self, idx: str) -> np.ndarray:
        off, n = self.index[idx]
        # A real copy, not just ascontiguousarray: the memmap is opened in mode
        # "r", so a contiguous slice stays a read-only view. torch.from_numpy()
        # on it rightly warns ("non-writable tensor ... undefined behavior").
        # The copy costs ~2 kB per item and makes ownership unambiguous.
        return np.array(self._map()[off:off + n], dtype=np.int16, copy=True)
