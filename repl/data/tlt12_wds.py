"""PyTorch DataLoader for WebDataset-packaged 3D Latent Traversal shards.

Mirrors the TraversalDataset dict API so training code works unchanged:

    frames               FloatTensor  [T, 3, H, W]   values in [0, 1]
    latents              FloatTensor  [T, 10]
    base_latent          FloatTensor  [10]
    traversal_velocities FloatTensor  [10]            0 = frozen, ±v = sweep speed
    synset_id            str
    obj_id               str
    traversal_factors    list[int]                    indices of varying factors

Shards are produced by package_wds.py.

Usage:
    from wds_dataset import TraversalWebDataset, make_loader

    ds     = TraversalWebDataset("./shards/shard-{00000..00020}.tar")
    loader = make_loader(ds, batch_size=16, num_workers=4)

    for batch in loader:
        frames = batch["frames"]   # [B, T, 3, H, W]
        ...
"""

import glob as _glob
import io
import json
import math
import os

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision.transforms.functional import to_tensor


CIRCULAR_LATENT_INDICES = {2, 3, 5, 6}
N_SINCOS                = len(CIRCULAR_LATENT_INDICES) * 2
HUE_LATENT_INDICES      = {3, 6}   # floor_hue=3, spot_hue=6


# ── per-sample decode ─────────────────────────────────────────────────────────

def _decode_sample(raw: dict) -> dict:
    """Decode raw byte fields from a WebDataset sample."""
    out: dict = {"__key__": raw["__key__"]}
    for k, v in raw.items():
        if k == "__key__":
            continue
        if k.endswith(".jpg"):
            out[k] = to_tensor(Image.open(io.BytesIO(v)).convert("RGB"))  # [3, H, W]
        elif k.endswith(".npy"):
            out[k] = np.load(io.BytesIO(v))
        elif k.endswith(".json"):
            out[k] = json.loads(v if isinstance(v, str) else v.decode())
        else:
            out[k] = v
    return out


def _build_sequence(decoded: dict) -> dict:
    """Stack frame tensors and assemble the final per-sequence dict."""
    frame_keys = sorted(k for k in decoded if k.startswith("frame_") and k.endswith(".jpg"))
    frames = torch.stack([decoded[k] for k in frame_keys])  # [T, 3, H, W]
    meta   = decoded["meta.json"]
    return {
        "frames":               frames,
        "latents":              torch.from_numpy(decoded["latents.npy"]),
        "base_latent":          torch.from_numpy(decoded["base_latent.npy"]),
        "traversal_velocities": torch.from_numpy(decoded["velocities.npy"]),
        "synset_id":            meta["synset_id"],
        "obj_id":               meta["obj_id"],
        "traversal_factors":    meta["traversal_factors"],
    }


# ── picklable callables for use in worker processes ───────────────────────────

class _SampleProcessor:
    """Decode + optional per-frame transform.  Picklable (unlike a closure)."""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, raw: dict) -> dict:
        sample = _build_sequence(_decode_sample(raw))
        if self.transform is not None:
            sample["frames"] = torch.stack(
                [self.transform(sample["frames"][t])
                 for t in range(sample["frames"].shape[0])]
            )
        return sample


class _FactorFilter:
    def __init__(self, factor_idx: int):
        self._idx = factor_idx

    def __call__(self, sample: dict) -> bool:
        return self._idx in sample["traversal_factors"]


class _SynsetFilter:
    def __init__(self, synset_id: str):
        self._sid = synset_id

    def __call__(self, sample: dict) -> bool:
        return sample["synset_id"] == self._sid


# ── batching ──────────────────────────────────────────────────────────────────

def collate_sequences(samples: list) -> dict:
    """Collate a list of sequence dicts into a batch.

    Tensors are stacked along dim 0.  String fields and traversal_factors
    (variable-length lists) are kept as Python lists.
    """
    return {
        "frames":               torch.stack([s["frames"] for s in samples]),
        "latents":              torch.stack([s["latents"] for s in samples]),
        "base_latent":          torch.stack([s["base_latent"] for s in samples]),
        "traversal_velocities": torch.stack([s["traversal_velocities"] for s in samples]),
        "synset_id":            [s["synset_id"] for s in samples],
        "obj_id":               [s["obj_id"] for s in samples],
        "traversal_factors":    [s["traversal_factors"] for s in samples],
    }


# ── dataset info ──────────────────────────────────────────────────────────────

def load_dataset_info(shard_dir: str) -> dict:
    """Load global metadata from dataset_info.json written by package_wds.py."""
    info_path = os.path.join(shard_dir, "dataset_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"dataset_info.json not found in {shard_dir}")
    with open(info_path) as fh:
        info = json.load(fh)
    info["latent_ranges"] = torch.tensor(info["latent_ranges"], dtype=torch.float32)
    return info


# ── main class ────────────────────────────────────────────────────────────────

class TraversalWebDataset:
    """WebDataset pipeline builder for 3D Latent Traversal tar shards.

    This is not a PyTorch Dataset subclass (WebDataset pipelines are iterable,
    not indexed).  Call .dataset() or make_loader() to get something you can
    iterate.

    Args:
        urls:          Glob pattern, brace-expanded WebDataset URL string, or
                       list of shard paths.
                       Examples:
                         "./shards/shard-{00000..00020}.tar"  (brace expansion)
                         "./shards/shard-*.tar"               (glob)
                         ["./shards/shard-00000.tar", ...]    (list)
        info_dir:      Directory containing dataset_info.json.  Inferred as
                       the parent directory of the URL pattern when omitted.
        transform:     Optional callable applied to each frame tensor
                       [3, H, W] → [3, H, W] before stacking.
        shuffle:       In-epoch sample shuffle buffer size (0 = no shuffle).
        shardshuffle:  If True, shard order is shuffled between epochs.
    """

    def __init__(
        self,
        urls,
        info_dir: str = None,
        transform=None,
        shuffle: int = 1000,
        shardshuffle: int = 10,
    ):
        # Expand glob patterns to a concrete list so wds.WebDataset never
        # receives a wildcard string (it would try to open "*.tar" literally).
        if isinstance(urls, str) and "*" in urls:
            expanded = sorted(_glob.glob(urls))
            if not expanded:
                raise FileNotFoundError(f"No shard files found matching: {urls}")
            self._urls = expanded
        else:
            self._urls = urls

        self._transform    = transform
        self._shuffle      = shuffle
        # webdataset expects shardshuffle as a non-negative integer (buffer
        # size), not a bool.  Convert True → 10 shards, False/0 → 0.
        self._shardshuffle = 10 if shardshuffle is True else int(shardshuffle)

        if info_dir is None:
            if isinstance(self._urls, list):
                info_dir = os.path.dirname(self._urls[0]) if self._urls else "."
            else:
                info_dir = os.path.dirname(self._urls) or "."

        try:
            info = load_dataset_info(info_dir)
        except FileNotFoundError:
            info = {}

        self.latent_names  = info.get("latent_names", [])
        self.latent_ranges = info.get("latent_ranges", None)   # FloatTensor [N, 2]
        self.n_frames      = info.get("n_frames", None)
        self.image_size    = info.get("image_size", None)
        self.n_sequences   = info.get("n_sequences", None)
        self.synset_list     = info.get("unique_synset_ids", None)
        if self.synset_list is not None:
            self.synset_list = sorted(self.synset_list)

    # ── pipeline builders ─────────────────────────────────────────────────────

    def _base_pipeline(self):
        pipeline = wds.WebDataset(self._urls, shardshuffle=self._shardshuffle)
        if self._shuffle > 0:
            pipeline = pipeline.shuffle(self._shuffle)
        return pipeline.map(_SampleProcessor(self._transform))

    def dataset(self, batch_size: int = None) -> wds.WebDataset:
        """Return the full pipeline.

        Args:
            batch_size: If given, batching is done inside the pipeline with
                        collate_sequences.  Pass batch_size=None (default) when
                        you handle batching in WebLoader / DataLoader yourself.
        """
        pipeline = self._base_pipeline()
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

    def filter_by_factor(self, factor, batch_size: int = None) -> wds.WebDataset:
        """Return a pipeline restricted to sequences where factor varies.

        Args:
            factor:     int index (0–9) or str name, e.g. 'rot_x'.
            batch_size: Optional in-pipeline batching (see dataset()).
        """
        if isinstance(factor, str):
            factor = self.latent_names.index(factor)
        pipeline = self._base_pipeline().select(_FactorFilter(factor))
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

    def filter_by_synset(self, synset_id: str, batch_size: int = None) -> wds.WebDataset:
        """Return a pipeline restricted to a single ShapeNet synset."""
        if self.synset_list is not None and synset_id not in self.synset_list:
            raise ValueError(f"Synset ID {synset_id} is not in the dataset.")
        pipeline = self._base_pipeline().select(_SynsetFilter(synset_id))
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

    def factor_name(self, idx: int) -> str:
        return self.latent_names[idx]


def make_loader(
    source,
    batch_size: int = 16,
    num_workers: int = 4,
    **twd_kwargs,
) -> wds.WebLoader:
    """Build a DataLoader from a TraversalWebDataset, a pipeline, or shard URLs.

    Batching is moved inside the pipeline (via .batched()) so each worker
    yields complete batches.  This is the recommended WebDataset pattern for
    multi-worker loading and avoids deadlocks that arise when collation is
    done outside the pipeline.

    Args:
        source:       TraversalWebDataset instance, a wds pipeline, or a URL
                      string / list of paths.
        batch_size:   Sequences per batch.
        num_workers:  Worker processes for parallel shard reading.
        **twd_kwargs: Forwarded to TraversalWebDataset when source is a URL.
    """
    if isinstance(source, TraversalWebDataset):
        pipeline = source.dataset(batch_size=batch_size)
    elif isinstance(source, (str, list)):
        pipeline = TraversalWebDataset(source, **twd_kwargs).dataset(batch_size=batch_size)
    else:
        # Assume source is already a batched wds pipeline
        pipeline = source

    # batch_size=None: batching already happened inside the pipeline
    return wds.WebLoader(pipeline, num_workers=num_workers, batch_size=None)


def _normalize_latents(latents: torch.Tensor, latent_ranges: torch.Tensor) -> torch.Tensor:
    """Normalize latents to match _CachingSubsetDataset."""
    lo = latent_ranges[:, 0]
    hi = latent_ranges[:, 1]
    span = (hi - lo).clamp(min=1e-8)
    mask = (lo < 0) & (hi > 0)
    normalized = torch.empty_like(latents)
    normalized[:, mask] = (latents[:, mask] - lo[mask]) / span[mask] * 2 - 1
    normalized[:, ~mask] = (latents[:, ~mask] - lo[~mask]) / span[~mask]
    return normalized


def _compute_sincos(raw_latents: torch.Tensor) -> torch.Tensor:
    """Sin/cos encoding for circular latents."""
    t_steps = raw_latents.shape[0]
    sincos = torch.empty(t_steps, N_SINCOS, dtype=torch.float32)
    for output_idx, latent_idx in enumerate(CIRCULAR_LATENT_INDICES):
        angle = raw_latents[:, latent_idx]
        if latent_idx in HUE_LATENT_INDICES:
            angle = angle * (2 * math.pi)
        sincos[:, 2 * output_idx] = torch.sin(angle)
        sincos[:, 2 * output_idx + 1] = torch.cos(angle)
    return sincos


def _compute_quaternions(raw_latents: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles in raw_latents to quaternions."""
    rot_x = raw_latents[:, 0] / 2
    rot_y = raw_latents[:, 1] / 2
    rot_z = raw_latents[:, 2] / 2

    cx = torch.cos(rot_x)
    sx = torch.sin(rot_x)
    cy = torch.cos(rot_y)
    sy = torch.sin(rot_y)
    cz = torch.cos(rot_z)
    sz = torch.sin(rot_z)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    return torch.stack([qw, qx, qy, qz], dim=-1)


def _compute_deltas(raw_latents: torch.Tensor) -> torch.Tensor:
    """Compute per-frame deltas with circular wrapping, matching _CachingSubsetDataset."""
    deltas = raw_latents[1:] - raw_latents[:-1]
    for latent_idx in CIRCULAR_LATENT_INDICES:
        if latent_idx in HUE_LATENT_INDICES:
            deltas[:, latent_idx] = (deltas[:, latent_idx] + 0.5) % 1.0 - 0.5
        else:
            deltas[:, latent_idx] = (deltas[:, latent_idx] + math.pi) % (2 * math.pi) - math.pi
    return torch.cat([deltas[0].unsqueeze(0), deltas], dim=0)


def _build_label_tuple(sample: dict, latent_ranges: torch.Tensor, synset_to_idx: dict, normalize_latents=True) -> tuple:
    """Build the cached-subset label tuple for a decoded sample."""
    raw_latents = sample["latents"].to(dtype=torch.float32)

    seq_labels = torch.zeros(1, dtype=torch.long)
    seq_labels[0] = synset_to_idx[sample["synset_id"]]

    dense_reg_labels = torch.cat([
        _normalize_latents(raw_latents, latent_ranges) if normalize_latents else raw_latents,
        _compute_sincos(raw_latents),
        _compute_deltas(raw_latents),
        _compute_quaternions(raw_latents),
    ], dim=-1)

    return (
        seq_labels,
        torch.tensor([]),
        torch.tensor([]),
        dense_reg_labels,
        torch.tensor([]),
    )


def _collate_labeled_sequences(samples: list) -> tuple:
    """Collate labeled sequence samples into batched tensors."""
    frames = torch.stack([sample[0] for sample in samples])
    labels = tuple(torch.stack([sample[1][i] for sample in samples]) for i in range(5))
    return frames, labels


class _LabeledSampleProcessor:
    """Convert decoded WebDataset samples into (frames, labels) tuples."""

    def __init__(self, latent_ranges: torch.Tensor, synset_to_idx: dict, normalize_latents=True):
        self._latent_ranges = latent_ranges
        self._synset_to_idx = synset_to_idx
        self._normalize_latents = normalize_latents

    def __call__(self, sample: dict) -> tuple:
        labels = _build_label_tuple(sample, self._latent_ranges, self._synset_to_idx, self._normalize_latents)
        return sample["frames"], labels

class TraversalWebDatasetSubset(TraversalWebDataset):
    """Only loads n_shards from the full TraversalWebDataset.  Useful for offline evaluation."""
    def __init__(self, base: TraversalWebDataset, n_shards: int):
        self._base = base
        self.latent_names = base.latent_names
        self.latent_ranges = base.latent_ranges
        self.n_frames = base.n_frames
        self.image_size = base.image_size
        self.n_sequences = base.n_sequences
        self.synset_list = base.synset_list
        self._shardshuffle = base._shardshuffle
        self._shuffle = base._shuffle
        self._transform = base._transform

        if n_shards > len(base._urls):
            raise ValueError(f"Requested {n_shards} shards, but only {len(base._urls)} available.")
        
        self._urls = base._urls[:n_shards]


class LabeledTraversalWebDataset:
    """Wrapper that adds _CachingSubset-style labels to TraversalWebDataset samples."""

    def __init__(self, base: TraversalWebDataset, normalize_latents=True):
        self._base = base
        self.latent_names = base.latent_names
        self.latent_ranges = base.latent_ranges
        self.n_frames = base.n_frames
        self.image_size = base.image_size
        self.n_sequences = base.n_sequences
        self.synset_list = base.synset_list
        if self.latent_ranges is None:
            raise ValueError("TraversalWebDataset label wrapper requires latent_ranges metadata")
        
        self.synset_to_idx = {synset: i for i, synset in enumerate(self.synset_list)}
        self._normalize_latents = normalize_latents

    def __getattr__(self, name):
        return getattr(self._base, name)
    
    def __len__(self):
        return self._base.n_sequences

    def _base_pipeline(self):
        return self._base._base_pipeline()

    def dataset(self, batch_size: int = None) -> wds.WebDataset:
        pipeline = self._base_pipeline().map(
            _LabeledSampleProcessor(self.latent_ranges, self.synset_to_idx, self._normalize_latents)
        )
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size,
                                        collation_fn=_collate_labeled_sequences,
                                        partial=False)
            pipeline = pipeline.with_length(self._base.n_sequences // batch_size)
        return pipeline

    def filter_by_factor(self, factor, batch_size: int = None) -> wds.WebDataset:
        if isinstance(factor, str):
            factor = self.latent_names.index(factor)
        pipeline = self._base_pipeline().select(_FactorFilter(factor)).map(
            _LabeledSampleProcessor(self.latent_ranges, self.synset_to_idx, self._normalize_latents)
        )
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=_collate_labeled_sequences, partial=False)
            pipeline = pipeline.with_length(self._base.n_sequences // batch_size)
        return pipeline

    def filter_by_synset(self, synset_id: str, batch_size: int = None) -> wds.WebDataset:
        pipeline = self._base_pipeline().select(_SynsetFilter(synset_id)).map(
            _LabeledSampleProcessor(self.latent_ranges, self.synset_to_idx, self._normalize_latents)
        )
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=_collate_labeled_sequences, partial=False)
            pipeline = pipeline.with_length(self._base.n_sequences // batch_size)
        return pipeline

    def factor_name(self, idx: int) -> str:
        return self.latent_names[idx]

    def loader(
        self,
        batch_size: int,
        num_workers: int = 0,
    ):
        length = self._base.n_sequences // batch_size
        pipeline = self.dataset(batch_size=batch_size)
        loader = wds.WebLoader(
            pipeline,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        return loader.with_length(length)
