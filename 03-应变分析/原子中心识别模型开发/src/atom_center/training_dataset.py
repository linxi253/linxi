"""Training-only dataset; module-level class supports Windows worker spawning."""
from ultralytics.data.dataset import YOLODataset


class VerifiedDataset(YOLODataset):
    def _load_or_scan_cache(self, cache_path, cache_hash):
        # Upstream cache hashes cover paths and sizes, not actual label bytes.
        # Always scan the verified source files, then replace the derived cache.
        return self.cache_labels(cache_path), False
