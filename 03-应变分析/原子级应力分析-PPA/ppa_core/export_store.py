"""Stage CSV plus a checksum-bound sidecar, restoring prior files on failure."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile


class CsvExport:
    def __init__(self, target):
        self.target = Path(target)
        self.sidecar = self.target.with_suffix('.metadata.json')
        self.temporary = []
        for path in (self.target, self.sidecar):
            if path.exists() and not path.is_file():
                raise OSError(f"Export target is not a file: {path}")
        self.csv_path = self._temp()

    def _temp(self):
        with NamedTemporaryFile(dir=self.target.parent, suffix='.tmp', delete=False) as handle:
            name = Path(handle.name)
        self.temporary.append(name)
        return name

    def abort(self):
        for path in self.temporary:
            path.unlink(missing_ok=True)
        self.temporary.clear()

    def commit(self, metadata):
        previous, installed = {}, []
        try:
            metadata = dict(metadata)
            metadata['csv_sha256'] = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()
            meta_temp = self._temp()
            meta_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            for target in (self.target, self.sidecar):
                backup = self._temp() if target.exists() else None
                if backup is not None:
                    shutil.copy2(target, backup)
                previous[target] = backup
            for pending, target in ((self.csv_path, self.target), (meta_temp, self.sidecar)):
                os.replace(pending, target)
                installed.append(target)
        except Exception:
            for target in reversed(installed):
                backup = previous[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
        finally:
            self.abort()
