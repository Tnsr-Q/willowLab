# willowlab/ingest/cache_dir.py
import os, json, hashlib, pathlib, time
from typing import Dict, Any, Optional

DEFAULT_CACHE_NAME = "ingest_cache.json"

def _hash_file(path: str, block_size: int=1024*1024)->str:
    """SHA256 of file contents, streamed (works for big blobs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

class DirCache:
    """
    Directory-level cache:
    - remembers file SHA256 and size/mtime
    - provides 'should_process' decision
    - records arbitrary ingest metadata (e.g., where we saved the Willow bundle)
    """
    def __init__(self, root_dir: str, cache_path: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.cache_path = cache_path or os.path.join(self.root_dir, DEFAULT_CACHE_NAME)
        self._db: Dict[str, Any] = {}
        if os.path.exists(self.cache_path):
            try:
                self._db = json.load(open(self.cache_path, "r"))
            except Exception:
                self._db = {}

    def file_record(self, path: str) -> Optional[Dict[str, Any]]:
        key = os.path.relpath(os.path.abspath(path), self.root_dir)
        return self._db.get(key)

    def should_process(self, path: str) -> bool:
        """Skip if same hash & size persisted; true means reprocess."""
        abspath = os.path.abspath(path)
        key = os.path.relpath(abspath, self.root_dir)
        size = os.path.getsize(abspath)
        mtime = _mtime(abspath)
        rec = self._db.get(key)
        if not rec:
            return True
        # Fast path: if size or mtime changed, re-hash and reprocess.
        if rec.get("size") != size or abs(rec.get("mtime", 0.0) - mtime) > 1e-6:
            return True
        # No change in size/mtime -> trust it as unchanged (cheap).
        # If you want belt+braces, re-hash here.
        return False

    def update(self, path: str, **ingest_meta):
        abspath = os.path.abspath(path)
        key = os.path.relpath(abspath, self.root_dir)
        size = os.path.getsize(abspath)
        mtime = _mtime(abspath)
        # Hash only when we actually processed; keeps idle runs fast.
        sha = _hash_file(abspath)
        self._db[key] = {"size": size, "mtime": mtime, "sha256": sha, "ingest_meta": ingest_meta, "updated": time.time()}

    def persist(self):
        pathlib.Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._db, f, indent=2, sort_keys=True)
