
import os
from pathlib import Path
from diskcache import Cache

# Use a local .cache directory in the project root
# This path is relative to where the app is run (usually backend/)
CACHE_DIR = Path(os.getcwd()) / ".cache" / "web_search"

# Initialize disk cache
# Size limit: 1GB
# Eviction policy: Least Recently Used (LRU)
cache = Cache(directory=str(CACHE_DIR), size_limit=1024 * 1024 * 1024)
