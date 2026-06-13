"""ml/ ni sys.path ga qo'shadi (evaluate, sync_labels importi uchun)."""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))
