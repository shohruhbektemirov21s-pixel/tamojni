"""Scanner ingestion adapterlari — papka/SDK/tarmoq -> avtomatik case.

Pluggable: ScannerSource (ABC) ni implementatsiya qilib yangi manba ulanadi.
Default: WatchedFolderSource (papkani watchdog bilan kuzatadi).
"""
from app.ingestion.base import ScannerSource
from app.ingestion.watched_folder import WatchedFolderSource

__all__ = ["ScannerSource", "WatchedFolderSource"]
