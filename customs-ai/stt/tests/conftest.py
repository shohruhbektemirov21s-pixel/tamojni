"""stt/ ni sys.path ga qo'shadi (evaluate_wer importi uchun)."""
import sys
from pathlib import Path

STT = Path(__file__).resolve().parents[1]
if str(STT) not in sys.path:
    sys.path.insert(0, str(STT))
