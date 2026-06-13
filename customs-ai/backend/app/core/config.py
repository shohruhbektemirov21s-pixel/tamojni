"""Konfiguratsiya yuklash — env (CUSTOMS_*) + YAML.

risk_rules.yaml / taxonomy.yaml ni Dev 2 to'ldiradi; yuklash mexanizmi shu yerda.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ ildizi (app/core/config.py -> parents[2])
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CUSTOMS_", env_file=".env", extra="ignore")

    # --- tarmoq / bind (Faza 1 auth = faqat loopback) ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- yo'llar ---
    data_dir: Path = BASE_DIR / "data"
    config_dir: Path = BASE_DIR / "config"

    # --- provayderlar ---
    # Dev 2/3/4 real modellarni yetkazguncha mock bilan ishlaymiz.
    use_mocks: bool = True

    # --- Vision / YOLO detektor (ADR-002: ONNX, CPU/DirectML) ---
    # Bo'sh bo'lsa yoki fayl topilmasa -> mock detector'ga xavfsiz fallback
    # (jamoa parallel ishlasin; domain-fine-tuned model kelguncha bloklanmasin).
    yolo_model_path: Path | None = None        # masalan: model_registry/v1/model.onnx
    yolo_labels_path: Path | None = None       # bo'sh bo'lsa model yonidagi labels.txt
    yolo_conf: float = 0.20                     # PAST (recall foydasiga)
    yolo_iou: float = 0.50
    yolo_imgsz: int = 640

    # --- GPU orchestrator (ADR-001/002) ---
    # manage_ollama=False bo'lsa, orchestrator tashqi/qo'lda ishga tushgan Ollama'ni
    # boshqarmaydi (test/dev uchun). True bo'lsa, daemon'ni o'zi start/stop qiladi.
    manage_ollama: bool = False
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    ollama_binary: str = "ollama"

    # --- LLM sintez / tushuntirish (Dev 4, Qwen3 4B Q4_K_M) ---
    # use_mocks=False bo'lганda real OllamaExplainer ulanadi (llm_enabled=True).
    # llm_enabled=False -> MockExplainer (jamoa LLM'siz parallel ishlasin).
    llm_enabled: bool = True
    llm_model: str = "qwen3:4b"
    # keep_alive: model VRAM'da qancha "issiq" qoladi. 4GB + MAX_LOADED_MODELS=1
    # da "5m" xavfsiz; OOM xavfini sezsangiz "0" (har case'dan keyin bo'shatish).
    llm_keep_alive: str = "5m"
    llm_num_ctx: int = 4096
    llm_num_predict: int = 512        # tushuntirish qisqa -> chiqish uzunligini chekla
    llm_temperature: float = 0.0      # grounded/reproduksiya (anti-hallucination)

    # --- STT / Whisper (Dev 3, ADR-002: CPU/int8) ---
    # use_mocks=False bo'lganda real WhisperTranscriber ulanadi; faster-whisper
    # yoki model yo'q bo'lsa -> mock'ga xavfsiz fallback (jamoa bloklanmaydi).
    stt_model_size: str = "medium"             # MVP; o'zbek uchun fine-tune keyin
    stt_model_path: Path | None = None         # lokal model katalogi (100% offline)
    stt_device: str = "cpu"                     # GPU Ollama'niki (ADR-002)
    stt_compute_type: str = "int8"
    stt_language: str | None = None            # None -> auto-detect (ru/uz)
    stt_beam_size: int = 5
    stt_vad: bool = True                        # Silero VAD bilan sukunatni kesish
    stt_cpu_threads: int = 0                    # 0 -> ct2 o'zi tanlaydi
    stt_download_root: Path | None = None       # oldindan yuklangan kesh (offline)

    # --- degradatsiya parametrlari (§10) ---
    stt_timeout_s: float = 30.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 1
    min_free_disk_mb: int = 500

    # --- navbat / event bus ---
    queue_maxsize: int = 100
    # WS subscriber'ning bounded navbati — sekin client pipeline'ni ushlamasligi
    # uchun to'lganda eng eski event tashlanadi (drop-oldest).
    event_subscriber_buffer: int = 1000

    # --- Tier strategiyasi (sozlanadigan; Tamoyil 7: GPU bottleneck'dan qochish) ---
    # Tier 1 (detect+risk) HAR DOIM avtomatik, <1s push. LLM sintez (Tier 2, GPU)
    # esa qimmat — default'da FAQAT on-demand (POST /cases/{id}/explain) ishlaydi.
    # llm_auto_on_high=True -> qo'shimcha: HIGH risk case'da avtomatik LLM stream.
    # llm_auto_always=True  -> har case'ga avtomatik LLM (eski xulq; GPU bottleneck,
    #                          tavsiya etilmaydi). always=True on_high'ni qamrab oladi.
    llm_auto_on_high: bool = True
    llm_auto_always: bool = False

    # ---- hosil qilinadigan yo'llar ----
    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "customs.db"

    @property
    def db_url(self) -> str:
        # ADR-006: SQLite (WAL) Faza 1; ORM orqali Postgres'ga ko'chadi.
        return f"sqlite:///{self.db_path}"


@lru_cache
def get_settings_singleton() -> Settings:
    """Process bo'yicha bitta Settings (uvicorn entrypoint uchun).

    Testlar Settings'ni to'g'ridan-to'g'ri yasab create_app(settings=...) ga
    beradi — bu singleton'dan foydalanmaydi.
    """
    return Settings()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Reference risk konfiguratsiyasi (risk_rules.yaml bo'sh/yo'q bo'lsa fallback).
# Haqiqiy qiymatlarni Dev 2 config/risk_rules.yaml da belgilaydi.
_DEFAULT_RISK_CONFIG: dict = {
    "default_weight": 0.2,
    "class_weights": {
        "qurol": 1.0,
        "o'q-dori": 1.0,
        "narkotik_paket": 1.0,
        "pichoq": 0.8,
        "valyuta_toplami": 0.7,
        "organik_anomaliya": 0.5,
        "yashirilgan_boshliq": 0.6,
    },
    "thresholds": {"HIGH": 0.7, "MEDIUM": 0.4},
    # recall-favoring default'lar (yaml bo'lmasa ham xavfsiz):
    "min_confidence": 0.25,
    "per_class_min_confidence": {
        "qurol": 0.10,
        "o'q-dori": 0.10,
        "narkotik_paket": 0.12,
        "pichoq": 0.15,
    },
    "prohibited_classes": [],
}


def load_risk_config(settings: Settings) -> dict:
    """risk_rules.yaml ni yuklaydi, yo'q bo'lsa xavfsiz default qaytaradi."""
    cfg = _load_yaml(settings.config_dir / "risk_rules.yaml")
    if not cfg:
        return dict(_DEFAULT_RISK_CONFIG)
    # default'lar bilan birlashtirish (yarim to'ldirilgan yaml uchun)
    merged = dict(_DEFAULT_RISK_CONFIG)
    merged.update(cfg)
    return merged


def load_taxonomy(settings: Settings) -> dict:
    return _load_yaml(settings.config_dir / "taxonomy.yaml")
