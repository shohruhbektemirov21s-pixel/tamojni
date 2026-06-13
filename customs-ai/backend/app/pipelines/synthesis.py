"""LLM sintez (tushuntirish) — EGASI: Dev 4.

Backend Core `Explainer` kontraktini chaqiradi. Qwen3 4B (Ollama, GGUF Q4_K_M).

⚠️ BUZILMAS TAMOYIL (1/2, ADR-004): LLM FAQAT inson tilidagi tushuntirish matnini
yozadi; risk score'ni HISOBLAMAYDI va O'ZGARTIRMAYDI. Risk allaqachon
deterministik hisoblangan (`scoring.py`) va `risk` argumenti sifatida beriladi.
Tushuntirish berilgan darajani (`level`/`score`) FAQAT inson tiliga "tarjima"
qiladi — u bilan bahslashmaydi, uni qayta baholamaydi. Bu huquqiy zaruriyat:
score sudda himoya qilinishi kerak, LLM gallyutsinatsiyasiga bog'liq emas.

GROUNDED (Tamoyil 3): tushuntirish FAQAT berilgan detection/transcript/risk
faktlariga asoslanadi. Yangi "fakt" o'ylab topilmaydi (temperature=0 + constrained
JSON output + grounded prompt). Eval: `tests/test_grounding_eval.py`.

DEGRADATSIYA (Tamoyil 6, §10) — jamoa idiomi (`speech.py` bilan bir xil):
QATTIQ xatolikda (Ollama o'chiq, HTTP 5xx/OOM, schema fail) bu yerda EXCEPTION
ko'tariladi. Worker (`core/worker.py::_run_synthesis`) uni `gpu_session()` ichida,
timeout + retry(+daemon restart) bilan o'raydi; oxir-oqibat tutib MODEL_FAILED
audit yozadi va `{"text": None, "generated_by": None, "available": False}`
qaytaradi. Shu tarzda butun case TO'XTAMAYDI — risk LLM'dan OLDIN hisoblangani
uchun case to'liq risk bilan yakunlanadi.

Xato qatlamlari (qaysi qatlam tuzatadi):
    * Infratuzilma (ConnectError, HTTP 5xx, ehtimoliy OOM) -> RAISE. Worker
      daemon'ni RESTART qiladi (ADR-001: VRAM kafolatli tozalash) va qayta uradi.
    * Generatsiya (JSON parse / schema invalid / bo'sh matn) -> ichki RE-PROMPT
      1x; baribir yaroqsiz bo'lsa -> RAISE (restart yordam bermaydi, lekin worker
      degradatsiyani yagona joyda boshqaradi -> available=False).

GPU egasi: Ollama daemon (ADR-002). base_url'ni Backend beradi
(`GpuOrchestrator.base_url`) — loopback, tashqi tarmoq YO'Q (offline by design).
"""
from __future__ import annotations

import json
import logging
import re

import httpx
from jsonschema import ValidationError, validate

log = logging.getLogger("customs.synthesis")

# Kontrakt (§7.1): explanation.generated_by shu qiymatda bo'ladi (model id emas,
# barqaror "logical" nom — audit/case result uchun).
GENERATED_BY = "qwen3-4b"

# Ollama strukturali chiqishi `format` ga beriladigan JSON schema VA jsonschema
# bilan mahalliy validatsiya. Minimal: bitta inson tilidagi matn maydoni.
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": ["text"],
    "additionalProperties": False,
}

# Qo'llab-quvvatlanadigan chiqish tillari (transcript tilidan tanlanadi).
_SUPPORTED_LANGS = ("uz", "ru")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _empty_result() -> dict:
    """Degradatsiya kontrakti (§7.1). Worker odatda exception'ni tutib O'ZI shunga
    o'xshash dict yasaydi; bu — to'g'ridan-to'g'ri chaqiruvlar uchun xavfsiz shakl."""
    return {"text": "", "generated_by": GENERATED_BY, "available": False}


# ---------------------------------------------------------------------------
# Grounding yordamchilari — risk faktlarini DETERMINISTIK inson tiliga tarjima.
# (LLM'ga toza material beradi; LLM yangi fakt qo'shmasligi uchun.)
# ---------------------------------------------------------------------------
_CONF_WORDS = {
    "uz": [(0.85, "juda yuqori ishonch"), (0.60, "yuqori ishonch"),
           (0.40, "o'rtacha ishonch"), (0.0, "past ishonch")],
    "ru": [(0.85, "очень высокая уверенность"), (0.60, "высокая уверенность"),
           (0.40, "средняя уверенность"), (0.0, "низкая уверенность")],
}
_LEVEL_WORDS = {
    "uz": {"HIGH": "YUQORI", "MEDIUM": "O'RTA", "LOW": "PAST"},
    "ru": {"HIGH": "ВЫСОКИЙ", "MEDIUM": "СРЕДНИЙ", "LOW": "НИЗКИЙ"},
}


def _confidence_word(conf: float, lang: str) -> str:
    for floor, word in _CONF_WORDS[lang]:
        if conf >= floor:
            return word
    return _CONF_WORDS[lang][-1][1]


def _pick_language(transcript: dict | None) -> str:
    """Chiqish tili: transcript tilidan (ru/uz). Aniqlanmasa -> 'uz' (default)."""
    lang = (transcript or {}).get("language")
    return lang if lang in _SUPPORTED_LANGS else "uz"


def _format_facts(
    detections: list[dict],
    transcript: dict | None,
    operator_notes: str | None,
    risk: dict,
    lang: str,
) -> str:
    """Barcha signallarni yagona, tartibli FAKTLAR bloki sifatida yozadi.

    Bu LLM uchun YAGONA haqiqat manbai — prompt unga "faqat shu bloydagi faktlardan
    foydalan" deb buyuradi (anti-hallucination). Risk darajasi/score bu yerda
    BERILGAN; LLM uni o'zgartirmaydi.
    """
    level = str(risk.get("level", "LOW")).upper()
    score = risk.get("score", 0.0)
    level_word = _LEVEL_WORDS[lang].get(level, level)
    lines: list[str] = []

    if lang == "ru":
        lines.append(f"РИСК (рассчитан детерминированно, НЕ изменять): "
                     f"уровень {level_word}, балл {score}.")
    else:
        lines.append(f"RISK (deterministik hisoblangan, O'ZGARTIRMA): "
                     f"daraja {level_word}, ball {score}.")

    # Risk drayverlari: faqat hissa qo'shgan faktorlar (contribution > 0).
    drivers = [f for f in (risk.get("factors") or []) if f.get("contribution", 0) > 0]
    ignored = [f for f in (risk.get("factors") or [])
               if f.get("rule") == "below_confidence_floor"]
    hdr = "Драйверы риска:" if lang == "ru" else "Risk omillari:"
    if drivers:
        lines.append(hdr)
        for f in drivers:
            cw = _confidence_word(float(f.get("confidence", 0.0)), lang)
            lines.append(f"  - {f.get('class')}: {cw} "
                         f"({'уверенность' if lang == 'ru' else 'ishonch'} "
                         f"{f.get('confidence')}).")
    else:
        lines.append("Драйверов риска нет." if lang == "ru" else "Risk omillari yo'q.")
    if ignored:
        note = ("Ниже порога уверенности (в балл НЕ вошли): "
                if lang == "ru" else "Ishonch pollidan past (ballga KIRMAGAN): ")
        lines.append(note + ", ".join(str(f.get("class")) for f in ignored) + ".")

    # Detection'lar (xom signal — risk omillari bilan mos, lekin to'liqroq).
    dets = detections or []
    if dets:
        dhdr = "Обнаруженные объекты:" if lang == "ru" else "Aniqlangan obyektlar:"
        lines.append(dhdr)
        for d in dets:
            lines.append(f"  - {d.get('class')} "
                         f"({'уверенность' if lang == 'ru' else 'ishonch'} "
                         f"{d.get('confidence')}).")
    else:
        lines.append("Объекты не обнаружены." if lang == "ru"
                     else "Obyekt aniqlanmadi.")

    # Transcript (mavjud bo'lsa).
    t = transcript or {}
    ttext = (t.get("text") or "").strip() if t.get("available") else ""
    if ttext:
        thdr = "Аудио-транскрипт:" if lang == "ru" else "Audio transkript:"
        lines.append(f"{thdr} «{ttext}»")
    else:
        lines.append("Аудио-транскрипт отсутствует." if lang == "ru"
                     else "Audio transkript yo'q.")

    # Operator izohi.
    notes = (operator_notes or "").strip()
    if notes:
        ohdr = "Заметка оператора:" if lang == "ru" else "Operator izohi:"
        lines.append(f"{ohdr} «{notes}»")

    return "\n".join(lines)


def _system_prompt(lang: str) -> str:
    if lang == "ru":
        return (
            "Ты — ассистент таможенного контроля. Твоя ЕДИНСТВЕННАЯ задача — на "
            "основе предоставленных фактов написать для оператора короткое, ясное "
            "объяснение НА РУССКОМ языке: почему присвоен данный уровень риска.\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            "1. Уровень и балл риска УЖЕ рассчитаны детерминированно. НЕ меняй их, "
            "НЕ оспаривай, НЕ предлагай другой уровень. Только объясни данный.\n"
            "2. Используй ТОЛЬКО факты из блока ФАКТЫ. НЕ придумывай объекты, "
            "вещества, числа или выводы, которых там нет.\n"
            "3. Ты НЕ принимаешь решение (досмотр/пропуск) — решает оператор. "
            "Можешь рекомендовать внимание, но не приказывай.\n"
            "4. Кратко: 2–5 предложений. Технические сигналы переводи на понятный "
            "человеку язык (напр. «класс qurol 0.91» -> «объект, похожий на оружие, "
            "с высокой уверенностью»).\n"
            "5. Верни ТОЛЬКО JSON: {\"text\": \"<объяснение>\"}."
        )
    return (
        "Sen — bojxona nazorati yordamchisisan. Yagona vazifang: berilgan faktlar "
        "asosida operator uchun qisqa, aniq tushuntirish yozish (O'ZBEK tilida): "
        "nega aynan shu risk darajasi berilgan.\n"
        "QAT'IY QOIDALAR:\n"
        "1. Risk darajasi va ball ALLAQACHON deterministik hisoblangan. Ularni "
        "O'ZGARTIRMA, bahslashma, boshqa daraja taklif qilma. Faqat berilganini "
        "tushuntir.\n"
        "2. FAQAT FAKTLAR blokidagi ma'lumotlardan foydalan. U yerda yo'q obyekt, "
        "modda, raqam yoki xulosani O'YLAB TOPMA.\n"
        "3. Sen qaror qabul qilMAYSAN (ko'rik/o'tkazish) — qaror operatorda. "
        "Diqqatni tavsiya qilishing mumkin, lekin buyruq berma.\n"
        "4. Qisqa: 2–5 jumla. Texnik signallarni inson tiliga tarjima qil (masalan "
        "«qurol class 0.91» -> «yuqori ishonch bilan qurolga o'xshash obyekt»).\n"
        "5. FAQAT JSON qaytar: {\"text\": \"<tushuntirish>\"}."
    )


def _parse_content(content: str) -> dict:
    """Model javobidan JSON ob'ektni ajratib oladi. `format` schema bilan bu odatda
    toza JSON, lekin <think> yoki matn oqib chiqsa ham bardosh beramiz."""
    cleaned = _THINK_RE.sub("", content).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(cleaned)
        if m:
            return json.loads(m.group(0))  # oxirgi imkon; fail bo'lsa chaqiruvchi tutadi
        raise


class OllamaExplainer:
    """Qwen3 4B (Ollama) — FAQAT tushuntirish sintezi (Explainer kontrakti).

    Lazy/holatsiz: konstruktor tarmoqqa chiqMAYDI (Ollama'ni faqat
    `generate_explanation` chaqirganda uradi). Har chaqiruv mustaqil.

    Parametrlar:
        base_url: Ollama loopback manzili (Backend `orchestrator.base_url` beradi).
        model: Ollama model tegi (GGUF Q4_K_M — `qwen3:4b`).
        keep_alive: model VRAM'da qancha "issiq" qoladi (keyingi case tezroq).
            4GB'da ehtiyot: OLLAMA_MAX_LOADED_MODELS=1 + bitta model -> "5m" xavfsiz.
        request_timeout_s: httpx HTTP timeout. Worker yana wait_for bilan o'raydi;
            bu ip (thread) cheksiz osilib qolmasligi uchun backstop.
        num_ctx/num_predict/temperature: grounding uchun temp=0 (+seed) determinizm.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "qwen3:4b",
        *,
        keep_alive: str = "5m",
        request_timeout_s: float = 120.0,
        num_ctx: int = 4096,
        num_predict: int = 512,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.request_timeout_s = request_timeout_s
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        # Faqat test uchun: httpx MockTransport ineksiyasi (jonli Ollama'siz unit-test).
        self._transport = transport

    # ---- Ollama chaqiruvi (bitta urinish) ----
    def _chat(self, messages: list[dict]) -> str:
        """Ollama /api/chat (stream=false, format=schema). Model javob matnini
        (message.content) qaytaradi. Infratuzilma xatosi -> EXCEPTION (worker
        daemon'ni restart qilib qayta uradi)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": OUTPUT_SCHEMA,   # constrained decoding -> schema-mos JSON
            "think": False,            # Qwen3 reasoning'ini o'chir (tezlik + toza JSON)
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "seed": 0,             # temp=0 bilan birga -> reproduksiyalanadigan eval
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        with httpx.Client(timeout=self.request_timeout_s, transport=self._transport) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()  # 5xx/OOM -> HTTPStatusError -> worker restart
            data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""

    # ---- KONTRAKT ----
    def generate_explanation(
        self,
        detections: list[dict],
        transcript: dict,
        operator_notes: str | None,
        risk: dict,
    ) -> dict:
        """-> {"text": str, "generated_by": "qwen3-4b", "available": bool}

        Risk'ni O'ZGARTIRMAYDI — faqat tushuntiradi (Tamoyil 1/2). Qattiq xatoda
        (infra yoki schema 2x fail) EXCEPTION ko'taradi; worker degradatsiya qiladi.
        """
        lang = _pick_language(transcript)
        facts = _format_facts(detections, transcript, operator_notes, risk, lang)
        messages = [
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": f"FAKTLAR / ФАКТЫ:\n{facts}"},
        ]

        last_exc: Exception | None = None
        # Schema/parse fail -> 1x re-prompt (qat'iyroq eslatma bilan). Infra xato
        # bu sikldan TASHQARI (darhol propagate -> worker restart).
        for attempt in range(2):
            content = self._chat(messages)  # infra xato -> raise (tutilmaydi)
            try:
                parsed = _parse_content(content)
                validate(instance=parsed, schema=OUTPUT_SCHEMA)
                text = parsed["text"].strip()
                if not text:
                    raise ValidationError("bo'sh 'text'")
                if attempt > 0:
                    log.info("synthesis: re-prompt'dan keyin schema-valid (case)")
                return {"text": text, "generated_by": GENERATED_BY, "available": True}
            except (ValidationError, json.JSONDecodeError, KeyError, AttributeError) as exc:
                last_exc = exc
                log.warning(
                    "synthesis: schema/parse fail (urinish %d/2): %s | xom=%r",
                    attempt + 1, exc, content[:200],
                )
                # Re-prompt: modelга nima noto'g'ri ketganini ayt.
                messages.append({"role": "assistant", "content": content[:500]})
                messages.append({
                    "role": "user",
                    "content": (
                        "Javob faqat {\"text\": \"...\"} ko'rinishidagi yaroqli JSON "
                        "bo'lishi shart, boshqa hech narsa qo'shma. Qayta yoz."
                        if lang == "uz" else
                        "Ответ должен быть только валидным JSON вида {\"text\": \"...\"}, "
                        "без чего-либо ещё. Перепиши."
                    ),
                })

        # 2x schema fail: restart yordam bermaydi -> raise (worker degradatsiya qiladi).
        raise RuntimeError(
            f"LLM sintez schema validatsiyasi 2 urinishda muvaffaqiyatsiz: {last_exc}"
        )
