# MannMitra — Unified Multimodal Student Mental-Health Triage

MannMitra combines two previously separate repositories into one
production-grade Streamlit application:

1. **Text, Triage & RAG**
   — GoEmotions sentiment analysis with a regex safety-net for crisis
   language, multi-turn distress compounding (GREEN / YELLOW / RED), and a
   persistent ChromaDB knowledge base of coping techniques and campus
   resources.
2. **Speech Emotion & Voice Processing**
   — Whisper-Tiny transcription, a fine-tuned Wav2Vec2 acoustic classifier
   (with graceful, transcript-only fallback if the fine-tuned weights are
   absent), and a transparent non-clinical stress-scoring heuristic.
           


## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in GEMINI_API_KEY
```

MP3 uploads use `librosa`/`audioread` as a fallback decoder, which requires
`ffmpeg` to be available on your system `PATH`. WAV recordings (the default
from `st.audio_input`) do not need it.

## Running

```bash
streamlit run app.py
```

On first launch:
- `./docs` is auto-seeded with default coping/resource documents if empty.
- `./chroma_db` is created and indexed automatically.
- If no fine-tuned acoustic weights are found under
  `speech_engine/models/ravdess_wav2vec2_finetuned/best/` and
  `HUGGINGFACE_MODEL_ID` is not set, voice messages still work: the app
  transparently falls back to a transcript-only (Whisper + GoEmotions/text
  cue) emotion estimate instead of erroring.

## Triage guardrails

- **Tier 0 (RED, crisis regex or score ≥ 0.75):** the LLM is **bypassed
  entirely**. The user receives the verified Tele-MANAS (`14416`) and KIRAN
  (`1800-599-0019`) helpline response immediately.
- **GREEN / YELLOW:** conversation history and any retrieved RAG context are
  passed to Gemini (`gemini-2.5-flash` by default; override with
  `GEMINI_MODEL`), which is instructed to safety threshold `BLOCK_NONE`
  across all four harm categories so that ordinary expressions of stress,
  fatigue, or frustration are never filtered before triage sees them.

## Notes

- This is peer-support tooling, **not** a diagnostic or clinical system.
  Stress scores and emotion labels from both the text and speech pipelines
  are explicitly framed as non-clinical heuristics.
- `app.py` resolves `./speech_engine` via `importlib` at runtime (with a
  `TYPE_CHECKING`-only static import for editor hints), so IDEs without this
  project's root configured won't raise spurious `reportMissingImports`
  warnings, while the app still works correctly when actually run.
