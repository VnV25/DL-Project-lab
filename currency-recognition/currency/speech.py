"""Spoken answers in English, Hindi and Marathi.

Uses Google Text-to-Speech (gTTS, needs internet, natural voices in all three languages).
If that fails (offline), falls back to pyttsx3, which uses your computer's built-in voices.
Numbers are written as digits; the Hindi and Marathi voices read them in their own language.
"""
import hashlib
import tempfile
import threading
from pathlib import Path

LANGUAGES = {"English": "en", "हिन्दी (Hindi)": "hi", "मराठी (Marathi)": "mr"}

PHRASES = {
    "en": {
        "note": "This is a {n} rupee note.",
        "uncertain": "I am not sure. Please take the photo again with the whole note visible.",
        "no_note": "No currency note found.",
        "added": "Added {n} rupees. Total is {t} rupees.",
        "total": "Total is {t} rupees.",
        "reset": "Total cleared.",
    },
    "hi": {
        "note": "यह {n} रुपये का नोट है।",
        "uncertain": "मुझे पक्का पता नहीं है। कृपया पूरा नोट दिखाकर फिर से फोटो लें।",
        "no_note": "कोई नोट नहीं मिला।",
        "added": "{n} रुपये जोड़े गए। कुल {t} रुपये।",
        "total": "कुल {t} रुपये।",
        "reset": "कुल रकम हटा दी गई।",
    },
    "mr": {
        "note": "ही {n} रुपयांची नोट आहे.",
        "uncertain": "मला खात्री नाही. कृपया संपूर्ण नोट दिसेल असा फोटो पुन्हा काढा.",
        "no_note": "कोणतीही नोट आढळली नाही.",
        "added": "{n} रुपये जोडले. एकूण {t} रुपये.",
        "total": "एकूण {t} रुपये.",
        "reset": "एकूण रक्कम काढून टाकली.",
    },
}

_CACHE = Path(tempfile.gettempdir()) / "currency_tts"


def phrase(key, lang="en", **kw):
    return PHRASES.get(lang, PHRASES["en"])[key].format(**kw)


def speak_to_file(text, lang="en"):
    """Create an audio file saying `text`. Returns its path, or None if no voice is available."""
    _CACHE.mkdir(exist_ok=True)
    key = hashlib.md5(f"{lang}|{text}".encode()).hexdigest()[:16]
    mp3 = _CACHE / f"{key}.mp3"
    if mp3.exists():
        return str(mp3)
    try:
        from gtts import gTTS
        gTTS(text=text, lang=lang).save(str(mp3))
        return str(mp3)
    except Exception:
        pass
    wav = _CACHE / f"{key}.wav"

    def offline():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, str(wav))
            engine.runAndWait()
        except Exception:
            pass

    # Some systems hang in the offline engine, so give it at most 10 seconds
    t = threading.Thread(target=offline, daemon=True)
    t.start()
    t.join(timeout=10)
    if wav.exists() and wav.stat().st_size > 0:
        return str(wav)
    return None
