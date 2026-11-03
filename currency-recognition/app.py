"""Step 4: the web app.

Take a photo with the webcam (or upload one), and the app shows and speaks the value of the note.
"Add to total" keeps a running sum, so you can count a handful of notes one by one.

Usage:  python app.py            (opens in your browser)
        python app.py --share    (also gives a public https link, so you can use your phone's camera)
"""
import argparse
import shutil
import tempfile
import uuid
from pathlib import Path

import gradio as gr

import config
from currency.predictor import Predictor
from currency.speech import LANGUAGES, phrase, speak_to_file

# Card colours roughly follow the colour of each note
NOTE_COLORS = {
    "10": ("#8a5a2b", "#ffffff"), "20": ("#b5bd2a", "#1b1b1b"), "50": ("#1fa2bf", "#ffffff"),
    "100": ("#8a78c2", "#ffffff"), "200": ("#f2b705", "#1b1b1b"), "500": ("#69706b", "#ffffff"),
    "2000": ("#c2185b", "#ffffff"),
}
EMPTY_STATE = {"last": None, "notes": []}
AUDIO_DIR = Path(tempfile.gettempdir()) / "currency_app_audio"

try:
    PRED = Predictor()
    LOAD_ERROR = None
except FileNotFoundError as e:
    PRED, LOAD_ERROR = None, str(e)


def card(big, small, bg="#e8edf5", fg="#1b1b1b"):
    return (f'<div class="result-card" style="background:{bg};color:{fg}">'
            f'<div class="big">{big}</div><div class="small">{small}</div></div>')


def total_card(notes):
    if not notes:
        return card("₹0", "Running total: identify a note, then press <b>Add to total</b>", "#f1f3f6")
    parts = " + ".join(f"₹{n}" for n in notes)
    return card(f"₹{sum(notes)}", f"{len(notes)} note{'s' if len(notes) != 1 else ''}: {parts}", "#e3f4e8")


def voice(text, lang):
    """Fresh copy of the audio file each time, so the player autoplays even for a repeated answer."""
    path = speak_to_file(text, lang)
    if not path:
        return None
    AUDIO_DIR.mkdir(exist_ok=True)
    new = AUDIO_DIR / f"{uuid.uuid4().hex}{Path(path).suffix}"
    shutil.copy(path, new)
    return str(new)


def identify(image, lang_name, state):
    lang = LANGUAGES[lang_name]
    state = dict(state or EMPTY_STATE)
    if PRED is None:
        return card("⚠️", LOAD_ERROR, "#fde8e8"), None, None, state
    if image is None:
        state["last"] = None
        return card("📷", "Take or upload a photo of a note"), None, None, state

    res = PRED.predict(image)
    probs = {("No note" if c == config.BACKGROUND else f"₹{c}"): p for c, p in res.probs.items()}
    if res.status == "note":
        text = phrase("note", lang, n=res.value)
        bg, fg = NOTE_COLORS.get(res.label, ("#2e7d32", "#ffffff"))
        html = card(f"₹{res.label}", f"{text}<br><span class='conf'>Confidence {res.confidence:.0%}</span>", bg, fg)
        state["last"] = res.value
    elif res.status == "no_note":
        text = phrase("no_note", lang)
        html = card("🚫", text, "#eceff1")
        state["last"] = None
    else:
        text = phrase("uncertain", lang)
        html = card("🤔", f"{text}<br><span class='conf'>Best guess: {max(probs, key=probs.get)} "
                         f"({res.confidence:.0%})</span>", "#fff1d6")
        state["last"] = None
    return html, probs, voice(text, lang), state


def add_to_total(lang_name, state):
    lang = LANGUAGES[lang_name]
    state = dict(state or EMPTY_STATE)
    if state.get("last") is None:
        return total_card(state["notes"]), voice(phrase("uncertain", lang), lang), state
    state["notes"] = state["notes"] + [state["last"]]
    state["last"] = None  # the same photo can't be added twice
    text = phrase("added", lang, n=state["notes"][-1], t=sum(state["notes"]))
    return total_card(state["notes"]), voice(text, lang), state


def reset_total(lang_name, state):
    lang = LANGUAGES[lang_name]
    state = dict(state or EMPTY_STATE)
    state["notes"] = []
    return total_card([]), voice(phrase("reset", lang), lang), state


def make_image_input():
    common = dict(sources=["webcam", "upload", "clipboard"], type="pil", label="Photo of the note", height=380)
    # The webcam preview is mirrored by default, which flips the printed numbers. Turn that off.
    try:
        return gr.Image(webcam_options=gr.WebcamOptions(mirror=False), **common)
    except (AttributeError, TypeError):
        return gr.Image(mirror_webcam=False, **common)


CSS = """
.result-card {border-radius: 18px; padding: 26px 20px; text-align: center; min-height: 150px;
              box-shadow: 0 2px 10px rgba(0,0,0,.08);}
.result-card .big {font-size: 64px; font-weight: 800; line-height: 1.1;}
.result-card .small {font-size: 19px; margin-top: 8px; line-height: 1.4;}
.result-card .conf {font-size: 15px; opacity: .85;}
#title h1 {margin-bottom: 0;}
"""


def build_ui():
    model_info = ""
    if PRED is not None:
        va = PRED.metrics.get("val_acc")
        model_info = f"Model: `{PRED.arch}` · classes: {', '.join(PRED.classes)}" + (
            f" · validation accuracy {va:.1%}" if va is not None else "")

    with gr.Blocks(title="Note Reader", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        state = gr.State(dict(EMPTY_STATE))
        gr.Markdown("# 💵 Note Reader\nIdentify Indian currency notes from a photo, with a spoken answer.",
                    elem_id="title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image = make_image_input()
                lang = gr.Radio(list(LANGUAGES), value="English", label="Voice language")
                with gr.Row():
                    identify_btn = gr.Button("🔍 Identify note", variant="primary", size="lg")
                    add_btn = gr.Button("➕ Add to total", size="lg")
            with gr.Column(scale=4):
                result = gr.HTML(card("📷", "Take or upload a photo of a note") if PRED
                                 else card("⚠️", LOAD_ERROR, "#fde8e8"))
                audio = gr.Audio(label="Spoken answer", type="filepath", autoplay=True, interactive=False)
                probs = gr.Label(label="How sure the model is", num_top_classes=3)
                total = gr.HTML(total_card([]))
                reset_btn = gr.Button("Clear total", size="sm")
        gr.Markdown(
            "**Tips for best results:** keep the whole note in the photo, avoid strong glare, "
            "and hold the camera steady. The note can be any way up. If the app is not sure, "
            "it asks you to take the photo again instead of guessing.\n\n" + model_info)

        outs = [result, probs, audio, state]
        image.change(identify, [image, lang, state], outs)
        identify_btn.click(identify, [image, lang, state], outs)
        add_btn.click(add_to_total, [lang, state], [total, audio, state])
        reset_btn.click(reset_total, [lang, state], [total, audio, state])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public link (use it on your phone)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(share=args.share, server_port=args.port, inbrowser=True)
