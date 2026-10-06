import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Usage: python generate_600.py [num_voices] [num_sentences]   (default 10 x 10 = 600 clips)
N_VOICES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_SENT = int(sys.argv[2]) if len(sys.argv) > 2 else 10

load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("XI_API_KEY")
if not api_key:
    sys.exit("No API key found. Put ELEVENLABS_API_KEY=... in the .env file next to this script.")
client = ElevenLabs(api_key=api_key)

MODEL_ID = "eleven_v4"
OUT = Path("emotion_test_600")

# Same sentences for every emotion, so only the delivery changes, not the words.
SENTENCES = [
    "The meeting will begin at ten o'clock tomorrow morning.",
    "I submitted the assignment yesterday afternoon.",
    "The train will arrive at platform number three.",
    "Please send me the document before the end of the day.",
    "The weather is expected to remain clear tomorrow.",
    "The computer is currently connected to the network.",
    "The next class starts at nine thirty.",
    "I finished reading the report this morning.",
    "The office will remain open until six o'clock.",
    "The results will be available after the evaluation.",
][:N_SENT]
TAGS = {"angry": "[angry] ", "disgust": "[disgusted] ", "fearful": "[fearful] ",
        "happy": "[happy] ", "neutral": "", "sad": "[sad] "}


# Used when your key cannot list voices (for example a restricted API key).
FALLBACK = [
    ("George", "JBFqnCBsd6RMkjVDRZzb", "male"), ("Sarah", "EXAVITQu4vr4xnSDxMaL", "female"),
    ("Charlie", "IKne3meq5aSn9XLyUdCD", "male"), ("Laura", "FGY2WhTYpPnrIDTdsKH5", "female"),
    ("Callum", "N2lVS1w4EtoT3dr4eOWO", "male"), ("Alice", "Xb7hH8MSUJpSbSDYk0k2", "female"),
    ("Liam", "TX3LPaxmHKxFdv7VOQHJ", "male"), ("Matilda", "XrExE9yKIg1WjnnlVkGX", "female"),
    ("Roger", "CwhRBWXzGAHq8TQ4Fs17", "male"), ("Will", "bIHbv24MWmeRgasZH58o", "male"),
]


def list_voices():
    last_error = None
    for name in ("search", "get_all"):
        fn = getattr(client.voices, name, None)
        if fn:
            try:
                return list(fn().voices)
            except Exception as e:
                last_error = e
    print("Could not list voices from your account:", last_error)
    print("Using the built-in list of default ElevenLabs voices instead.\n")
    return [SimpleNamespace(name=n, voice_id=i, labels={"gender": g}) for n, i, g in FALLBACK]


def pick_voices(n):
    vs = sorted(list_voices(), key=lambda v: (v.name or "").lower())
    def gender(v):
        return ((getattr(v, "labels", None) or {}).get("gender") or "unknown").lower()
    male = [v for v in vs if gender(v) == "male"]
    female = [v for v in vs if gender(v) == "female"]
    other = [v for v in vs if gender(v) not in ("male", "female")]
    mixed = []
    for i in range(max(len(male), len(female))):
        mixed += [lst[i] for lst in (male, female) if i < len(lst)]
    return (mixed + other)[:n], gender


voices, gender = pick_voices(N_VOICES)
if not voices:
    sys.exit("Could not list voices from your ElevenLabs account.")

plan = []
for s_idx, sentence in enumerate(SENTENCES, 1):          # sentence-major, so a partial run stays balanced
    for v in voices:
        for emo, tag in TAGS.items():
            safe = "".join(c for c in v.name if c.isalnum()) or v.voice_id[:6]
            rel = f"{OUT.name}/{safe}/{emo}/{emo}_{s_idx:02d}.mp3"
            plan.append({"filename": rel, "emotion": emo, "voice_id": v.voice_id,
                         "voice_name": v.name, "gender": gender(v),
                         "sentence_id": s_idx, "text": tag + sentence})

chars = sum(len(p["text"]) for p in plan if not (OUT.parent / p["filename"]).exists())
print("Voices:", ", ".join(f"{v.name} ({gender(v)})" for v in voices))
print(f"Clips planned: {len(plan)}   Still to generate: "
      f"{sum(1 for p in plan if not Path(p['filename']).exists())}   Approx. characters: {chars}")
if input("Proceed? (y/n): ").strip().lower() != "y":
    sys.exit("Cancelled.")


def write_labels():
    done = [p for p in plan if Path(p["filename"]).exists()]
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
        w.writeheader()
        w.writerows(done)
    return len(done)


OUT.mkdir(exist_ok=True)
failures = 0
voice_fails = {}
try:
    for i, p in enumerate(plan, 1):
        path = Path(p["filename"])
        if path.exists() or voice_fails.get(p["voice_id"], 0) >= 3:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            audio = client.text_to_speech.convert(
                voice_id=p["voice_id"], text=p["text"], model_id=MODEL_ID,
                output_format="mp3_44100_128")
            path.write_bytes(b"".join(audio))
            failures = 0
            print(f"[{i}/{len(plan)}] {path.name}  ({p['voice_name']})")
            time.sleep(0.3)
        except Exception as e:
            failures += 1
            voice_fails[p["voice_id"]] = voice_fails.get(p["voice_id"], 0) + 1
            print(f"[{i}/{len(plan)}] FAILED {path.name}: {e}")
            if failures >= 6:
                print("Six failures in a row (quota or key problem?). Stopping; run again to resume.")
                break
finally:
    print(f"\nlabels.csv written with {write_labels()} clips in {OUT}")