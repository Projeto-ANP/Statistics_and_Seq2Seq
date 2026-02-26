from pathlib import Path
import ollama

MODEL = "qwen3-vl:8b"
IMAGE_PATH = Path("figs_out/all_folds_H3.png")

PROMPT = """You are a time-series forecasting ensemble expert.

You are given ONE image (a mosaic of 3 folds: fold 0, fold 1, fold 2).
Each fold contains:
- Panel A: signed normalized mean error heatmap (model × horizon H1..H3), err_norm = (yhat - y) / mean(|y_true|).
  Blue=negative (underestimate), Red=positive (overestimate).
- Panel C: Top models by normalized MAE per horizon.
- Panel D: y_true vs top model forecasts (context).

Task:
Select 3 to 6 models to build a robust ensemble across all folds.
Prefer complementarity (different error signatures in Panel A across folds/horizons).
Avoid redundant models (very similar error signatures).

Rules:
- Use the exact model names as written in the figure (case-sensitive).
- List 3 to 6 chosen models and explain why each was chosen.
- DO NOT USE EMOJIS in your answer.
""".strip()


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH.resolve()}")

    # 1) non-streaming (simplest)
    try:
        print("Calling ollama.chat (non-streaming)...")
        resp = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(IMAGE_PATH)],
            }],
            options={"temperature": 0.2},
        )
        text = resp.get("message", {}).get("content", "") or ""
    except Exception as e:
        print("Non-streaming call failed:", repr(e))
        text = ""

    # 2) fallback streaming if empty
    if not text.strip():
        print("Empty response. Falling back to streaming...")
        stream = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(IMAGE_PATH)],
            }],
            options={"temperature": 0.2},
            stream=True,
        )
        parts = []
        for chunk in stream:
            part = chunk.get("message", {}).get("content", "") or ""
            if part:
                print(part, end="", flush=True)
                parts.append(part)
        print()  # newline
        text = "".join(parts)

    print("\n\n--- Raw model output ---\n")
    print(text)

    saved_output_path = Path("model_output_raw.txt")
    saved_output_path.write_text(text, encoding="utf-8")
    print(f"\nSaved raw model output to {saved_output_path.resolve()}")


if __name__ == "__main__":
    main()