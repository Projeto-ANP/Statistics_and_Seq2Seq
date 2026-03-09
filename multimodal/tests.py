from pathlib import Path
import ollama

MODEL = "qwen3-vl:8b"
IMAGE_PATH = Path(__file__).parent / "fis_2" / "all_folds_H3.png"

PROMPT = """You are an expert Data Scientist specializing in Time-Series Forecasting and Ensemble Learning.

You are given ONE image containing a mosaic of 3 validation folds (Fold 0, Fold 1, Fold 2).
Each fold contains 4 panels designed to help you select the best models for an ensemble:
- Panel A (Top-Left): Horizontal Bar Chart showing the overall MAE ranking. Models at the top have the lowest (best) overall error.
- Panel B (Top-Right): Scatter Plot of Diversity (Bias vs. MAE). The X-axis is the Absolute Error (MAE) and the Y-axis is the Signed Error (Bias). 
- Panel C (Bottom-Left): Heatmap of MAE per forecasting step (H1, H2, H3) within the validation window, with exact annotated values. Note: H1, H2, H3 represent the sequential steps within the sliding window, not the final forecasting horizon.
- Panel D (Bottom-Right): Time-series plot of y_true vs ALL models.

Your Task:
Select exactly 3 to 6 models to build a highly robust and diverse ensemble across all folds.

Expert Strategy for Selection:
1. Performance: Start by identifying models with low overall MAE (top of Panel A, left side of Panel B).
2. Diversity (Crucial): Use Panel B to find models that make DIFFERENT types of errors. Select models that are on opposite sides of the Bias=0 line (e.g., combine a model that slightly overestimates with one that underestimates) so their errors cancel out when averaged.
3. Consistency: Ensure the chosen models perform reasonably well across all 3 folds, not just one.
4. Avoid Redundancy: Do not select models that are clustered tightly together in Panel B, as they will not add new information to the ensemble.

Rules:
- Use the EXACT model names as written in the figure (case-sensitive).
- List the 3 to 6 chosen models.
- Provide a detailed, expert justification for EACH chosen model, explicitly referencing their behavior in Panels A, B, and C across the folds.
- Explain how the models complement each other (e.g., "Model X offsets the positive bias of Model Y").
- DO NOT USE EMOJIS in your answer.
""".strip()


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH.resolve()}")

    # 1) non-streaming (simplest)
    try:
        print("Calling ollama.chat (non-streaming)...")
        import base64
        with open(IMAGE_PATH, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        resp = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [encoded_string],
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
                "images": [encoded_string],
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