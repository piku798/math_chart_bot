# Smart Math Buddy

A small Flask app + simple ML classifier to solve primary-school math questions and fall back to a local Ollama LLM for open-ended queries.

## Files
- `app.py` — Flask web server and solver; key functions: `extract_numbers`, `ollama_mistral_answer`, `solve_question`.
- `trainmodel.py` — training script that creates `math_model.pkl` and `vectorizer.pkl`.
- `templates/index.html` — frontend UI and JS.
- `static/style.css` — frontend styles.
- `output/` — exported assets, logs, model artifacts, and demo recordings.

## Output (included)
- `output/Smart Math Buddy and 1 more page - Personal - Microsoft​ Edge 2025-09-01 11-10-17.mp4` — demo video recording (added as project output).

## Quick concepts
- The classifier predicts intents (e.g., `addition`, `lcm`, `area_circle`) and `solve_question` uses extracted numbers to compute answers.
- Falls back to a local Ollama LLM at `http://localhost:11434` if the classifier is unsure.

## Requirements
- Python 3.8+
- pip packages: flask, pandas, scikit-learn, joblib, requests

## Setup & run
1. Install dependencies:
```sh
pip install flask pandas scikit-learn joblib requests
```
2. (Optional) Train the model:
```sh
python trainmodel.py
```
3. Run the app:
```sh
python app.py
```
4. Open http://127.0.0.1:5000/ to use the UI.

## Notes & troubleshooting
- Ensure `large_math_question_dataset.xlsx` exists before running `trainmodel.py`.
- If using Ollama fallback, run Ollama locally and ensure it's reachable on port `11434`.
- Model files created by training: `math_model.pkl`, `vectorizer.pkl`.

## License
Add a LICENSE file as needed.