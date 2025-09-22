# CNN Article → Headline Generator

This repository contains a trained RNN-based headline generator for CNN articles and a small Gradio web app (`app.py`) to interactively generate headlines from article text. There's also a Jupyter notebook (`CODE.ipynb`) used during experimentation and development.

# hugging face UI 
https://huggingface.co/spaces/JHANANISHRI-B/RNN

## Contents

- `app.py` — Gradio application that loads a trained Keras model and tokenizers to generate headlines from input articles.
- `CODE.ipynb` — Notebook used during model training/experimentation.
- `CNN_Articles.csv` — (Optional) dataset used to train the model (if present).
- `requirements.txt` — Python dependencies for this project.
- `headline_model_best.h5` — Trained Keras model (NOT included). Place it in the project root.
- `article_tokenizer.pkl` — Pickled tokenizer used for articles (NOT included). Place it in the project root.
- `headline_tokenizer.pkl` — Pickled tokenizer used for headlines (NOT included). Place it in the project root.

## Quick start

1. Create and activate a virtual environment (recommended):

   - On Windows (PowerShell):

     ```powershell
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
     ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Place the trained model and tokenizer files in the project root:

   - `headline_model_best.h5`
   - `article_tokenizer.pkl`
   - `headline_tokenizer.pkl`

   The `app.py` script checks for these files at startup and will raise a FileNotFoundError if they are missing.

4. Run the Gradio app:

   ```powershell
   python app.py
   ```

   This will start a local Gradio server and open a browser UI where you can paste an article and click "Generate Headline".

## How it works (short)

- `app.py` loads the trained Keras sequence-to-sequence model and two tokenizers (articles and headlines).
- The article text is tokenized and padded to `MAX_ARTICLE_LEN` before being fed to the encoder.
- The decoder is fed a growing headline token sequence and the model predicts the next token at each step until it either predicts the padding/stop token or reaches `MAX_HEADLINE_LEN`.

## Notebook (`CODE.ipynb`)

Open `CODE.ipynb` in Jupyter or VS Code. The notebook likely contains data loading, preprocessing, tokenizer saving, model training, and evaluation code. Use it to retrain or inspect preprocessing steps if your tokenizers or model are missing.



## Requirements

Dependencies are listed in `requirements.txt`. Key packages include:

- Python 3.8+ (recommended)
- tensorflow
- gradio
- numpy
- huggingface_hub (optional — used in `app.py` import but not required at runtime unless fetching files)

Install them using `pip install -r requirements.txt`.

## Troubleshooting

- FileNotFoundError on startup: ensure `headline_model_best.h5`, `article_tokenizer.pkl`, and `headline_tokenizer.pkl` are in the project root and spelled exactly as in the README.
- TensorFlow version issues: this project uses Keras from TensorFlow (tf.keras). If you trained the model with a different TF version, you may need the same version to load the model without errors.
- Large model memory: the model may use significant RAM/VRAM. Close other processes or run on a machine with more memory if you encounter OOM errors.
- If generated headlines are empty or contain unknown tokens: the tokenizers used for inference must match those used during training. Recreate/save tokenizers from the training notebook if needed.

## Next steps and enhancements

- Add a simple unit test to check model and tokenizer file presence.
- Add a small sample article text file for demo use.
- Package the model and tokenizers or provide download links (e.g., upload to Hugging Face and use `hf_hub_download` in `app.py`).



