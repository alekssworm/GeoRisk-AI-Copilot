# Troubleshooting

## `ModuleNotFoundError: No module named 'ml'`

Run Streamlit from the repository root:

```powershell
cd D:\pro
uv run --with-requirements requirements-dev.txt streamlit run frontend/streamlit_app.py
```

The frontend also inserts the project root into `sys.path`, so direct Streamlit
launches from the `frontend/` directory should still work.

## `Internal Server Error` on `/ml/predict`

Regenerate the model artifact:

```powershell
uv run --with-requirements requirements-dev.txt python -m ml.train
```

Older artifacts may reference `__main__.ModelArtifact`. Current artifacts use
`ml.artifact.ModelArtifact`, and the API retrains automatically if a stale model
cannot be loaded.

## `uv` cannot initialize cache

If a sandbox or Windows permission issue blocks the global `uv` cache, rerun the
command from a normal terminal session with access to your user cache directory.

## API is unavailable from Streamlit

Start the backend first:

```powershell
uv run --with-requirements requirements-dev.txt uvicorn app.main:app --reload
```

Then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## PDF uploads return no chunks

Some PDFs contain scanned images without embedded text. Use OCR first, then upload
the text-searchable PDF.
