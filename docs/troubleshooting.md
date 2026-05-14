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

## `401 Missing or invalid API key`

Protected POST endpoints require `GEORISK_API_KEY` by default. Set it in `.env`,
restart the API and frontend, then send the same value from the frontend sidebar
or in API requests:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ml/predict `
  -Headers @{ "X-API-Key" = $env:GEORISK_API_KEY } `
  -ContentType "application/json" `
  -Body "{}"
```

For throwaway local demos only, you can set `GEORISK_ALLOW_UNAUTHENTICATED=true`.
Do not use that setting for deployed environments.

## `429 Rate limit exceeded`

The API includes an in-memory limiter for expensive endpoints. For local testing,
wait for the configured window to reset or tune:

```text
GEORISK_RATE_LIMIT_WINDOW_SECONDS
GEORISK_RATE_LIMIT_REQUESTS
GEORISK_TRAIN_RATE_LIMIT_REQUESTS
GEORISK_UPLOAD_RATE_LIMIT_REQUESTS
GEORISK_RAG_RATE_LIMIT_REQUESTS
```

## PDF uploads return no chunks

Some PDFs contain scanned images without embedded text. Use OCR first, then upload
the text-searchable PDF.
