# API Examples

Start the API:

```bash
uvicorn app.main:app --reload
```

Predict risk:

```bash
curl -X POST http://localhost:8000/ml/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $GEORISK_API_KEY" \
  -d '{
    "contamination_bq_m2": 35000,
    "soil_clay_pct": 28,
    "soil_organic_pct": 6,
    "rainfall_mm_year": 750,
    "elevation_m": 120,
    "slope_deg": 4,
    "distance_to_water_km": 2.5,
    "population_density_km2": 180,
    "latitude": 59.91,
    "longitude": 10.75,
    "land_cover_urban_pct": 35
  }'
```

Compare scenarios:

```bash
curl -X POST http://localhost:8000/ml/scenarios \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $GEORISK_API_KEY" \
  -d '{
    "baseline": {
      "contamination_bq_m2": 35000,
      "soil_clay_pct": 28,
      "soil_organic_pct": 6,
      "rainfall_mm_year": 750,
      "elevation_m": 120,
      "slope_deg": 4,
      "distance_to_water_km": 2.5,
      "population_density_km2": 180,
      "latitude": 59.91,
      "longitude": 10.75,
      "land_cover_urban_pct": 35
    },
    "scenarios": [
      {"name": "Wet year", "overrides": {"rainfall_mm_year": 1100}},
      {"name": "Remediation", "overrides": {"contamination_bq_m2": 15000}}
    ]
  }'
```

Train the advanced real-data model:

```bash
curl -X POST http://localhost:8000/ml/train/advanced \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $GEORISK_API_KEY" \
  -d '{"n_estimators": 300, "random_state": 42, "cv_splits": 5}'
```

Predict with advanced nuclide features:

```bash
curl -X POST http://localhost:8000/ml/predict/advanced \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $GEORISK_API_KEY" \
  -d '{
    "cs137_kBq_m2": 35,
    "sr90_kBq_m2": 4,
    "k40_Bq_kg": 420,
    "ra226_Bq_kg": 35,
    "th232_Bq_kg": 32,
    "latitude": 59.91,
    "longitude": 10.75
  }'
```

Ask the RAG assistant after uploading PDFs:

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $GEORISK_API_KEY" \
  -d '{"question": "What monitoring actions are recommended after heavy rainfall?", "top_k": 4}'
```

Protected POST endpoints require `X-API-Key` unless
`GEORISK_ALLOW_UNAUTHENTICATED=true` is explicitly enabled for local demos.
