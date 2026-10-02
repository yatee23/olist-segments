# Olist customer segments UI

Streamlit app for the DATA 309 customer segmentation project. It shows the six k-means
segments, the LLM-generated segment descriptions and campaign recommendations, segment
comparison charts, the model selection results, and a customer lookup.

## Run locally
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
Then open http://localhost:8501

## Files the app needs (in the same folder as app.py)
- app.py
- requirements.txt
- the LLM output JSON (any JSON with "interpretation" and "recommendation" per cluster)
- the customer-level CSV with customer_unique_id
- the cluster labels CSV (joined on customer_unique_id, or by row order if it has no id)

The app finds these files automatically. The sidebar shows which files were loaded.

## Privacy
Only coarse location (state and city tier) is displayed. ZIP codes, city names and
coordinates are never shown.
