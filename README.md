Activates the virtual environment from the project root: .venv\Scripts\activate

Install Playwright and Chromium once: python -m pip install playwright; python -m playwright install chromium

Terminal 1 — launches the web app from the project root: python web_app\server.py

Terminal 2 — runs ingest, embed, cluster, and render in sequence: python pipeline.py

Renders missing images for all dated cluster snapshots and refreshes the latest day's images: python pipeline\render.py --clean

Daily headlines and clusters are saved under history/ as headlines_YYYY-MM-DD.json and clusters_YYYY-MM-DD.json. Rendered images are archived under static/maps/YYYY-MM-DD/.

The web app opens on the newest cluster snapshot. Use the Newer day and Older day links to browse available snapshots; each day's grid and map pages use that date's archived data and images.

The grid shows every cluster whose average similarity is at least 0.75, with at least 10 distinct sources across at least 4 countries. The same criteria are used when rendering map images.

Cluster labels are deterministic noun phrases built from terms shared by the cluster headlines, with a maximum of seven words. English headlines are preferred when present; otherwise the source language is retained.

Shared country-code helpers are in shared/, and the generated embedding artifact is stored at data/embeddings.npz.