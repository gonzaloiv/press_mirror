Activates the virtual environment from the project root: .venv\Scripts\activate

Install Playwright and Chromium once: python -m pip install playwright; python -m playwright install chromium

Terminal 1 — launches the web app from the project root: python web_app\server.py

Terminal 2 — runs ingest, embed, cluster, and render in sequence: python pipeline.py

Renders images only for the latest cluster snapshot, replacing only that day's images: python pipeline\render.py --clean

Daily headlines and clusters are saved under history/ as headlines_YYYY-MM-DD.json and clusters_YYYY-MM-DD.json. Rendered images are archived under static/maps/YYYY-MM-DD/.
