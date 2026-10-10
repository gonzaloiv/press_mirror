Activates the virtual environment from the project root: .venv\Scripts\activate

Install Playwright and Chromium once: python -m pip install playwright; python -m playwright install chromium

Terminal 1 — launches the web app from the project root: python web_app\server.py

Terminal 2 — runs ingest, embed, cluster, and render in sequence: python pipeline.py

Renders missing images for all dated cluster snapshots and refreshes the latest day's images: python pipeline\render.py --clean

Each day is stored in its own folder under history/ as history/YYYY-MM-DD/headlines.json, history/YYYY-MM-DD/clusters.json, and rendered images under history/YYYY-MM-DD/maps/.

The web app opens on the newest cluster snapshot. Use the Newer day and Older day links to browse available snapshots; each day's grid and map pages use that date's archived data and images.

The grid shows the coverage-filtered clusters — those meeting the source, country, and representative-similarity minimums defined in `shared/cluster_selection.py`. From those it takes the five most similar plus any remaining cluster at or above the similarity floor, ordered most similar first. The same criteria decide which map images are rendered.

Cluster labels are extractive, not generated: each cluster reports up to seven terms that recur across its headlines, ranked by how many headlines they touch. The multilingual embedding model groups equivalent forms from different languages (for example "drone" / "Drohne" / "dron") into one concept, so translations compete as a single shared term. The list is stored as `terms`. A local LLM (Qwen2.5-1.5B-Instruct) rewrites those terms plus the cluster's headlines into one short neutral English headline for `label`, translating non-English clusters when needed; if the model is unavailable or fails, `label` falls back to placing the terms in the word order of the headline nearest the cluster centroid (for example "Moscow drone attack") or, failing that, joining them with ` · `. `label_source` records whether the label came from the LLM or the terms. Pages show `label` as the title and `terms` as the subtitle.

Shared country-code helpers are in shared/, and the generated embedding artifact is stored at data/embeddings.npz.