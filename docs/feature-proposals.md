# Press\Mirror — Product Review and Feature Proposals

Review date: 2026-11-14

## Today

A daily press-comparison product: 42 feeds across 19 countries are embedded,
clustered, and filtered to 5–9 stories per day. Each story becomes a
coverage map and a detail page listing one headline per outlet, grouped by
country. A GitHub Actions pipeline regenerates the site daily; Buffer
syndicates the top stories.

The core promise is strong: *what happened today, and who covered it*. The
main gaps are readability without images, continuity across days, and the
ambiguity between "not covered" and "we have no data".

## Findings

- **The homepage is images only.** `web_app/templates/home.html` renders
  `<img … alt="">` with no story text, so crawlers and screen readers get
  only header and footer copy. Detail pages have no meta description or
  Open Graph tags.
- **Feed failures look like silence.** `pipeline/ingest.py` reports failures
  only to CI logs. The server derives tracked countries from the day's
  headlines (`active_countries()`), so a country whose feeds all fail is
  drawn as "not tracked". Hungary is configured but absent from four of the
  last five snapshots, and 11 of 19 countries depend on a single feed.
- **Stories have no memory.** Clustering is per-day, and navigation is only
  older/newer day. Recurring events appear as unrelated cards, and
  `published` timestamps are dropped from cluster entries.
- **The detail page lists outlets but does not compare countries.** RSS
  summaries are captured during ingest but never shown. The map also needs
  JavaScript and a third-party GeoJSON fetch from `raw.githubusercontent.com`.
- **Most stories are invisible.** Snapshots store 89–101 clusters, but only
  the gated 5–9 are surfaced; there is no archive, search, or date picker.

## Proposals

| # | Feature | User value | Effort |
|---|---------|------------|--------|
| 1 | Story timelines across days | Follow an event over time; see who joined or dropped it | L |
| 2 | Coverage integrity state | Prevent no-data from being shown as no-coverage | S–M |
| 3 | Country-by-country comparison | Show *how* countries differed, not just whether they covered | M |
| 4 | Accessible, shareable cards | Real story text for users, crawlers, and social previews | S |
| 5 | Archive and search | Retrieve past stories instead of only browsing sequentially | M |
| 6 | RSS / email digest | A familiar retention channel for a news audience | S–M |

### 1. Story timelines

Match cluster centroids across adjacent snapshots and assign stable story
IDs. Add "appeared on" dates and prev/next story links to the detail page,
a `/story/<id>` page with outlets and countries per day, and a recurring
badge on homepage cards.

Persist per-day embeddings or cluster centroids under `history/<date>/`
because `data/embeddings.npz` is overwritten each run. Keep `published` on
cluster articles so the timeline can be ordered.

### 2. Coverage integrity

Write `history/<date>/ingest.json` with per-feed status (`ok`, `empty`,
`failed`), article count, and reason. Derive tracked countries from
`config/sources.json`, add a fourth "no data" map state, and disclose gaps
on the homepage and detail pages. Fail or warn in CI when failures cross a
threshold.

This is the smallest change with the largest credibility impact.

### 3. Country-by-country comparison

Render a compact block on the detail page with one representative headline
and short summary per country, using the existing `sources_by_country`
grouping and RSS summaries from `headlines.json`. Optionally add a
`/compare/<date>` matrix of stories × countries.

### 4. Accessible, shareable cards

Render labels, terms, outlet/country counts, and source names as real HTML
beside each image; give images meaningful alt text. Add meta description,
Open Graph, and Twitter tags using the existing `_large.png`, plus
`/sitemap.xml`, `/robots.txt`, and a share button.

### 5. Archive and search

Add `/archive` with dates, story counts, and labels, a date picker, and
`/search?q=` across labels, terms, outlets, and countries. Generate
`data/index.json` in the pipeline once the archive outgrows on-demand scans.

### 6. Syndication

Publish `/feed.xml` for the latest day's stories and reuse the same content
for a daily email digest.

## Suggested sequence

1. **Quick wins (1–2 days):** accessible/shareable cards (#4) and coverage
   integrity (#2).
2. **Flagship (3–5 days):** story timelines (#1).
3. **Depth (2–3 days):** country-by-country comparison (#3).
4. **Retention (2–3 days):** archive/search (#5) and RSS (#6).

If only one feature is built next, choose **story timelines** for
differentiation — but ship the **coverage-integrity guardrail** first.
Timelines built on maps that can silently mislabel a country would amplify
the trust problem rather than solve it.
