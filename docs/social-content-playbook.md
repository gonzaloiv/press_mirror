# Social Content Playbook

How to turn Press\Mirror's daily clusters into posts that reach more people.

## Principle

The map answers *who covered it*. The differentiator is *how they told it*.
Post the comparison, not just the conclusion: one event, several national
presses, side by side. That is visual, scarce, and hard to replicate.

## Formats

### 1. "One story, N headlines" — the flagship

- **Instagram carousel:** map + coverage counts → one headline per country
  with outlet and English translation → "not covered by tracked sources
  in ..." → CTA/link in bio.
- **X thread:** hook post with the map, one post per country, final post
  linking to the full map.
- **Data:** each cluster already has one article per outlet with `source`,
  `country`, `title`, and `link`.

### 2. "The silence card"

The map plus the countries where none of our tracked outlets covered the
story. Phrase it as *"No coverage in the sources we track"* — never as
"didn't cover it."

### 3. Translation card

One event, four headlines in four languages, with English translations.
Shows the multilingual value immediately.

### 4. "Same event, different words"

Use the cluster's shared terms and their language variants
(`source_terms`): for example, *drone / Drohne / dron* as one concept, or
which words each national press preferred.

### 5. Country lens

A rotating series: "What the French press is covering today", "Germany's
front page, translated." Best for DE, FR, ES, IT, UK, NL, and PL, where we
have several outlets.

### 6. Quiz / poll

"Which of these countries did *not* cover this story?" or "Guess which
headline is from which country." Works well in Instagram Stories and X
polls, and drives replies and saves.

### 7. Weekly recap

Top five stories, total outlets and countries tracked, biggest coverage
gap. Suitable for LinkedIn, a newsletter, or a Sunday carousel.

### 8. Methodology / behind the scenes

"How we decide two headlines are the same story", "Why 'not tracked' is
different from 'not covered'." Builds trust and is shareable by
journalism and media-literacy audiences.

### 9. Story update

"Day 3: four more countries joined this story; two dropped it." This
becomes possible once cross-day story IDs exist.

### 10. Animated map

Coverage spreading over days as a 10–20 second Reel or Short. Same data,
much higher reach on Instagram, TikTok, and YouTube.

## Platform fit

| Platform | Best formats |
|---|---|
| X | Threads with 4–6 headline posts, map card, polls, recurring coverage-gap posts |
| Instagram | Carousels (map → headlines → gap → CTA), Stories polls/quizzes, Reels, link in bio |
| LinkedIn | Weekly analysis, methodology, coverage charts |
| Bluesky / Threads / Mastodon | Mirror the X thread or card with light adaptation |
| Newsletter | Weekly recap and one story of the week |

## Cadence

- **Daily:** most-covered story — map plus three headlines, not the map
  alone.
- **Twice a week:** "Same story, N headlines" carousel or thread.
- **Weekly:** "Coverage gap of the week" plus recap.
- **Rotating:** country lens.
- **Monthly:** methodology or transparency post.

## Sample posts

**X thread hook**

> One story. 24 outlets. 12 countries.
>
> Here is how Europe's press framed it. 🧵

**X country post**

> 🇩🇪 Die Welt: "Koalition streitet über Drohnen-Lieferung"
> *Coalition argues over drone delivery.*
>
> Full map: [link]

**Instagram carousel**

1. Map + "24 outlets in 12 countries covered this story."
2. 🇩🇪 Germany — headline + outlet + translation.
3. 🇫🇷 France — headline + outlet + translation.
4. 🇮🇹 Italy — headline + outlet + translation.
5. "Not covered by tracked sources in: ..."
6. "See every headline → link in bio."

**Stories poll**

> Which country's press skipped this story?
> A) Poland  B) Portugal  C) Neither

## Production support

Add a **social kit** stage after rendering:

- For each selected cluster, generate the large map, a headline card
  (1200×1200 or 1080×1350), and a JSON file with captions, alt text,
  per-country headline lines, and links.
- Use a Playwright template similar to `card.html`; add
  `pipeline/social.py` and let `pipeline/publish.py` consume the kit.
- Choose editorial slots instead of only the top three: one
  most-covered, one biggest gap, one recurring story, one country lens.

## Guardrails

- Short headline excerpts only, always with outlet and link; no
  front-page screenshots.
- Do not post "X didn't cover it" until ingestion health is tracked —
  otherwise a feed outage becomes a false claim.
- Add alt text to every image.
- Say "the sources we track", not "Europe" or "the press" in general.
- Adapt tone and format per platform; do not cross-post identically.

## Priority

Start with the two formats that fit today's data best:

1. "One story, N headlines" carousel and thread.
2. The silence card.

Together they form a repeatable daily pair and express the product better
than the map alone. Add the weekly coverage-gap series once the
ingestion-health work in `docs/feature-proposals.md` is in place.
