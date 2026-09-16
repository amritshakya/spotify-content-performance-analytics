# Executive Summary — Spotify Popularity Modeling (Phase 3)

*For a non-technical analyst or hiring manager. Full technical detail is in
`notebooks/03_popularity_modeling.ipynb`; data-quality background is in
`data_quality_report.md` and the root `README.md`.*

## Question

How much of the variation in a track's popularity score is associated with
things we can directly observe about the track itself — its sound, its
genre, its release context — and does knowing how popular the *artist* is
add meaningfully more?

**This is an observational dataset. Nothing here shows that any of these
factors *cause* popularity** — only that they're associated with it,
evaluated across ten separate holdouts to check the result isn't a fluke of
one lucky (or unlucky) test set.

## 1. Observable track/content/context features provide modest predictive information

Using only what's directly observable about a track — loudness, musical
positivity ("valence"), tempo, length, album type, genre, and release
decade — a plain, interpretable model consistently explains a modest share
of why some tracks in this dataset score higher than others: roughly
11–13% of the variation, holding steady across both held-out-artist and
random-sample evaluation.

## 2. The model strongly compresses predictions toward the middle of the popularity range

Actual popularity scores in this dataset range from 0 to 100 and are
genuinely spread out, but the model's predictions almost never go above the
mid-40s. For tracks that are *actually* very popular, the model's
predictions average roughly half their true score — both models
substantially underpredict the scores of high-popularity tracks. That's
specific to this dataset, this feature set, and this style of model — it
is not a claim that a song's content never matters for how popular it
becomes.

## 3. Adding artist popularity materially improves prediction, under both evaluation designs

Adding the artist's own popularity score roughly doubles the explained
variation (to about 27–30%), consistently whether tested against held-out
recognized artists or a random sample. One caveat: a meaningful slice of
tracks carry placeholder or unidentified artist credits (not a real
performer's name), so the held-out-artist test does not guarantee the
model was evaluated against performers with no popularity history at all.
Separately, an artist's own popularity score is conceptually close to a
track's popularity — this is read as "artists who are already popular
tend to have popular tracks," not as a discovery about what makes a track
popular, and it is not treated as a causal effect.

## 4. Metadata limitations materially affect interpretation

- **Missing genre information** is common (about 60% of tracks) and is not
  randomly missing — tracks without a listed genre tend to be less popular.
- **Placeholder artist labels.** About 5% of tracks are credited to
  `"Various Artists"` — a catalog label for compilation albums, not a real
  performer — plus a handful of similar placeholders (`"Unknown"`,
  `"Original Cast"`). These carry a flat, uninformative "artist popularity"
  value despite covering tracks with widely varying actual popularity, so
  each such credit is treated as its own case rather than grouped as one
  artist.

## Limitations

- Observational data, no causal claims, unverified sampling frame and
  collection date (carried over from the data-quality phase of this
  project).
- The artist field is still not a perfectly clean artist identifier — it
  may contain concatenated collaborator names in other, harder-to-detect
  cases beyond the placeholder labels above.
- A small fraction of the artist-popularity model's predictions — roughly 1
  in 2,500 to 1 in 700, depending on the test sample — fall slightly below
  the valid 0–100 popularity range.
- Model performance overall is modest, and it specifically struggles with
  extreme popularity (see finding 2). This is not a production popularity
  predictor and was not built to be one.

## What this analysis does NOT establish

- It does **not** show that any track characteristic, genre, or artist
  popularity *causes* higher or lower popularity.
- It does **not** establish how well the model would rank or select
  tracks — the diagnostics here measure prediction error, not ranking or
  selection performance.
- It does **not** generalize beyond this specific data extract to Spotify's
  catalog as a whole, and it was not used to inform any real business,
  marketing, or programming decision.
