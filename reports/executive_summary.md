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
factors *cause* popularity — only that they're associated with it, to a
degree we quantify honestly below.**

## Dataset

277,937 tracks from a Spotify catalog extract supplied for coursework
(provenance and sampling-frame caveats in the root README — this is not
established as a representative sample of Spotify's full catalog, and the
extract's collection date is unknown).

## Model A: content and context only

Using only observable track/content/context characteristics — loudness,
musical positivity ("valence"), tempo, track length, album type, genre, and
release decade — a plain linear model explains a **modest but real** share
of why some tracks in this dataset score higher than others (roughly
11–15% of the variation, depending on how the test set is built — see
below). That is a legitimate, honestly-reported result, not a strong
predictive model and not presented as one.

## What changed when artist popularity was added

Adding the artist's own popularity score (Model B) produces a genuinely
**mixed and instructive** result rather than a clean improvement:

- Evaluated the *conventional* way (a random sample held out for testing,
  where the same artist can appear in both the training data and the test
  data), Model B looks much better — explaining roughly 28% of the
  variation, nearly double Model A.
- Evaluated the way we consider more trustworthy for this question (holding
  out entire *artists* the model never saw during training), Model B
  actually performs *worse* than Model A, and produces a much larger share
  of nonsensical predictions (popularity scores below zero).

## What the grouped split revealed

This gap is the single most important finding in this analysis, and we
traced it to a specific, verifiable cause rather than leaving it as a
puzzle: **artist popularity is so conceptually close to track popularity**
that a model can partly "memorize" a specific artist's typical popularity
from training data rather than learning something that generalizes to
artists it has never seen. We found direct, concrete evidence of this: a
placeholder value in the data called `"Various Artists"` (used for
compilation albums, not a real artist) happened to fall entirely in the
held-out test set for our specific evaluation, has a fixed "artist
popularity" score of exactly zero regardless of the track, and covers
tracks whose actual popularity varies enormously. That one data quirk
alone flips the aggregate result. This does not mean artist popularity is
useless — it means it cannot be treated as a clean, reliable, independent
predictor of a track's popularity in this dataset, and any claim that it
improves prediction needs to be checked against exactly this kind of
leakage, not taken at face value from a single, convenient evaluation.

## Key descriptive/modeling findings

- Louder and more positive-sounding ("valence") tracks are associated with
  modestly higher popularity in this dataset; longer tracks and (weakly)
  faster tempo are associated with modestly lower popularity.
- Singles tend to have higher popularity than albums; compilations
  noticeably lower — consistent with earlier SQL findings in this project.
- The model's errors are not evenly spread: it systematically underpredicts
  genuinely high-popularity tracks (a known behavior of simple linear
  models applied to a skewed outcome) and shows a large, specific bias for
  compilation tracks that traces to the same "Various Artists" artifact
  described above.
- Re-running everything on a secondary version of the dataset (collapsing
  repeated track/artist entries to one row each) changes nothing
  meaningful — results are essentially identical.

## Limitations

- Observational data, no causal claims, unverified sampling frame and
  collection date (carried over from the data-quality phase of this
  project).
- The artist field is not a perfectly clean artist identifier in this
  extract (the "Various Artists" placeholder is one concrete example).
- Roughly 1 in 5 of the artist-popularity model's predictions on the
  primary test set fall outside the valid 0–100 popularity range — a real
  limitation of a simple linear model applied to a bounded outcome,
  reported directly rather than hidden by post-hoc clipping.
- Model performance overall is modest. This analysis is not a production
  popularity predictor and was not built to be one.

## What this analysis does NOT establish

- It does **not** show that any track characteristic, genre, or artist
  popularity *causes* higher or lower popularity.
- It does **not** establish that artist popularity is a reliable,
  generalizable predictor once evaluated properly — the evidence points the
  other way, at least in this dataset and this evaluation.
- It does **not** generalize beyond this specific data extract to Spotify's
  catalog as a whole, and it was not used to inform any real business,
  marketing, or programming decision.
