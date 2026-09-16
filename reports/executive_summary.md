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
factors *cause* popularity** — only that they're associated with it, to a
degree we quantify honestly below, evaluated the same way ten separate
times to make sure the result isn't a fluke of one lucky (or unlucky) test
set.

## 1. Observable track/content/context features provide modest predictive information

Using only what's directly observable about a track — loudness, musical
positivity ("valence"), tempo, length, album type, genre, and release
decade — a plain, interpretable model consistently explains a real but
modest share of why some tracks in this dataset score higher than others,
roughly 11–13% of the variation, holding steady whether we test on
completely new artists or on a random slice of the catalog. That's an
honest, unglamorous result, and we report it as such.

## 2. The model strongly compresses predictions toward the middle of the popularity range

This is the most important limitation of the content-only model, not a
minor footnote: actual popularity scores in this dataset range from 0 to
100 and are genuinely spread out, but the model's predictions almost never
go above the mid-40s. For tracks that are *actually* very popular, the
model's predictions average roughly half their true score. In plain terms:
**these features can distinguish an ordinary track from a somewhat more
popular one, but they cannot pick out a smash hit.** That's specific to this
dataset, this feature set, and this style of model — it is not a claim
that a song's content never matters for how popular it becomes.

## 3. Adding artist popularity materially changes prediction — but "unseen artist" comes with a caveat

Adding the artist's own popularity score roughly doubles the explained
variation (to about 27–30%), and — after correcting how we grouped artists
for testing (see below) — this improvement holds up consistently whether we
test on a held-out group of artists or on a random sample. That is a real,
repeatable improvement in this dataset, not a one-off. One caveat on what
"held-out" means here: recognized, named artists are cleanly separated
between training and testing, but a meaningful slice of tracks carry
placeholder or unidentified artist credits (not a real performer's name —
see below); those are each treated as their own isolated case rather than
grouped together, but we cannot rule out that the actual, unnamed performer
behind one of those credits also appears elsewhere in the training data. So
this is a good test of generalizing to a *different named-artist grouping*,
not a guaranteed test of generalizing to a *brand-new performer with no
popularity history at all*. Separately, an artist's own popularity score is
so conceptually close to a track's popularity that this should be read as
"artists who are already popular tend to have popular tracks" rather than
as a discovery about what makes a track popular — it is not treated as a
causal effect.

## 4. Random and artist-grouped evaluation answer different practical questions

We tested the model two ways: once holding out entire recognized,
named-artist groups the model never saw during training, and once with a
conventional random sample (closer to a platform working with artists it
already has some history on). The held-out-artist test is not a clean
guarantee of predicting for genuinely brand-new catalog entrants, since a
meaningful slice of tracks carry placeholder or unidentified artist credits
whose real, unnamed performer could still overlap with training (see
above). These two approaches still answer different real-world questions,
and neither is "more correct" in general. In this dataset, once the
artist-grouping was corrected (see below), both approaches told a
consistent story.

## 5. Metadata limitations materially affect interpretation

Two data-quality issues turned out to matter a lot for getting this
analysis right, and both are worth understanding, not just footnoting:

- **Missing genre information** is common (about 60% of tracks) and is not
  randomly missing — tracks without a listed genre tend to be less popular,
  a pattern carried over from the data-quality phase of this project.
- **Placeholder artist labels.** A meaningful slice of tracks (about 5% of
  the dataset) are credited to `"Various Artists"` — a catalog label for
  compilation albums, not a real performer — plus a handful of similar
  placeholders (`"Unknown"`, `"Original Cast"`). Early in this analysis, we
  found that treating `"Various Artists"` as if it were one consistent
  artist could distort the artist-based evaluation, because the model had
  no way to learn anything meaningful from a label that covers thousands of
  unrelated tracks with a flat, uninformative "artist popularity" value of
  exactly zero. We corrected this by treating each such track as its own
  unidentified case rather than lumping them together, and re-ran the
  evaluation ten separate times on independently chosen samples to confirm
  the corrected result holds up. This is a good example of how a subtle
  data-quality issue can otherwise be mistaken for a modeling insight.

## Limitations

- Observational data, no causal claims, unverified sampling frame and
  collection date (carried over from the data-quality phase of this
  project).
- The artist field is still not a perfectly clean artist identifier even
  after the correction above — it may contain concatenated collaborator
  names in other, harder-to-detect cases.
- A small fraction of the artist-popularity model's predictions — roughly 1
  in 2,500 to 1 in 700, depending on the test sample — fall slightly below
  the valid 0–100 popularity range across our repeated tests — a small,
  honestly-reported limitation of a simple linear model, not hidden by
  post-hoc adjustment.
- Model performance overall is modest, and it specifically struggles to
  identify extreme popularity (see finding 2). This is not a production
  popularity predictor and was not built to be one.

## What this analysis does NOT establish

- It does **not** show that any track characteristic, genre, or artist
  popularity *causes* higher or lower popularity.
- It does **not** identify which tracks will become genuinely popular —
  only that it can distinguish ordinary tracks from moderately popular ones
  to a modest degree.
- It does **not** generalize beyond this specific data extract to Spotify's
  catalog as a whole, and it was not used to inform any real business,
  marketing, or programming decision.
