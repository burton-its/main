"""
recommender.py

Starter content-based recommender:
- Loads a CSV of tracks with Spotify-style audio features
- Builds a normalized feature matrix
- Creates a user preference vector from sliders + optional genre filtering
- Ranks songs by cosine similarity

Dependencies:
  pip install pandas numpy scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# --- Configure which columns you want to use as "audio features" ---
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    # We'll include tempo, but we will scale it like everything else
    "tempo",
]

# Optional columns you might want in outputs
OUTPUT_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
]


@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    feature_matrix: np.ndarray          # shape (n_tracks, n_features)
    scaler: MinMaxScaler                # used to scale both songs + user vector
    feature_names: List[str]


def load_and_build_artifacts(csv_path: str) -> RecommenderArtifacts:
    df = pd.read_csv(csv_path)

    # Basic hygiene: drop rows missing any required feature
    needed = set(AUDIO_FEATURES + ["track_genre", "track_id", "track_name", "artists"])
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    clean = df.dropna(subset=AUDIO_FEATURES).copy()

    # Ensure numeric types for audio features
    for col in AUDIO_FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=AUDIO_FEATURES)

    # Scale audio features to 0..1 so similarity behaves nicely
    scaler = MinMaxScaler()
    X = scaler.fit_transform(clean[AUDIO_FEATURES].values)

    return RecommenderArtifacts(
        df=clean.reset_index(drop=True),
        feature_matrix=X,
        scaler=scaler,
        feature_names=list(AUDIO_FEATURES),
    )


def _build_user_vector(
    artifacts: RecommenderArtifacts,
    preferences: Dict[str, float],
    tempo_bpm: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Build a 1 x d user vector in the same scaled feature space as songs.

    preferences should contain 0..1 values for most audio features.
    tempo_bpm is provided separately and will be inserted into 'tempo' before scaling.

    weights optionally re-weights dimensions after scaling.
    """
    # Start from "neutral" defaults in case user doesn't set everything
    raw = {name: 0.5 for name in artifacts.feature_names}

    # Fill provided slider prefs (expected 0..1)
    for k, v in preferences.items():
        if k in raw:
            raw[k] = float(v)

    # Tempo: user might provide BPM (e.g. 60..200). We'll put it into raw tempo slot.
    if tempo_bpm is not None:
        raw["tempo"] = float(tempo_bpm)
    else:
        # If no tempo provided, keep neutral (will get scaled appropriately)
        raw["tempo"] = raw.get("tempo", 120.0)

    # Put into the same order as features
    user_raw_vec = np.array([[raw[f] for f in artifacts.feature_names]], dtype=float)

    # Scale using the same scaler as the track data
    user_scaled = artifacts.scaler.transform(user_raw_vec)

    # Apply weights (post-scaling)
    if weights:
        w = np.array([weights.get(f, 1.0) for f in artifacts.feature_names], dtype=float)
        user_scaled = user_scaled * w

    return user_scaled


def recommend(
    artifacts: RecommenderArtifacts,
    preferences: Dict[str, float],
    genres: Optional[List[str]] = None,
    tempo_bpm: Optional[float] = None,
    k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    diversify_by_artist: bool = True,
    max_per_artist: int = 2,
) -> pd.DataFrame:
    """
    Returns a DataFrame of top-k recommended tracks.
    """
    df = artifacts.df
    X = artifacts.feature_matrix

    # Optional genre filtering (simple + effective)
    if genres:
        genres_set = {g.strip().lower() for g in genres}
        mask = df["track_genre"].astype(str).str.lower().isin(genres_set)
        candidate_idx = np.where(mask.values)[0]
        if len(candidate_idx) == 0:
            # If filtering removes everything, fallback to all songs
            candidate_idx = np.arange(len(df))
    else:
        candidate_idx = np.arange(len(df))

    # Candidate feature matrix
    Xc = X[candidate_idx, :].copy()

    # Weight candidates the same way as user (post-scaling)
    if weights:
        w = np.array([weights.get(f, 1.0) for f in artifacts.feature_names], dtype=float)
        Xc = Xc * w

    user_vec = _build_user_vector(
        artifacts=artifacts,
        preferences=preferences,
        tempo_bpm=tempo_bpm,
        weights=weights,
    )

    # Cosine similarity: (n_candidates, 1)
    sims = cosine_similarity(Xc, user_vec).reshape(-1)

    # Sort candidates by similarity desc
    order = np.argsort(-sims)

    results = df.iloc[candidate_idx[order]].copy()
    results["similarity"] = sims[order]

    # Optional: diversify so you don't return 10 songs from the same artist
    if diversify_by_artist and "artists" in results.columns:
        kept_rows = []
        counts: Dict[str, int] = {}

        for _, row in results.iterrows():
            artist = str(row.get("artists", "")).strip().lower()
            counts.setdefault(artist, 0)
            if counts[artist] < max_per_artist:
                kept_rows.append(row)
                counts[artist] += 1
            if len(kept_rows) >= k:
                break

        results = pd.DataFrame(kept_rows)

    # Final: select nice columns
    cols = [c for c in OUTPUT_COLUMNS if c in results.columns] + ["similarity"]
    return results[cols].head(k).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage:
    artifacts = load_and_build_artifacts("tracks.csv")

    # A pretend user who wants upbeat dancey vibes
    prefs = {
        "energy": 0.85,
        "danceability": 0.75,
        "valence": 0.65,
        "acousticness": 0.15,
        "instrumentalness": 0.10,
        "speechiness": 0.25,
        "liveness": 0.35,
    }
    weights = {
        "energy": 1.3,
        "valence": 1.2,
        "danceability": 1.1,
        "tempo": 0.7,
        "liveness": 0.6,
    }

    recs = recommend(
        artifacts,
        preferences=prefs,
        genres=["pop", "dance"],
        tempo_bpm=130,
        k=1,
        weights=weights,
        diversify_by_artist=True,
        max_per_artist=2,
    )

    print(recs.to_string(index=False))