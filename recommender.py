"""

loads tracks CSV
scales audio features
recommends top-1by cosine similarity to a default user vector (all 0.5)
optional genre filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
# thanks to siddhardhan
from sklearn.preprocessing import MinMaxScaler
# actual features
from sklearn.metrics.pairwise import cosine_similarity

# features
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",]

# return these
RETURN_COLUMNS = ["track_name", "artists", "track_genre"] 


@dataclass
class Artifacts:
    df: pd.DataFrame
    X: np.ndarray
    scaler: MinMaxScaler


def load_artifacts(csv_path: str) -> Artifacts:
    df = pd.read_csv(csv_path)

    #cleaning: drop rows missing required columns
    df = df.dropna(subset=AUDIO_FEATURES + ["track_name", "artists", "track_genre"]).copy()

    #ensure numeric
    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)
    # scale everything to 0-1 so we can have everything 
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES].values)

    return Artifacts(df=df, X=X, scaler=scaler)

# recommend function
def recommend(
    artifacts: Artifacts,genres: Optional[List[str]] = None,k: int = 1,) -> pd.DataFrame:

    
    df, X = artifacts.df, artifacts.X

    # genre filter
    if genres:
        genres_set = {g.strip().lower() for g in genres}
        mask = df["track_genre"].astype(str).str.lower().isin(genres_set)
        # set the value with where
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            idx = np.arange(len(df))
    else:
        idx = np.arange(len(df))
    # get the actual array
    array1 = X[idx, :]

    # 0.5 defualt
    user_raw = np.full((1, len(AUDIO_FEATURES)), 0.5, dtype=float)
    user_vec = artifacts.scaler.transform(user_raw)
    # get actual cosine similarity
    sims = cosine_similarity(array1, user_vec).reshape(-1)
    order = np.argsort(-sims)

    results = df.iloc[idx[order]].copy()
    results["similarity"] = sims[order]

    cols = [c for c in RETURN_COLUMNS if c in results.columns] + ["similarity"]
    return results[cols].head(k).reset_index(drop=True)


if __name__ == "__main__":
    artifacts = load_artifacts("tracks.csv")
    recs = recommend(artifacts, genres=["pop", "dance"], k=1)
    print(recs.to_string(index=False))