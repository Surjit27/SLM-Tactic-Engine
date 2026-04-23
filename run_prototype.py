"""Run prototype inference pipeline end-to-end with latency summary."""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
from data_utils import load_split
from generate_dataset import generate_and_save
from prototype_inference import PrototypeInferencePipeline
from prototype_latency import LatencyRecorder
from prototype_vector_db import InMemoryVectorDB


def _build_vector_index(observations: np.ndarray, tactical_targets: np.ndarray) -> InMemoryVectorDB:
    # Mirror the lightweight embedding used in the inference pipeline.
    hist = observations[:, 96:256].reshape(-1, 5, 32)
    embeddings = np.stack(
        [
            observations[:, :32].mean(axis=1),
            observations[:, 32:64].mean(axis=1),
            observations[:, 64:96].mean(axis=1),
            hist.mean(axis=(1, 2)),
            hist.std(axis=(1, 2)),
            np.abs(hist).mean(axis=(1, 2)),
            observations[:, 86],
            observations[:, 6],
        ],
        axis=1,
    ).astype(np.float32)

    # Synthetic risk proxy: higher threat and lower health => higher risk.
    risk_scores = np.clip(0.7 * observations[:, 86] + 0.3 * (1.0 - observations[:, 6]), 0.0, 1.0).astype(np.float32)

    db = InMemoryVectorDB()
    db.build_index(embeddings=embeddings, tactical_labels=tactical_targets, risk_scores=risk_scores)
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype inference runner with vector retrieval and latency metrics")
    parser.add_argument("--samples", type=int, default=20, help="Number of inference samples to run")
    parser.add_argument("--dataset", type=str, default="data/generated/train.npz")
    parser.add_argument("--regen-data", action="store_true", help="Regenerate synthetic data before run")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if args.regen_data:
        generate_and_save(train_samples=5000, val_samples=1000, seed=config.RANDOM_SEED)

    split = load_split(args.dataset)
    vector_db = _build_vector_index(split.observations, split.tactical_targets)
    pipeline = PrototypeInferencePipeline(vector_db=vector_db, top_k=args.top_k)

    recorder = LatencyRecorder()
    results = []

    n = min(args.samples, len(split.observations))
    for i in range(n):
        st = recorder.start()
        out = pipeline.predict(split.observations[i])
        recorder.stop("full_inference", st)
        results.append(out)

    print(f"Ran prototype inference on {n} samples.")
    print("\nLast sample prediction:")
    last = results[-1]
    print(f"- tactical_class: {last.tactical_class}")
    print(f"- tactical_probs: {np.round(last.tactical_probs, 4).tolist()}")
    print(f"- action_shape: {last.action.shape}")
    print(f"- latency_breakdown_ms: {json.dumps(last.latency_breakdown_ms, indent=2)}")
    retrieval_preview = {
        "risk_score": float(last.retrieval["risk_score"]),
        "rare_case_flag": float(last.retrieval["rare_case_flag"]),
        "neighbor_similarity_mean": float(last.retrieval["neighbor_similarity_mean"]),
    }
    print(f"- retrieval_summary: {json.dumps(retrieval_preview, indent=2)}")

    print("\nAggregate latency:")
    print(json.dumps(recorder.summary(), indent=2))


if __name__ == "__main__":
    main()
