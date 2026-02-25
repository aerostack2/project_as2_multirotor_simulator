"""
Robot Assignment Model Training Pipeline
=========================================
This module generates training data and trains a decision tree classifier
for robot task assignment. The trained model is saved to disk for use in
assigner.py for inference.

Usage:
    python3 data_gen.py

Output:
    - robot_assignment_model.pkl (trained model file)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from pathlib import Path

# ----------------------------
# Teacher policy (oracle)
# ----------------------------
# This is the "expert" policy used to generate training labels.
# The sklearn decision tree learns from these labels.
# Adjust thresholds to fit your scenario.
# ----------------------------
def teacher_label(dist: float, energy: float, tasks_done: int) -> int:
    """
    Returns 1 if robot should be considered a 'good candidate', else 0.
    Tune thresholds to fit your reality.
    """
    if energy < 15:
        return 0
    if dist > 60 and energy < 30:
        return 0

    # "good" if close enough OR very energetic, but not overloaded
    if dist <= 25 and energy >= 30 and tasks_done <= 7:
        return 1
    if dist <= 10 and energy >= 20 and tasks_done <= 9:
        return 1
    if dist <= 45 and energy >= 70 and tasks_done <= 6:
        return 1

    return 0


# ----------------------------
# Training data generation
# ----------------------------
def make_training_data(n_samples: int, seed_value: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Features: [dist, energy, tasks_done]
    Label: 1 good candidate, 0 otherwise
    """
    rng = np.random.default_rng(seed_value)

    # sample plausible ranges
    dist = rng.uniform(0, 100, size=n_samples)
    energy = rng.uniform(0, 100, size=n_samples)
    tasks_done = rng.integers(0, 11, size=n_samples)

    X = np.column_stack([dist, energy, tasks_done]).astype(float)
    y = np.array([teacher_label(d, e, int(w)) for d, e, w in X], dtype=int)
    return X, y


def train_decision_tree(seed_value: int = 0) -> DecisionTreeClassifier:
    X_train, y_train = make_training_data(n_samples=50_000, seed_value=seed_value)

    # A small-ish tree to stay interpretable
    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        random_state=seed_value,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(model: DecisionTreeClassifier, filepath: str = "robot_assignment_model.pkl"):
    """Save the trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def train_and_save_model(seed_value: int = 42, output_path: str = "robot_assignment_model.pkl"):
    """
    Training pipeline: generates data, trains model, and saves it.
    This is the main entry point for the data generation pipeline.
    """
    print("="*60)
    print("ROBOT ASSIGNMENT MODEL TRAINING PIPELINE")
    print("="*60)
    
    print("\n1. Generating training data...")
    X_train, y_train = make_training_data(n_samples=50_000, seed_value=seed_value)
    print(f"   Generated {len(X_train)} training samples")
    print(f"   Positive examples: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    
    print("\n2. Training decision tree classifier...")
    model = train_decision_tree(seed_value=seed_value)
    print(f"   Tree depth: {model.get_depth()}")
    print(f"   Leaf nodes: {model.get_n_leaves()}")
    
    # Evaluate on training data (for reference)
    train_score = model.score(X_train, y_train)
    print(f"   Training accuracy: {train_score:.3f}")
    
    print("\n3. Saving model...")
    save_model(model, output_path)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nTrained model saved to: {output_path}")
    print("You can now use this model in assigner.py")
    
    return model


if __name__ == "__main__":
    # Run the training pipeline
    train_and_save_model(seed_value=42, output_path="robot_assignment_model.pkl")
