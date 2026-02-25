"""
Robot Task Assignment - Inference Module
========================================
This module loads a trained decision tree model and uses it to assign 
tasks to robots based on their distance, energy, and workload.

The model must be trained first using data_gen.py.

Usage:
    python3 assigner.py  # Run demo
    
Or import and use:
    from assigner import assign_task_to_n_robots_sklearn, load_model
"""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from random import randint, uniform, seed
from typing import List, Tuple
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib


@dataclass
class Robot:
    id: int
    x: float
    y: float
    energy: float          # 0..100
    tasks_done: int        # workload / previously carried out tasks
    busy: bool = False     # if True, robot is unavailable


@dataclass
class Task:
    id: int
    goal_x: float
    goal_y: float


def distance_to_goal(r: Robot, t: Task) -> float:
    return hypot(r.x - t.goal_x, r.y - t.goal_y)


def load_model(filepath: str = "robot_assignment_model.pkl") -> DecisionTreeClassifier:
    """Load a trained decision tree model from disk."""
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"Model file '{filepath}' not found. "
            f"Please run data_gen.py first to train and save the model."
        )
    return joblib.load(filepath)


def decision_tree_tier(dist: float, energy: float, tasks_done: int) -> int:
    """
    Lower tier is better.
    This is the "decision tree" (explicit branching rules).
    Adjust thresholds to fit your scenario.
    """
    # Hard filters / immediate deprioritization conditions
    if energy < 15:
        return 99  # effectively "do not assign"
    if dist > 60 and energy < 30:
        return 90  # too far for low-ish energy

    # Tree: proximity first, then energy, then workload
    if dist <= 10:
        # Very close robots: prefer those with enough energy and not overloaded
        if energy >= 60:
            return 0 if tasks_done <= 3 else 1
        else:
            return 2 if tasks_done <= 2 else 3

    elif dist <= 25:
        # Medium close
        if energy >= 50:
            return 1 if tasks_done <= 4 else 2
        elif energy >= 30:
            return 3 if tasks_done <= 3 else 4
        else:
            return 7

    elif dist <= 45:
        # Farther away
        if energy >= 70:
            return 2 if tasks_done <= 6 else 3
        elif energy >= 45:
            return 5 if tasks_done <= 4 else 6
        else:
            return 8

    else:
        # Very far
        if energy >= 80 and tasks_done <= 3:
            return 4
        return 85


def tie_break_score(dist: float, energy: float, tasks_done: int) -> float:
    """
    Used within the same tier to rank robots.
    Higher score is better.
    """
    # Normalize-ish terms
    # - closer => better (negative distance)
    # - more energy => better
    # - more workload => worse
    return (energy * 1.2) - (dist * 2.0) - (tasks_done * 8.0)


def assign_task_to_n_robots(robots: List[Robot], task: Task, n: int) -> List[Tuple[Robot, dict]]:
    """
    Returns selected robots (best N) plus diagnostics for why they were picked.
    Uses the manual decision tree tier system (legacy).
    """
    candidates = []
    for r in robots:
        if r.busy:
            continue

        dist = distance_to_goal(r, task)
        tier = decision_tree_tier(dist, r.energy, r.tasks_done)
        score = tie_break_score(dist, r.energy, r.tasks_done)

        diagnostics = {
            "dist": round(dist, 2),
            "energy": round(r.energy, 1),
            "tasks_done": r.tasks_done,
            "tier": tier,
            "score": round(score, 2),
        }
        candidates.append((tier, -score, r, diagnostics))
        # note: we sort by (tier asc, score desc). Using -score to sort ascending.

    candidates.sort(key=lambda x: (x[0], x[1]))

    chosen = []
    for (tier, neg_score, r, diag) in candidates[: max(0, min(n, len(candidates)))]:
        chosen.append((r, diag))
    return chosen


def assign_task_to_n_robots_sklearn(
    robots: List[Robot],
    task: Task,
    n: int,
    model: DecisionTreeClassifier,
) -> List[Tuple[Robot, dict]]:
    """
    Uses trained sklearn model to rank robots by probability of being a good candidate.
    Picks top N among non-busy robots.
    
    Features: [distance, energy, tasks_done]
    Model predicts probability that robot is a "good candidate"
    """
    candidates = []
    for r in robots:
        if r.busy:
            continue

        dist = distance_to_goal(r, task)
        X = np.array([[dist, r.energy, r.tasks_done]], dtype=float)
        proba_good = float(model.predict_proba(X)[0, 1])  # probability of class 1

        # Tie-break: prefer closer & higher energy & lower workload if probabilities match
        tie_break = (r.energy * 1.2) - (dist * 2.0) - (r.tasks_done * 8.0)

        diagnostics = {
            "dist": round(dist, 2),
            "energy": round(r.energy, 1),
            "tasks_done": r.tasks_done,
            "p_good": round(proba_good, 4),
            "tie_break": round(tie_break, 2),
        }
        candidates.append((proba_good, tie_break, r, diagnostics))

    # sort: highest probability first, then tie_break
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    chosen = [(r, diag) for (_, _, r, diag) in candidates[: max(0, min(n, len(candidates)))]]
    return chosen


def demo():
    seed(7)
    
    # Load the trained model
    print("Loading trained decision tree model...")
    try:
        model = load_model("robot_assignment_model.pkl")
        print(f"Model loaded successfully!")
        print(f"  Tree depth: {model.get_depth()}")
        print(f"  Leaf nodes: {model.get_n_leaves()}")
        use_sklearn = True
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Falling back to manual decision tree.")
        use_sklearn = False

    # Create M robots
    M = 12
    robots = []
    for i in range(M):
        robots.append(
            Robot(
                id=i,
                x=uniform(0, 100),
                y=uniform(0, 100),
                energy=uniform(5, 100),
                tasks_done=randint(0, 10),
                busy=(randint(0, 10) < 2),  # ~20% busy
            )
        )

    # One task with a goal point
    task = Task(id=1, goal_x=50, goal_y=50)

    # Select N robots
    N = 4
    if use_sklearn:
        chosen = assign_task_to_n_robots_sklearn(robots, task, N, model)
    else:
        chosen = assign_task_to_n_robots(robots, task, N)

    print(f"\nTask {task.id} goal=({task.goal_x},{task.goal_y})")
    print(f"Selecting N={N} robots out of M={M} (excluding busy).")
    print(f"Using: {'sklearn model' if use_sklearn else 'manual decision tree'}")
    
    print("\n--- All robots ---")
    for r in robots:
        d = distance_to_goal(r, task)
        print(
            f"R{r.id:02d} pos=({r.x:5.1f},{r.y:5.1f}) "
            f"dist={d:6.2f} energy={r.energy:5.1f} tasks_done={r.tasks_done:2d} busy={r.busy}"
        )

    print("\n--- Chosen robots (ranked) ---")
    for rank, (r, d) in enumerate(chosen, start=1):
        if use_sklearn:
            print(
                f"{rank}. R{r.id:02d}  p_good={d['p_good']:<6}  "
                f"dist={d['dist']:<6} energy={d['energy']:<5} "
                f"tasks_done={d['tasks_done']:<2} tie_break={d['tie_break']}"
            )
        else:
            print(
                f"{rank}. R{r.id:02d}  tier={d['tier']:>2}  score={d['score']:>7}  "
                f"dist={d['dist']:>6}  energy={d['energy']:>5}  tasks_done={d['tasks_done']}"
            )

    if use_sklearn:
        print("\nNote: p_good is the probability from the sklearn model; higher is better.")
    else:
        print("\nNote: tier comes from the explicit decision tree; score breaks ties within a tier.")


if __name__ == "__main__":

    demo()
