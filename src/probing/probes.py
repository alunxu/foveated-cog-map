"""
Linear probes for analysing GRU hidden state content.

Probes decode spatial information from the agent's memory to test:
- H1: Does the foveated agent over-represent peripherally-seen regions?
- H2: Does the hidden state encode location-dependent confidence?

Member C is responsible for this module.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
import numpy as np


class LinearProbe:
    """Linear probe for decoding information from GRU hidden states.

    Uses sklearn for simplicity and to guarantee that we're
    measuring what the representation *linearly encodes*, not what
    a nonlinear decoder could extract.

    Args:
        probe_type: 'regression' or 'classification'.
        alpha: Regularisation strength (Ridge alpha or LogReg C inverse).
    """

    def __init__(self, probe_type: str = "regression", alpha: float = 1.0):
        self.probe_type = probe_type
        if probe_type == "regression":
            self.model = Ridge(alpha=alpha)
        elif probe_type == "classification":
            self.model = LogisticRegression(C=1.0 / alpha, max_iter=1000)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

    def fit(self, hidden_states: np.ndarray, targets: np.ndarray):
        """Train probe.

        Args:
            hidden_states: (N, hidden_size) GRU hidden states.
            targets: (N,) or (N, D) probe targets.
        """
        self.model.fit(hidden_states, targets)
        return self

    def predict(self, hidden_states: np.ndarray) -> np.ndarray:
        return self.model.predict(hidden_states)

    def score(self, hidden_states: np.ndarray, targets: np.ndarray) -> float:
        """Evaluate probe accuracy.

        Returns:
            R² for regression, accuracy for classification.
        """
        if self.probe_type == "regression":
            preds = self.predict(hidden_states)
            return r2_score(targets, preds)
        else:
            return accuracy_score(targets, self.predict(hidden_states))


class OccupancyProbe(LinearProbe):
    """Probes whether hidden state encodes the room layout (occupancy grid).

    Decodes a flattened occupancy grid from the GRU hidden state.
    For H1: compare probe accuracy for peripherally-seen vs. foveated regions.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(probe_type="regression", alpha=alpha)


class TargetLocationProbe(LinearProbe):
    """Probes whether hidden state encodes the target (goal) location.

    Decodes (x, y) target position from the GRU hidden state.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(probe_type="regression", alpha=alpha)


class FoveationHistoryProbe(LinearProbe):
    """Probes whether hidden state encodes which regions were foveated vs. peripheral.

    For each spatial location, classifies whether it was seen foveally (1)
    or only peripherally (0). Tests H1 directly.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(probe_type="classification", alpha=alpha)


class UncertaintyProbe(LinearProbe):
    """Probes whether hidden state encodes location-dependent perceptual confidence.

    Decodes the per-location uncertainty map from the GRU hidden state.
    Tests H2: does the agent maintain a "belief map"?
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(probe_type="regression", alpha=alpha)
