"""
core/bocpd.py — Bayesian Online Changepoint Detection (BOCPD).
============================================================
Computes exact run-length posterior distributions over streaming anomaly inputs,
featuring automatic pruning boundaries and a whitening filter to address non-stationarity.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.stats import t

class PredictiveAnomalyDetector:
    """Predictive Bayesian Online Changepoint Detector for tracking model collapse regimes."""
    def __init__(
        self,
        hazard_rate: float = 500.0,  # Expected 1 collapse per 500 tokens (h = 0.002)
        max_horizon: int = 256,
        prune_threshold: float = 1e-5,
        alpha0: float = 2.0,
        beta0: float = 0.05,
        kappa0: float = 1.0,
        mu0: float = 0.75  # Target hybrid unperturbed baseline
    ):
        self.hazard = 1.0 / hazard_rate
        self.max_horizon = max_horizon
        self.prune_threshold = prune_threshold
        
        # Hyperparameters for the Normal-Inverse-Gamma (NIG) Conjugate Prior
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.mu0 = mu0
        
        # Active vector blocks for running hypotheses
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])
        self.kappa = np.array([kappa0])
        self.mu = np.array([mu0])
        
        # Run-length posterior distribution vector: P(r_0 = 0) = 1.0
        self.R = np.array([1.0])
        self.ema = None
        self.ema_alpha = 0.15

    def process_step(self, zeta: float, r_ratio: float) -> float:
        """Fuses metrics, whitens non-stationary drift, and updates run-length distributions.
        
        Returns
        -------
        changepoint_probability : float (Range [0.0, 1.0])
        """
        # 1. Metric Fusion Core
        raw_x = (zeta ** 0.65) * (r_ratio ** 0.35)
        
        # 2. Online Whitening Transformation
        if self.ema is None:
            self.ema = raw_x
            return 0.0
        
        self.ema = (self.ema_alpha * raw_x) + ((1.0 - self.ema_alpha) * self.ema)
        whitened_x = raw_x - self.ema

        # 3. Compute Student-t Predictive Likelihood Vector Across Active Hypotheses
        dof = 2.0 * self.alpha
        loc = self.mu
        scale = np.sqrt((self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa))
        
        # Guard scale from zero or negative anomalies
        scale = np.maximum(scale, 1e-9)
        pred_likelihoods = t.pdf(whitened_x, df=dof, loc=loc, scale=scale)
        pred_likelihoods = np.nan_to_num(pred_likelihoods, nan=1e-12)

        # Continue evidence under current active run-length hypotheses (for surprise detection)
        cont_ev = float(np.sum(self.R * pred_likelihoods))

        # 4. Formulate Run-Length Growth vs Shift Realignment Matrices
        current_len = len(self.R)
        new_R = np.zeros(current_len + 1)

        # Predictive likelihood under the *prior* for a fresh changepoint hypothesis (r=0)
        dof0 = 2.0 * self.alpha0
        loc0 = self.mu0
        scale0 = np.sqrt((self.beta0 * (self.kappa0 + 1.0)) / (self.alpha0 * self.kappa0))
        scale0 = max(scale0, 1e-9)
        cp_pred_lik = t.pdf(whitened_x, df=dof0, loc=loc0, scale=scale0)
        cp_pred_lik = float(np.nan_to_num(cp_pred_lik, nan=1e-12))

        # Apply Growth rule (r_t = r_{t-1} + 1) and Changepoint rule (r_t = 0)
        # Use old hypotheses' predictive for growth; use prior predictive for cp branch.
        new_R[1:] = self.R * pred_likelihoods * (1.0 - self.hazard)
        new_R[0] = cp_pred_lik * self.hazard   # (sum R == 1)

        # Standardize distribution space
        total_mass = np.sum(new_R)
        if total_mass > 0.0:
            self.R = new_R / total_mass
        else:
            self.R = np.zeros(current_len + 1)
            self.R[0] = 1.0

        # 5. Continuous Parameter Updates
        # Update continuing (grown) hypotheses from their previous posteriors + new obs
        updated_kappa = self.kappa + 1.0
        updated_mu = ((self.kappa * self.mu) + whitened_x) / updated_kappa
        updated_alpha = self.alpha + 0.5
        updated_beta = self.beta + (0.5 * self.kappa * ((whitened_x - self.mu) ** 2)) / updated_kappa

        # The new changepoint hypothesis (r=0) starts from prior and is updated with this obs
        cp_kappa = self.kappa0 + 1.0
        cp_mu = (self.kappa0 * self.mu0 + whitened_x) / cp_kappa
        cp_alpha = self.alpha0 + 0.5
        cp_beta = self.beta0 + (0.5 * self.kappa0 * ((whitened_x - self.mu0) ** 2)) / cp_kappa

        self.kappa = np.append(cp_kappa, updated_kappa)
        self.mu = np.append(cp_mu, updated_mu)
        self.alpha = np.append(cp_alpha, updated_alpha)
        self.beta = np.append(cp_beta, updated_beta)

        # 6. Compute Pruning Boundary to Prevent Memory Growth
        if len(self.R) > self.max_horizon:
            keep_indices = np.where(self.R >= self.prune_threshold)[0]
            if len(keep_indices) == 0 or keep_indices[0] != 0:
                keep_indices = np.append(0, keep_indices)
            
            self.R = self.R[keep_indices]
            self.kappa = self.kappa[keep_indices]
            self.mu = self.mu[keep_indices]
            self.alpha = self.alpha[keep_indices]
            self.beta = self.beta[keep_indices]
            
            # Re-normalize trimmed vector weights
            mass = np.sum(self.R)
            if mass > 0:
                self.R /= mass

        changepoint_probability = float(self.R[0])

        # Sensitivity boost for the unitarity-lab self-healing application:
        # When the observation is surprising under *all* active run hypotheses
        # (cont_ev << 1), report high confidence in a regime change so that
        # alerts (>0.92) and proactive flux kicks (>0.95) can fire with low
        # latency while preserving a low false-positive rate on stable data
        # (where cont_ev is typically >1 and boost has no effect, R0 ~ hazard).
        if cont_ev < 1.0:
            surprise = max(0.0, 1.0 - cont_ev)
            boost = min(0.999, 0.5 + 0.5 * surprise)
            changepoint_probability = max(changepoint_probability, boost)

        return changepoint_probability
