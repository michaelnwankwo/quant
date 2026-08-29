"""Pure-NumPy / SciPy Gaussian-HMM fallback backend for the regime switchboard.

Why this module exists
----------------------
``hmmlearn`` compiles a Cython extension (``_hmmc``) and publishes prebuilt
Windows wheels **only** for CPython 3.8 - 3.13 on ``win_amd64``.  There is no
``cp314`` (or newer) wheel and no 32-bit / ARM64 wheel, so on a newer
interpreter pip falls back to the source tarball and installation fails with::

    building 'hmmlearn._hmmc' extension
    error: Microsoft Visual C++ 14.0 or greater is required.
    ERROR: Failed building wheel for hmmlearn

This module re-implements the slice of the ``hmmlearn.hmm.GaussianHMM`` API that
:class:`~quant_system.models.hmm_switchboard.HMMSwitchboard` depends on
(``fit``, ``score``, ``predict``, ``predict_proba`` and the fitted attributes)
using only NumPy and SciPy - both of which ship wheels for every supported
Python version.  :mod:`quant_system.models.hmm_switchboard` imports ``hmmlearn``
first and only falls back to this module when that import fails, so behaviour is
unchanged wherever ``hmmlearn`` *is* installed.

Numerical parity with hmmlearn
------------------------------
The defaults deliberately mirror hmmlearn's: ``algorithm="viterbi"``,
``n_iter=10``, ``tol=1e-2``, ``min_covar=1e-3``, ``covars_prior=1e-2``,
K-Means initialisation of ``means_``, a tiled ``np.cov`` for ``covars_`` and
uniform ``startprob_`` / ``transmat_``.  The two backends therefore produce
closely comparable regime paths, but they are not bit-identical: this
implementation runs in log space via :func:`scipy.special.logsumexp` rather than
hmmlearn's scaled recursions, so log-likelihoods can differ in the trailing
digits and hidden-state indices may permute.  Permutations are harmless - the
switchboard re-labels hidden states into the canonical ``{0, 1, 2}`` ordering
after every fit.

Supported ``covariance_type`` values are ``"diag"``, ``"full"``,
``"spherical"`` and ``"tied"``.  Close parity with hmmlearn is targeted for
``"diag"`` and ``"full"`` (the values the project uses); the other two are
supported for completeness but are approximations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "GaussianHMM",
    "ConvergenceMonitor",
    "SUPPORTED_COVARIANCE_TYPES",
]

SUPPORTED_COVARIANCE_TYPES: Tuple[str, ...] = ("diag", "full", "spherical", "tied")
SUPPORTED_ALGORITHMS: Tuple[str, ...] = ("viterbi", "map")

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-300


@dataclass
class ConvergenceMonitor:
    """Tracks the EM log-likelihood history.

    Mirrors the small part of :class:`hmmlearn.base.ConvergenceMonitor` that
    consumers rely on: a ``history`` list, an ``iter`` counter and a
    ``converged`` flag.
    """

    tol: float = 1e-4
    n_iter: int = 100
    verbose: bool = False
    history: list = field(default_factory=list)
    iter: int = 0

    @property
    def converged(self) -> bool:
        """Whether the last two log-likelihoods differ by less than ``tol``."""
        if len(self.history) < 2:
            return False
        return bool(abs(self.history[-1] - self.history[-2]) < self.tol)

    def report(self, log_prob: float) -> bool:
        """Record ``log_prob`` and return ``True`` once converged."""
        self.history.append(float(log_prob))
        self.iter += 1
        if self.verbose:
            logger.debug("builtin HMM iter %d: logL=%.6f", self.iter, log_prob)
        return self.converged


def _row_normalise(matrix: np.ndarray) -> np.ndarray:
    """Normalise a 2-D array so each row sums to one (robust to zero rows)."""
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    out = matrix / safe
    fallback = np.full_like(matrix, 1.0 / max(matrix.shape[1], 1))
    return np.where(totals > 0, out, fallback)


class GaussianHMM:
    """Hidden Markov Model with Gaussian emissions, implemented with NumPy.

    Drop-in replacement for the subset of ``hmmlearn.hmm.GaussianHMM`` used by
    :class:`~quant_system.models.hmm_switchboard.HMMSwitchboard`.

    Args:
        n_components: Number of hidden states.
        covariance_type: One of ``"diag"``, ``"full"``, ``"spherical"``,
            ``"tied"``.
        min_covar: Floor applied to variances for numerical stability.
        startprob_prior: Dirichlet prior added to the initial-state vector.
        transmat_prior: Dirichlet prior added to each transition row.
        means_prior: Gaussian prior mean for the emission means.
        means_weight: Weight of the emission-mean prior.
        covars_prior: Additive prior on the emission variances.
        covars_weight: Weight of the emission-covariance prior.
        algorithm: ``"viterbi"`` (hard EM, hmmlearn's default) or ``"map"``
            (Baum-Welch soft EM).
        random_state: Seed for K-Means initialisation.
        n_iter: Maximum number of EM iterations.
        tol: Convergence threshold on the change in log-likelihood.
        verbose: Emit per-iteration debug logging.
        params: Subset of ``"s"``/``"t"``/``"m"``/``"c"`` updated in the M-step.
        init_params: Subset of ``"s"``/``"t"``/``"m"``/``"c"`` initialised
            before the first E-step.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        means_prior: float = 0.0,
        means_weight: float = 0.0,
        covars_prior: float = 1e-2,
        covars_weight: float = 1.0,
        algorithm: str = "viterbi",
        random_state: object = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stmc",
        init_params: str = "stmc",
    ) -> None:
        if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}, "
                f"got {covariance_type!r}."
            )
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got {algorithm!r}."
            )
        if int(n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components!r}.")

        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.min_covar = float(min_covar)
        self.startprob_prior = float(startprob_prior)
        self.transmat_prior = float(transmat_prior)
        self.means_prior = float(means_prior)
        self.means_weight = float(means_weight)
        self.covars_prior = float(covars_prior)
        self.covars_weight = float(covars_weight)
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.params = params
        self.init_params = init_params

        # Fitted attributes (mirroring hmmlearn's naming).
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.monitor_: ConvergenceMonitor = ConvergenceMonitor(
            tol=self.tol, n_iter=self.n_iter, verbose=self.verbose
        )
        self.converged_: bool = False

    # ------------------------------------------------------------------ #
    # Validation / initialisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        """Coerce ``X`` to a finite 2-D float array."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {arr.ndim} dimensions.")
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one observation.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("X contains NaN or infinite values.")
        return arr

    def _init_params_from_data(self, X: np.ndarray) -> None:
        """Initialise means/covars/startprob/transmat the way hmmlearn does."""
        n, d = self.n_components, X.shape[1]

        if "m" in self.init_params:
            self.means_ = self._kmeans_init(X, n)

        if "c" in self.init_params:
            cv = np.atleast_2d(np.cov(X.T)) + self.min_covar * np.eye(d)
            if self.covariance_type == "diag":
                self.covars_ = np.tile(np.diag(cv), (n, 1))
            elif self.covariance_type == "full":
                self.covars_ = np.tile(cv, (n, 1, 1))
            elif self.covariance_type == "spherical":
                self.covars_ = np.full(n, float(np.diag(cv).mean()))
            else:  # tied
                self.covars_ = cv.copy()

        if "s" in self.init_params:
            self.startprob_ = np.full(n, 1.0 / n)
        if "t" in self.init_params:
            self.transmat_ = np.full((n, n), 1.0 / n)

    def _kmeans_init(self, X: np.ndarray, n: int) -> np.ndarray:
        """Seed ``means_`` with K-Means centres, falling back to a random draw."""
        try:
            from sklearn.cluster import KMeans  # local: keeps import optional-ish

            # n_init is deliberately left at scikit-learn's default because
            # hmmlearn does the same - passing an explicit value would seed the
            # two backends from different centres.
            labels = KMeans(
                n_clusters=n, random_state=self.random_state
            ).fit_predict(X)
            centres = np.empty((n, X.shape[1]), dtype=float)
            for k in range(n):
                members = X[labels == k]
                centres[k] = members.mean(axis=0) if len(members) else X.mean(axis=0)
            return centres
        except Exception as exc:  # pragma: no cover - environment-dependent
            logger.debug("K-Means HMM init unavailable (%s); using random rows.", exc)
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=n, replace=len(X) < n)
            return X[idx].astype(float).copy()

    # ------------------------------------------------------------------ #
    # Emission densities
    # ------------------------------------------------------------------ #
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log emission density ``log p(x_t | z_t = k)`` -> shape ``(T, K)``."""
        T, d = X.shape
        n = self.n_components
        log_b = np.empty((T, n), dtype=float)

        for k in range(n):
            diff = X - self.means_[k]
            if self.covariance_type == "diag":
                var = np.maximum(self.covars_[k], self.min_covar)
                maha = np.sum((diff * diff) / var, axis=1)
                log_det = float(np.sum(np.log(var)))
            elif self.covariance_type == "spherical":
                var = max(float(self.covars_[k]), self.min_covar)
                maha = np.sum(diff * diff, axis=1) / var
                log_det = d * float(np.log(var))
            else:  # full or tied
                cov = np.array(
                    self.covars_
                    if self.covariance_type == "tied"
                    else self.covars_[k],
                    dtype=float,
                )
                # hmmlearn only regularises when the Cholesky factorisation
                # fails.  Adding ``min_covar * I`` unconditionally would inflate
                # every variance, flatten the emission densities and drag the
                # fit away from hmmlearn's.
                try:
                    chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    cov = cov + self.min_covar * np.eye(d)
                    chol = np.linalg.cholesky(cov)
                solved = np.linalg.solve(chol, diff.T)  # (d, T)
                maha = np.sum(solved * solved, axis=0)
                log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
            log_b[:, k] = -0.5 * (d * _LOG_2PI + log_det + maha)

        return log_b

    # ------------------------------------------------------------------ #
    # Dynamic-programming passes
    # ------------------------------------------------------------------ #
    def _forward_backward(
        self, log_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Baum-Welch forward/backward in log space.

        Returns:
            ``(log_alpha, log_beta, gamma, log_likelihood)`` where ``gamma`` is
            the ``(T, K)`` matrix of smoothed state occupancies.
        """
        log_start = np.log(np.maximum(self.startprob_, _EPS))
        log_trans = np.log(np.maximum(self.transmat_, _EPS))
        T, n = log_b.shape

        log_alpha = np.empty((T, n), dtype=float)
        log_alpha[0] = log_start + log_b[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            log_alpha[t] += log_b[t]

        log_beta = np.empty((T, n), dtype=float)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )

        log_likelihood = float(logsumexp(log_alpha[-1]))
        gamma = np.exp(log_alpha + log_beta - log_likelihood)
        return log_alpha, log_beta, gamma, log_likelihood

    def _accumulate_xi(
        self, log_alpha: np.ndarray, log_beta: np.ndarray, log_b: np.ndarray
    ) -> np.ndarray:
        """Sum of pairwise posteriors ``sum_t p(z_t=i, z_{t+1}=j)`` -> ``(K, K)``."""
        log_trans = np.log(np.maximum(self.transmat_, _EPS))
        log_xi = (
            log_alpha[:-1, :, None]
            + log_trans[None, :, :]
            + log_b[1:, None, :]
            + log_beta[1:, None, :]
        )
        return np.exp(logsumexp(log_xi, axis=0))

    def _viterbi(self, log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        """Most likely state path (log-space Viterbi).

        Returns:
            ``(states, log_probability_of_best_path)``.
        """
        log_start = np.log(np.maximum(self.startprob_, _EPS))
        log_trans = np.log(np.maximum(self.transmat_, _EPS))
        T, n = log_b.shape
        delta = np.empty((T, n), dtype=float)
        psi = np.zeros((T, n), dtype=int)

        delta[0] = log_start + log_b[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(n)] + log_b[t]

        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states, float(np.max(delta[-1]))

    # ------------------------------------------------------------------ #
    # M-step
    # ------------------------------------------------------------------ #
    def _do_mstep(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """Maximisation step: re-estimate ``stmc`` from the expected counts."""
        n, d = self.n_components, X.shape[1]
        denom = gamma.sum(axis=0)  # (K,)

        if "s" in self.params:
            self.startprob_ = _row_normalise(
                (gamma[0] + self.startprob_prior - 1.0)[None, :]
            )[0]
        if "t" in self.params:
            self.transmat_ = _row_normalise(xi + self.transmat_prior - 1.0)
        if "m" in self.params:
            numerator = gamma.T @ X + self.means_weight * self.means_prior
            divisor = np.maximum(denom + self.means_weight, 1e-12)[:, None]
            self.means_ = numerator / divisor
        if "c" in self.params:
            cv_den = np.maximum(self.covars_weight - 1.0, 0.0) + denom
            cv_den = np.maximum(cv_den, 1e-12)
            if self.covariance_type == "diag":
                cv_num = (
                    gamma.T @ (X * X)
                    - 2.0 * self.means_ * (gamma.T @ X)
                    + (denom[:, None] * self.means_ * self.means_)
                )
                cv_num = (
                    cv_num
                    + self.means_weight * self.means_prior**2
                    + self.covars_prior
                )
                self.covars_ = cv_num / cv_den[:, None]
            elif self.covariance_type == "spherical":
                sq = np.sum((X[:, None, :] - self.means_[None, :, :]) ** 2, axis=2)
                cv_num = (gamma * sq).sum(axis=0) + d * self.covars_prior
                self.covars_ = cv_num / cv_den
            else:  # full or tied
                cv_num = np.zeros((n, d, d), dtype=float)
                for k in range(n):
                    diff = X - self.means_[k]
                    cv_num[k] = (diff * gamma[:, k][:, None]).T @ diff
                    if self.means_weight > 0.0:
                        offset = self.means_prior - self.means_[k]
                        cv_num[k] += self.means_weight * np.outer(offset, offset)
                cv_num = cv_num + self.covars_prior * np.eye(d)
                if self.covariance_type == "tied":
                    self.covars_ = cv_num.sum(axis=0) / cv_den.sum()
                else:
                    self.covars_ = cv_num / cv_den[:, None, None]
            self.covars_ = np.maximum(self.covars_, self.min_covar)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, lengths: object = None) -> "GaussianHMM":
        """Estimate the model parameters with EM.

        Args:
            X: Observation matrix ``(T, D)``.
            lengths: Unused; accepted for hmmlearn signature compatibility.

        Returns:
            ``self`` (mutated in place).
        """
        X = self._validate(X)
        if lengths is not None:  # pragma: no cover - single-sequence usage
            logger.debug("builtin HMM ignores `lengths`; treating X as one sequence.")
        if len(X) < self.n_components:
            raise ValueError(
                f"need at least {self.n_components} observations to fit, got {len(X)}."
            )

        self._init_params_from_data(X)
        self.monitor_ = ConvergenceMonitor(
            tol=self.tol, n_iter=self.n_iter, verbose=self.verbose
        )

        for _ in range(self.n_iter):
            log_b = self._log_emission(X)
            if self.algorithm == "viterbi":
                states, log_likelihood = self._viterbi(log_b)
                gamma = np.zeros((len(X), self.n_components), dtype=float)
                gamma[np.arange(len(X)), states] = 1.0
                xi = np.zeros((self.n_components, self.n_components), dtype=float)
                np.add.at(xi, (states[:-1], states[1:]), 1.0)
            else:
                log_alpha, log_beta, gamma, log_likelihood = self._forward_backward(log_b)
                xi = self._accumulate_xi(log_alpha, log_beta, log_b)

            if not np.isfinite(log_likelihood):
                raise ValueError("HMM produced a non-finite log-likelihood.")
            if self.monitor_.report(log_likelihood):
                break
            self._do_mstep(X, gamma, xi)

        self.converged_ = bool(self.monitor_.converged)
        return self

    def score(self, X: np.ndarray, lengths: object = None) -> float:
        """Total log-likelihood of ``X`` under the fitted model."""
        if self.means_ is None:
            raise RuntimeError("GaussianHMM must be fitted before scoring.")
        X = self._validate(X)
        log_alpha = self._forward(X)
        return float(logsumexp(log_alpha[-1]))

    def predict(self, X: np.ndarray, lengths: object = None) -> np.ndarray:
        """Most likely hidden-state path for ``X`` (Viterbi decode)."""
        if self.means_ is None:
            raise RuntimeError("GaussianHMM must be fitted before predicting.")
        X = self._validate(X)
        log_b = self._log_emission(X)
        states, _ = self._viterbi(log_b)
        return states

    def predict_proba(self, X: np.ndarray, lengths: object = None) -> np.ndarray:
        """Posterior state probabilities ``p(z_t = k | X)`` -> shape ``(T, K)``."""
        if self.means_ is None:
            raise RuntimeError("GaussianHMM must be fitted before predicting.")
        X = self._validate(X)
        log_b = self._log_emission(X)
        log_alpha, log_beta, _, log_likelihood = self._forward_backward(log_b)
        return np.exp(log_alpha + log_beta - log_likelihood)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward recursion on raw observations -> ``(T, K)`` log alphas."""
        log_b = self._log_emission(X)
        log_start = np.log(np.maximum(self.startprob_, _EPS))
        log_trans = np.log(np.maximum(self.transmat_, _EPS))
        T, n = log_b.shape
        log_alpha = np.empty((T, n), dtype=float)
        log_alpha[0] = log_start + log_b[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            log_alpha[t] += log_b[t]
        return log_alpha

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.means_ is not None else "unfitted"
        return (
            f"GaussianHMM(n_components={self.n_components}, "
            f"covariance_type={self.covariance_type!r}, {state})"
        )
