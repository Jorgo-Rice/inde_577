"""Multilayer perceptron classifier with backpropagation, written from scratch.

Mathematical contract
---------------------
For layers :math:`\\ell = 1, \\dots, L` with weights :math:`W_\\ell` and biases
:math:`b_\\ell`, a mini-batch :math:`A_0 = X` flows forward as

.. math::

    Z_\\ell = A_{\\ell-1} W_\\ell + b_\\ell, \\qquad
    A_\\ell = \\operatorname{ReLU}(Z_\\ell) \\;(\\ell < L), \\qquad
    P = \\operatorname{softmax}(Z_L).

Training minimizes mean cross-entropy plus an L2 penalty on the weights,

.. math::

    J = -\\frac{1}{m} \\sum_{i,k} Y_{ik} \\log P_{ik}
        + \\frac{\\alpha}{2} \\sum_\\ell \\lVert W_\\ell \\rVert_F^2 .

**Backpropagation** applies the chain rule from the output backwards. With
:math:`\\Delta_L = (P - Y)/m` (softmax and cross-entropy cancel neatly),

.. math::

    \\nabla_{W_\\ell} J = A_{\\ell-1}^\\top \\Delta_\\ell + \\alpha W_\\ell, \\quad
    \\nabla_{b_\\ell} J = \\mathbf{1}^\\top \\Delta_\\ell, \\quad
    \\Delta_{\\ell-1} = (\\Delta_\\ell W_\\ell^\\top)
        \\odot \\mathbb{1}[Z_{\\ell-1} > 0].

Weights start from He initialization, :math:`\\mathcal{N}(0, 2/\\text{fan-in})`,
which keeps ReLU activations at a stable scale across layers; biases start at
zero. Parameters are updated on shuffled mini-batches with **Adam**, which
rescales each parameter's step by running averages of its gradient and squared
gradient.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

Parameters = list[tuple[NDArray[np.float64], NDArray[np.float64]]]


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def forward(
    parameters: Parameters, features: NDArray[np.float64]
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Return pre-activations ``Z`` and activations ``A`` (``A[0]`` is the input)."""

    activations, pre_activations = [features], []
    for index, (weights, bias) in enumerate(parameters):
        z = activations[-1] @ weights + bias
        pre_activations.append(z)
        last = index == len(parameters) - 1
        activations.append(_softmax(z) if last else np.maximum(z, 0.0))
    return pre_activations, activations


def loss_and_gradients(
    parameters: Parameters,
    features: NDArray[np.float64],
    one_hot: NDArray[np.float64],
    alpha: float = 0.0,
) -> tuple[float, Parameters]:
    """Penalized mean cross-entropy and its gradients by backpropagation."""

    pre_activations, activations = forward(parameters, features)
    probabilities = activations[-1]
    rows = features.shape[0]
    data_loss = (
        -float(np.sum(one_hot * np.log(np.clip(probabilities, 1e-300, None)))) / rows
    )
    penalty = 0.5 * alpha * sum(float(np.sum(w**2)) for w, _ in parameters)

    gradients: Parameters = [None] * len(parameters)  # type: ignore[list-item]
    delta = (probabilities - one_hot) / rows
    for index in range(len(parameters) - 1, -1, -1):
        weights, _ = parameters[index]
        grad_w = activations[index].T @ delta + alpha * weights
        grad_b = delta.sum(axis=0)
        gradients[index] = (grad_w, grad_b)
        if index > 0:
            delta = (delta @ weights.T) * (pre_activations[index - 1] > 0)
    return data_loss + penalty, gradients


class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward neural network (ReLU hidden layers, softmax output).

    Parameters
    ----------
    hidden_layer_sizes
        Width of each hidden layer, e.g. ``(32,)`` or ``(32, 16)``.
    alpha
        L2 penalty on the weights.
    learning_rate
        Adam step size.
    batch_size
        Rows per mini-batch (capped at the number of rows).
    max_epochs
        Maximum passes over the training data.
    tol, n_iter_no_change
        Stop when the epoch loss has not improved by ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
    random_state
        Seed for initialization and shuffling.

    Attributes
    ----------
    classes_, n_features_in_
    parameters_ : list of (weights, bias)
    loss_curve_ : ndarray
        Mean training loss (with penalty) after each epoch.
    n_epochs_ : int
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (32,),
        *,
        alpha: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 200,
        tol: float = 1e-5,
        n_iter_no_change: int = 10,
        random_state: int | None = None,
    ) -> None:
        sizes = tuple(hidden_layer_sizes)
        if not sizes or any(not isinstance(s, int) or s < 1 for s in sizes):
            raise ValueError("hidden_layer_sizes must be positive integers")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if not learning_rate > 0:
            raise ValueError("learning_rate must be positive")
        for name, value in (("batch_size", batch_size), ("max_epochs", max_epochs),
                            ("n_iter_no_change", n_iter_no_change)):  # fmt: skip
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be nonnegative")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Train with mini-batch Adam until the loss stops improving."""

        rows = _validated(features)
        labels = np.asarray(targets)
        if labels.shape != (rows.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("at least two distinct classes are required")
        one_hot = np.eye(classes.size)[encoded]
        rng = np.random.default_rng(self.random_state)

        sizes = [rows.shape[1], *self.hidden_layer_sizes, classes.size]
        parameters: Parameters = [
            (
                rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)),
                np.zeros(fan_out),
            )
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        first = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]
        second = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]
        beta1, beta2, epsilon, step = 0.9, 0.999, 1e-8, 0

        batch = min(self.batch_size, rows.shape[0])
        losses, best, stale = [], np.inf, 0
        for _ in range(self.max_epochs):
            order = rng.permutation(rows.shape[0])
            epoch_loss = 0.0
            for start in range(0, rows.shape[0], batch):
                index = order[start : start + batch]
                loss, gradients = loss_and_gradients(
                    parameters, rows[index], one_hot[index], self.alpha
                )
                epoch_loss += loss * index.size
                step += 1
                updated = []
                for layer, ((w, b), (gw, gb)) in enumerate(
                    zip(parameters, gradients, strict=True)
                ):
                    mw, mb = first[layer]
                    vw, vb = second[layer]
                    mw, mb = (
                        beta1 * mw + (1 - beta1) * gw,
                        beta1 * mb + (1 - beta1) * gb,
                    )
                    vw, vb = (
                        beta2 * vw + (1 - beta2) * gw**2,
                        beta2 * vb + (1 - beta2) * gb**2,
                    )
                    first[layer], second[layer] = (mw, mb), (vw, vb)
                    correction1, correction2 = 1 - beta1**step, 1 - beta2**step
                    w = w - self.learning_rate * (mw / correction1) / (
                        np.sqrt(vw / correction2) + epsilon
                    )
                    b = b - self.learning_rate * (mb / correction1) / (
                        np.sqrt(vb / correction2) + epsilon
                    )
                    updated.append((w, b))
                parameters = updated
            epoch_loss /= rows.shape[0]
            losses.append(epoch_loss)
            if epoch_loss < best - self.tol:
                best, stale = epoch_loss, 0
            else:
                stale += 1
                if stale >= self.n_iter_no_change:
                    break

        self.parameters_ = parameters
        self.loss_curve_ = np.asarray(losses)
        self.n_epochs_ = len(losses)
        self.classes_ = classes
        self.n_features_in_ = rows.shape[1]
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Class probabilities from the softmax output (columns = ``classes_``)."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("fit must be called before prediction")
        rows = _validated(features)
        if rows.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return forward(self.parameters_, rows)[1][-1]

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Binary: log-odds of ``classes_[1]``, shape ``(n,)``; else output scores."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("fit must be called before prediction")
        rows = _validated(features)
        if rows.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        scores = forward(self.parameters_, rows)[0][-1]
        return scores[:, 1] - scores[:, 0] if scores.shape[1] == 2 else scores

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Most probable class for each row."""

        probabilities = self.predict_proba(features)  # checks the model is fitted
        return self.classes_[np.argmax(probabilities, axis=1)]


def _validated(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
