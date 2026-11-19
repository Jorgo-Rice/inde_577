"""Tests for the student k-means and PCA implementations."""

import numpy as np
import pytest
from sklearn.cluster import KMeans as TrustedKMeans
from sklearn.decomposition import PCA as TrustedPCA
from sklearn.metrics import adjusted_rand_score

from dsm_project.algorithms import PCA, KMeans, create_student_algorithm
from dsm_project.algorithms.student.unsupervised import (
    kmeans_plus_plus,
    squared_distances,
)


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 1.0], [0.0, 5.0, 2.0]])
    rows = np.vstack([rng.normal(c, 0.6, size=(80, 3)) for c in centers])
    truth = np.repeat([0, 1, 2], 80)
    return rows, truth


# --------------------------------------------------------------------------
# k-means
# --------------------------------------------------------------------------


def test_squared_distances_match_hand_computation() -> None:
    rows = np.array([[0.0, 0.0], [3.0, 4.0]])
    centers = np.array([[0.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(squared_distances(rows, centers), [[0, 9], [25, 16]])


def test_kmeans_on_a_hand_computable_case() -> None:
    # Two obvious groups on a line: {0, 1, 2} and {10, 11, 12}.
    rows = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    model = KMeans(2, random_state=0).fit(rows)

    np.testing.assert_allclose(np.sort(model.cluster_centers_[:, 0]), [1.0, 11.0])
    assert model.inertia_ == pytest.approx(4.0)  # (1 + 0 + 1) twice
    assert len(set(model.labels_[:3])) == 1 and len(set(model.labels_[3:])) == 1


def test_kmeans_recovers_separated_blobs(blobs) -> None:
    rows, truth = blobs
    model = KMeans(3, random_state=1).fit(rows)
    assert adjusted_rand_score(truth, model.labels_) > 0.98


def test_kmeans_matches_trusted_kmeans_from_the_same_start(blobs) -> None:
    rows, _ = blobs
    rng = np.random.default_rng(5)
    start = rows[rng.choice(len(rows), 3, replace=False)]
    student = KMeans(3, init=start, tol=0).fit(rows)
    trusted = TrustedKMeans(3, init=start, n_init=1, tol=0, algorithm="lloyd").fit(rows)

    np.testing.assert_array_equal(student.labels_, trusted.labels_)
    np.testing.assert_allclose(
        student.cluster_centers_, trusted.cluster_centers_, atol=1e-10
    )
    assert student.inertia_ == pytest.approx(trusted.inertia_, rel=1e-10)


def test_kmeans_restarts_never_do_worse_than_one_start(blobs) -> None:
    rows, _ = blobs
    single = KMeans(5, n_init=1, random_state=3).fit(rows)
    many = KMeans(5, n_init=10, random_state=3).fit(rows)
    assert many.inertia_ <= single.inertia_ + 1e-9


def test_inertia_decreases_as_clusters_are_added(blobs) -> None:
    rows, _ = blobs
    inertia = [KMeans(k, random_state=0).fit(rows).inertia_ for k in range(1, 6)]
    assert all(a >= b for a, b in zip(inertia, inertia[1:], strict=False))
    # One cluster: inertia equals the total sum of squares about the mean.
    assert inertia[0] == pytest.approx(np.sum((rows - rows.mean(axis=0)) ** 2))


def test_kmeans_is_reproducible_and_predict_matches_labels(blobs) -> None:
    rows, _ = blobs
    first = KMeans(3, random_state=7).fit(rows)
    second = KMeans(3, random_state=7).fit(rows)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.predict(rows), first.labels_)
    assert first.transform(rows).shape == (len(rows), 3)
    assert first.score(rows) == pytest.approx(-first.inertia_)


def test_kmeans_plus_plus_spreads_starts_across_blobs(blobs) -> None:
    # k-means++ is random: it makes far-apart starts likely, not certain.
    # Uniformly random starts cover all three equal blobs with probability
    # 3!/3^3 = 0.22; k-means++ should do far better across many seeds.
    rows, truth = blobs

    def covers_all_blobs(centers):
        nearest = {truth[np.argmin(np.sum((rows - c) ** 2, axis=1))] for c in centers}
        return len(nearest) == 3

    seeds = range(200)
    plus_plus = np.mean(
        [
            covers_all_blobs(kmeans_plus_plus(rows, 3, np.random.default_rng(s)))
            for s in seeds
        ]
    )
    uniform = np.mean(
        [
            covers_all_blobs(
                rows[np.random.default_rng(s).choice(len(rows), 3, replace=False)]
            )
            for s in seeds
        ]
    )
    # Measured: about 0.82 for k-means++ versus about 0.26 for uniform starts.
    assert plus_plus > 0.7
    assert plus_plus > 2 * uniform


def test_kmeans_handles_duplicate_rows_and_empty_clusters() -> None:
    rows = np.array([[0.0, 0.0]] * 5 + [[1.0, 1.0]])
    model = KMeans(3, random_state=0).fit(rows)
    assert model.inertia_ == pytest.approx(0.0)
    start = np.array([[0.0, 0.0], [50.0, 50.0], [60.0, 60.0]])  # two far, empty starts
    fixed = KMeans(3, init=start).fit(
        np.array([[0.0], [0.1], [5.0], [5.1]]).repeat(2, axis=1)
    )
    assert len(set(fixed.labels_)) >= 2


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_clusters": 0},
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": -1.0},
        {"init": "random"},
    ],
)
def test_kmeans_rejects_invalid_hyperparameters(parameters) -> None:
    arguments = {"n_clusters": 2, **parameters}
    with pytest.raises(ValueError):
        KMeans(**arguments)


def test_kmeans_rejects_bad_data_and_early_use() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        KMeans(2).predict([[0.0]])
    with pytest.raises(ValueError, match="cannot exceed"):
        KMeans(5).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="init centers"):
        KMeans(2, init=[[0.0, 0.0]]).fit([[0.0, 0.0], [1.0, 1.0]])
    model = KMeans(2, random_state=0).fit([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    with pytest.raises(ValueError, match="wrong shape"):
        model.predict([[0.0]])


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------


def test_pca_on_points_along_a_line() -> None:
    # All variance lies along (1, 1)/sqrt(2); the second component explains none.
    t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    rows = np.column_stack([t, t]) + 3.0
    model = PCA().fit(rows)

    np.testing.assert_allclose(model.components_[0], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(model.explained_variance_ratio_, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.mean_, [3.0, 3.0])
    np.testing.assert_allclose(model.transform(rows)[:, 0], t * np.sqrt(2))


def test_pca_components_are_orthonormal_and_ordered(blobs) -> None:
    rows, _ = blobs
    model = PCA().fit(rows)
    np.testing.assert_allclose(
        model.components_ @ model.components_.T, np.eye(3), atol=1e-12
    )
    assert np.all(np.diff(model.explained_variance_) <= 1e-12)
    assert model.explained_variance_ratio_.sum() == pytest.approx(1.0)
    # Explained variance equals the variance of the scores.
    np.testing.assert_allclose(
        model.explained_variance_, model.transform(rows).var(axis=0, ddof=1)
    )


def test_pca_reconstruction_is_exact_with_all_components(blobs) -> None:
    rows, _ = blobs
    full = PCA().fit(rows)
    np.testing.assert_allclose(
        full.inverse_transform(full.transform(rows)), rows, atol=1e-12
    )
    partial = PCA(2).fit(rows)
    error = np.sum((partial.inverse_transform(partial.transform(rows)) - rows) ** 2)
    # The dropped component's variance accounts for the whole reconstruction error.
    dropped = PCA().fit(rows).explained_variance_[2] * (len(rows) - 1)
    assert error == pytest.approx(dropped, rel=1e-9)


def test_pca_matches_trusted_pca(blobs) -> None:
    rows, _ = blobs
    student = PCA(2).fit(rows)
    trusted = TrustedPCA(2).fit(rows)
    np.testing.assert_allclose(student.components_, trusted.components_, atol=1e-12)
    np.testing.assert_allclose(student.explained_variance_, trusted.explained_variance_)
    np.testing.assert_allclose(
        student.transform(rows), trusted.transform(rows), atol=1e-10
    )


def test_pca_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        PCA(0)
    with pytest.raises(ValueError, match="cannot exceed"):
        PCA(3).fit([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="two rows"):
        PCA().fit([[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="fit must be called"):
        PCA().transform([[0.0]])
    model = PCA(1).fit([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="one column per kept component"):
        model.inverse_transform([[0.0, 1.0]])


def test_registry_builds_unsupervised_models() -> None:
    assert isinstance(create_student_algorithm("k_means", n_clusters=4), KMeans)
    assert isinstance(create_student_algorithm("pca", n_components=2), PCA)
