"""
Tests for EngagementModel (autoencoder) — no GPU needed, CPU only.
"""
import numpy as np
import pytest

from backend.app.model.engagement_model import EngagementModel
from backend.app.model.implicit_aggregator import N_TOTAL, LATENT_FEATURE_NAMES


@pytest.fixture
def model():
    return EngagementModel()


def test_latent_dim():
    assert len(LATENT_FEATURE_NAMES) == 3


def test_fit_encodes_correctly(model, tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((30, N_TOTAL)).astype(np.float32)

    losses = model.fit(X, epochs=5)
    assert len(losses) == 5
    assert all(isinstance(l, float) for l in losses)
    assert model.is_fitted


def test_encode_output_shape(model):
    rng = np.random.default_rng(0)
    X = rng.random((20, N_TOTAL)).astype(np.float32)
    model.fit(X, epochs=3)

    latent = model.encode(X)
    assert latent.shape == (20, 3)


def test_cold_start_rows_are_zero(model):
    rng = np.random.default_rng(1)
    X = rng.random((10, N_TOTAL)).astype(np.float32)
    # Make first two rows all-zero (cold start)
    X[0] = 0.0
    X[1] = 0.0

    model.fit(X, epochs=3)
    latent = model.encode(X)

    assert np.all(latent[0] == 0.0), "Cold-start row must produce zero latent"
    assert np.all(latent[1] == 0.0), "Cold-start row must produce zero latent"


def test_encode_learner_single(model):
    rng = np.random.default_rng(7)
    X = rng.random((15, N_TOTAL)).astype(np.float32)
    model.fit(X, epochs=3)

    vec = X[0].tolist()
    result = model.encode_learner(vec)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_cold_start_encode_learner(model):
    rng = np.random.default_rng(99)
    X = rng.random((10, N_TOTAL)).astype(np.float32)
    model.fit(X, epochs=3)

    result = model.encode_learner([0.0] * N_TOTAL)
    assert result == [0.0, 0.0, 0.0]


def test_save_load_roundtrip(model, tmp_path):
    rng = np.random.default_rng(5)
    X = rng.random((15, N_TOTAL)).astype(np.float32)
    model.fit(X, epochs=3)
    latent_before = model.encode(X)

    model.save(tmp_path / "eng_model")
    loaded = EngagementModel.load(tmp_path / "eng_model")

    assert loaded.is_fitted
    latent_after = loaded.encode(X)
    np.testing.assert_allclose(latent_before, latent_after, atol=1e-5)


def test_too_few_samples_no_crash(model):
    """With fewer than min_samples, model initialises without error."""
    X = np.zeros((3, N_TOTAL), dtype=np.float32)
    losses = model.fit(X, epochs=5, min_samples=5)
    assert losses == []
    assert model.is_fitted
