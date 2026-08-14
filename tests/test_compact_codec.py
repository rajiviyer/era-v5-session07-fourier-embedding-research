"""Compact codec tests — run with: pytest tests/test_compact_codec.py -v"""

import numpy as np
import pytest

from codec import codec_from_string
from compact_codec import compact_codec, compact_codec_from_string
from fourier_codec import fourier_codec_from_string
from similarity import centered_cosine


class TestCompactCodec:
    def test_output_shape(self):
        out = compact_codec(list(b"run"), dim=128, d_p=16)
        assert out.shape == (128,)

    def test_smaller_than_kronecker(self):
        c = compact_codec(list(b"run"), dim=256, d_p=16)
        assert c.size == 256
        assert c.size < 256 * 16

    def test_znorm_from_string(self):
        out = compact_codec_from_string("run", dim=256, d_p=16)
        assert out.mean() == pytest.approx(0.0, abs=1e-7)
        assert out.std() == pytest.approx(1.0, rel=1e-5)

    def test_empty_bytes_returns_zeros(self):
        assert np.all(compact_codec([], dim=256, d_p=16) == 0.0)

    def test_run_vs_RUN_low_similarity(self):
        cos = centered_cosine(
            compact_codec_from_string("run", dim=256, d_p=16),
            compact_codec_from_string("RUN", dim=256, d_p=16),
        )
        assert cos < 0.35

    def test_separate_vs_seperate_high_similarity(self):
        cos = centered_cosine(
            compact_codec_from_string("separate", dim=256, d_p=16),
            compact_codec_from_string("seperate", dim=256, d_p=16),
        )
        assert cos > 0.65

    def test_run_vs_runs_high_similarity(self):
        cos = centered_cosine(
            compact_codec_from_string("run", dim=256, d_p=16),
            compact_codec_from_string("runs", dim=256, d_p=16),
        )
        assert cos > 0.55

    def test_bind_strong_on_morphology(self):
        """bind (Hadamard) couples byte×position; beats add on full-vocab morph probes."""
        bind_cos = centered_cosine(
            compact_codec_from_string("run", dim=256, d_p=16, combine="bind"),
            compact_codec_from_string("runs", dim=256, d_p=16, combine="bind"),
        )
        assert bind_cos > 0.75

    def test_differs_from_full_fourier(self):
        c = compact_codec_from_string("runner", dim=256, d_p=16)
        f = fourier_codec_from_string("runner", d_p=16)
        # Fourier at d_p=16 has D=4096; compact is D=256 — different spaces.
        assert c.shape != f.shape
