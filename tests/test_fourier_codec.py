"""V2 Fourier codec tests — run with: pytest tests/test_fourier_codec.py -v"""

import numpy as np
import pytest

from codec import codec_from_string, kronecker_codec, z_normalize
from fourier_codec import build_fourier_codec_matrix, fourier_codec, fourier_codec_from_string
from similarity import centered_cosine


class TestFourierCodec:
    def test_output_shape_matches_kronecker(self):
        f = fourier_codec(list(b"run"), d_c=256, d_p=32)
        k = kronecker_codec(list(b"run"), d_c=256, d_p=32)
        assert f.shape == k.shape == (256 * 32,)

    def test_dense_not_sparse(self):
        f = fourier_codec(list(b"run"), d_c=256, d_p=32)
        assert np.count_nonzero(f) == f.size

    def test_znorm_from_string(self):
        out = fourier_codec_from_string("run")
        assert out.mean() == pytest.approx(0.0, abs=1e-7)
        assert out.std() == pytest.approx(1.0, rel=1e-5)

    def test_empty_bytes_returns_zeros(self):
        assert np.all(fourier_codec([], d_p=32) == 0.0)

    def test_run_vs_RUN_low_similarity(self):
        cos = centered_cosine(
            fourier_codec_from_string("run"),
            fourier_codec_from_string("RUN"),
        )
        assert cos < 0.35

    def test_separate_vs_seperate_high_similarity(self):
        cos = centered_cosine(
            fourier_codec_from_string("separate"),
            fourier_codec_from_string("seperate"),
        )
        assert cos > 0.75

    def test_run_vs_runs_high_similarity(self):
        cos = centered_cosine(
            fourier_codec_from_string("run"),
            fourier_codec_from_string("runs"),
        )
        assert cos > 0.65

    def test_differs_from_kronecker_same_string(self):
        f = fourier_codec_from_string("runner")
        k = codec_from_string("runner")
        cos = centered_cosine(f, k)
        # Different encodings of the same bytes — not identical geometry.
        assert cos < 0.5


class TestBuildFourierMatrix:
    def test_matrix_shape(self):
        from byte_table import get_gpt2_tokenizer

        tok = get_gpt2_tokenizer()
        matrix = build_fourier_codec_matrix(tok, d_p=16)
        assert matrix.shape == (tok.n_vocab, 256 * 16)
