"""Unit tests for Indic UTF-8 geometry probes."""

from __future__ import annotations

from codec import codec_from_string
from compact_codec import compact_codec_from_string
from fourier_codec import fourier_codec_from_string
from similarity import centered_cosine


def test_cross_script_love_prem_low():
    k = centered_cosine(codec_from_string("love"), codec_from_string("प्रेम"))
    f = centered_cosine(
        fourier_codec_from_string("love"), fourier_codec_from_string("प्रेम")
    )
    assert k < 0.1
    assert f < 0.1


def test_devanagari_prefix_high():
    k = centered_cosine(codec_from_string("भारत"), codec_from_string("भारती"))
    c = centered_cosine(
        compact_codec_from_string("भारत", dim=128, combine="bind"),
        compact_codec_from_string("भारती", dim=128, combine="bind"),
    )
    assert k > 0.8
    assert c > 0.8
