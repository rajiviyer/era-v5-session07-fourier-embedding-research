"""Entry point — run geometry proof experiments."""

from playground_v2_fourier import (
    experiment_1_pairwise,
    experiment_2_neighbors,
    experiment_3_aggregate_morph,
    experiment_4_wave_intuition,
    experiment_5_compact_sweep,
    experiment_6_indic_utf8,
)

if __name__ == "__main__":
    experiment_1_pairwise()
    experiment_2_neighbors()
    experiment_3_aggregate_morph()
    experiment_4_wave_intuition()
    experiment_5_compact_sweep()
    experiment_6_indic_utf8()
