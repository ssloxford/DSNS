import numpy as np
import pytest

from dsns.constellation import IDHelper, WalkerConstellation, WalkerISLHelper
from dsns.helpers import EARTH_RADIUS


def _old_isls(satellites, P, N, intra, inter, disable_seam):
    """Reference implementation of the pre-phase_offset linking rule."""
    isls = []
    for i in range(P):
        for j in range(N):
            sat = satellites[i * N + j]
            if intra:
                isls.append((sat.sat_id, satellites[i * N + (j + 1) % N].sat_id))
            if inter is not None:
                if disable_seam and i == P - 1:
                    continue
                isls.append((sat.sat_id, satellites[((i + 1) % P) * N + (j + inter) % N].sat_id))
    return isls


def _build(P, N, F, inter, **helper_kwargs):
    po = F * 360 / (P * N)
    helper = WalkerISLHelper(P, N, True, inter, **helper_kwargs)
    con = WalkerConstellation("w", P, N, 53.0, 550e3 + EARTH_RADIUS, 0.0, 360.0, po, helper, IDHelper())
    return con, helper, po


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize("inter", [None, 0, 2])
def test_default_links_unchanged(inter, disable):
    con, helper, _ = _build(4, 6, 1, inter, disable_cross_seam_links=disable)
    assert helper.seam_shift == 0
    got = helper.get_isls(con.satellites, con.satellite_positions)
    assert got == _old_isls(con.satellites, 4, 6, True, inter, disable)


def test_phase_offset_shifts_only_seam_links():
    P, N, F, inter = 6, 8, 3, 1
    po = F * 360 / (P * N)
    con, _, _ = _build(P, N, F, inter)
    default = WalkerISLHelper(P, N, True, inter).get_isls(con.satellites, con.satellite_positions)
    helper = WalkerISLHelper(P, N, True, inter, phase_offset=po)
    shifted = helper.get_isls(con.satellites, con.satellite_positions)
    assert helper.seam_shift == F
    assert len(shifted) == len(default)
    diff = [(a, b) for a, b in zip(default, shifted) if a != b]
    assert len(diff) == N  # only the N links leaving the last plane change
    for (old_src, old_dst), (src, dst) in zip(default, shifted):
        if old_src != src or old_dst != dst:
            assert old_src == src
            idx = [s.sat_id for s in con.satellites]
            assert idx.index(dst) == (idx.index(old_dst) // N) * N + (idx.index(old_dst) % N + F) % N
            assert idx.index(src) // N == P - 1 and idx.index(dst) // N == 0


def test_seam_shift_formula():
    assert WalkerISLHelper(6, 8, True, 1, phase_offset=0).seam_shift == 0
    assert WalkerISLHelper(6, 8, True, 1, phase_offset=22.5).seam_shift == 3
    assert WalkerISLHelper(12, 10, True, 1, phase_offset=15.0).seam_shift == 5
    assert WalkerISLHelper(24, 22, True, 2, phase_offset=11 * 360 / (24 * 22)).seam_shift == 11
    # Starlink and Iridium preset phasing: seam needs no shift
    assert WalkerISLHelper(72, 22, True, 2, phase_offset=(65 / 72) * 360.0).seam_shift == 0
    assert WalkerISLHelper(6, 11, True, 5, phase_offset=180.0).seam_shift == 0


def test_starlink_parameters_seam_matches_normal():
    P, N, inter = 72, 22, 2
    po = (65 / 72) * 360.0
    helper = WalkerISLHelper(P, N, True, inter, phase_offset=po)
    con = WalkerConstellation("s", P, N, 53.0, 550e3 + EARTH_RADIUS, 0.0, 360.0, po, helper, IDHelper())
    pos = con.satellite_positions
    idx = {s.sat_id: k for k, s in enumerate(con.satellites)}
    normal, seam = [], []
    for a, b in helper.get_isls(con.satellites, pos):
        ia, ib = idx[a], idx[b]
        if ia // N != ib // N:
            (seam if ia // N == P - 1 else normal).append(np.linalg.norm(pos[ia] - pos[ib]))
    assert np.mean(seam) == pytest.approx(np.mean(normal), rel=1e-2)
    default = WalkerISLHelper(P, N, True, inter).get_isls(con.satellites, pos)
    assert helper.get_isls(con.satellites, pos) == default


@pytest.mark.parametrize("P,N,F,inter", [(12, 10, 5, 1), (12, 10, 7, 2), (8, 6, 5, 0)])
def test_seam_links_as_long_as_other_interplane_links(P, N, F, inter):
    po = F * 360 / (P * N)
    con, _, _ = _build(P, N, F, inter)
    pos = con.satellite_positions
    idx = {s.sat_id: k for k, s in enumerate(con.satellites)}

    def lengths(helper):
        normal, seam = [], []
        for a, b in helper.get_isls(con.satellites, pos):
            ia, ib = idx[a], idx[b]
            if ia // N == ib // N:
                continue
            (seam if ia // N == P - 1 else normal).append(np.linalg.norm(pos[ia] - pos[ib]))
        return np.mean(normal), np.mean(seam)

    normal, seam_old = lengths(WalkerISLHelper(P, N, True, inter))
    normal2, seam_new = lengths(WalkerISLHelper(P, N, True, inter, phase_offset=po))
    assert normal == pytest.approx(normal2)
    assert seam_new == pytest.approx(normal, rel=1e-3)
    assert seam_old > 1.1 * normal
