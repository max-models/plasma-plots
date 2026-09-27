"""Fluid, MHD and cold-plasma waves of struphy_plots.theory.waves, against identities, limits and
the eigenvalues of the linearized equations."""

import numpy as np
import pytest

from struphy_plots.theory import waves
from struphy_plots.theory.waves import Species, electron_ion


def _omega_of_matrix(matrix):
    """The frequencies ω = iλ of d/dt X = M X, for exp(−iωt)."""
    return 1j * np.linalg.eigvals(matrix)


def _assert_same_roots(got, expected, tol=1e-12):
    """Two sets of complex roots agree, in any order."""
    got, expected = list(np.ravel(got)), list(np.ravel(expected))
    assert len(got) == len(expected)
    for g in got:
        distances = [abs(g - e) for e in expected]
        i = int(np.argmin(distances))
        assert distances[i] <= tol * max(1.0, abs(g)), (got, expected)
        expected.pop(i)


# ----------------------------------------------------------------------------------------------
# unmagnetized


def test_light_waves():
    k = np.array([-2.0, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(waves.light_wave(k, c=2.0), 2 * np.abs(k))
    assert waves.light_wave(k).dtype == complex
    w = waves.plasma_light_wave(k, plasma_frequency=1.5, c=2.0)
    np.testing.assert_allclose(w**2, 1.5**2 + 4 * k**2)
    # the plasma light wave is the O mode across B₀
    n2_o = waves.appleton_hartree(w[2:].real, np.pi / 2, plasma_frequency=1.5, cyclotron_frequency=0.7)["O"]
    np.testing.assert_allclose(n2_o, 4 * k[2:] ** 2 / w[2:].real ** 2)


# ----------------------------------------------------------------------------------------------
# MHD


def test_magnetosonic_identities_and_limits():
    theta = np.linspace(0, np.pi, 13)
    va, cs = 1.3, 0.7
    v = waves.magnetosonic_speeds(theta, va, cs)
    np.testing.assert_allclose(v["fast"] ** 2 + v["slow"] ** 2, cs**2 + va**2)
    np.testing.assert_allclose(v["fast"] ** 2 * v["slow"] ** 2, cs**2 * va**2 * np.cos(theta) ** 2, atol=1e-14)
    assert np.all(v["slow"] <= v["shear Alfvén"] + 1e-14)
    assert np.all(v["shear Alfvén"] <= v["fast"] + 1e-14)
    # parallel: sound and Alfvén; perpendicular: fast magnetosonic, the others vanish
    parallel = waves.magnetosonic_speeds(0.0, va, cs)
    assert parallel["fast"] == pytest.approx(max(va, cs))
    assert parallel["slow"] == pytest.approx(min(va, cs))
    assert parallel["shear Alfvén"] == pytest.approx(va)
    perpendicular = waves.magnetosonic_speeds(np.pi / 2, va, cs)
    assert perpendicular["fast"] == pytest.approx(np.hypot(va, cs))
    assert perpendicular["slow"] == pytest.approx(0.0, abs=1e-14)
    k = np.linspace(-3, 3, 7)
    w = waves.mhd_waves(k[:, None], theta[None], va, cs)
    np.testing.assert_allclose(w["fast"], np.abs(k)[:, None] * v["fast"][None])
    assert w["slow"].shape == (7, 13) and w["slow"].dtype == complex


def _ideal_mhd_frequencies(k, theta, va, cs, d=0.0):
    """Positive frequencies of the linearized (Hall) MHD system for u, b (velocity units) and p/ρ₀."""
    kv = k * np.array([np.sin(theta), 0.0, np.cos(theta)])
    b0 = va * np.array([0.0, 0.0, 1.0])

    def rhs(state):
        u, b, p = state[:3], state[3:6], state[6]
        ikb = np.cross(1j * kv, b)
        du = -1j * kv * p + np.cross(ikb, b0)
        db = np.cross(1j * kv, np.cross(u, b0)) - d * np.cross(1j * kv, np.cross(ikb, b0))
        dp = -(cs**2) * 1j * kv @ u
        return np.concatenate([du, db, [dp]])

    matrix = np.stack([rhs(e) for e in np.eye(7, dtype=complex)], axis=1)
    omega = _omega_of_matrix(matrix)
    assert np.allclose(omega.imag, 0, atol=1e-10)
    return np.sort(omega.real[omega.real > 1e-9])


@pytest.mark.parametrize("theta", [0.3, 0.9, 1.4])
def test_mhd_waves_against_linear_system(theta):
    k, va, cs = 1.7, 1.0, 0.6
    expected = _ideal_mhd_frequencies(k, theta, va, cs)
    w = waves.mhd_waves(k, theta, va, cs)
    np.testing.assert_allclose(sorted(o.real for o in w.values()), expected, rtol=1e-10)


def test_dissipative_alfven_against_linear_system():
    va = 1.2
    for k, eta, nu, theta in [(1.0, 0.1, 0.03, 0.2), (3.0, 0.5, 0.0, 0.0), (10.0, 0.4, 0.05, 0.5), (2.0, 0.2, 0.2, 1.0)]:
        k_par = k * np.cos(theta)
        matrix = np.array([[-nu * k**2, 1j * k_par * va], [1j * k_par * va, -eta * k**2]])
        expected = _omega_of_matrix(matrix)
        w = waves.dissipative_alfven(k, va, resistivity=eta, viscosity=nu, theta=theta)
        got = np.array([w["forward"], w["backward"]])
        _assert_same_roots(got, expected)
        assert w["forward"].real >= 0
    # ideal limit and the equal-diffusivity damping rate (η + ν)k²/2
    assert waves.dissipative_alfven(2.0, 1.5)["forward"] == pytest.approx(3.0)
    assert waves.dissipative_alfven(2.0, 1.0, 0.1, 0.1)["forward"].imag == pytest.approx(-0.4)
    # overdamped: purely imaginary
    assert waves.dissipative_alfven(10.0, 1.0, resistivity=1.0)["forward"].real == pytest.approx(0.0)


def test_hall_mhd_parallel_limits():
    va, d = 1.3, 0.4
    w = waves.hall_mhd_parallel(np.array([1e3, 1e4]), va, d)
    k = np.array([1e3, 1e4])
    np.testing.assert_allclose(w["whistler"].real, k**2 * va * d, rtol=1e-3)
    np.testing.assert_allclose(w["ion cyclotron"].real, va / d, rtol=1e-4)
    small = waves.hall_mhd_parallel(1e-4, va, d)
    assert small["whistler"].real == pytest.approx(1e-4 * va, rel=1e-4)
    # the product and difference of the two roots: ω_w ω_ic = k²v_A², ω_w − ω_ic = k²v_A d
    kk = np.linspace(0.1, 20, 30)
    w = waves.hall_mhd_parallel(kk, va, d)
    np.testing.assert_allclose(w["whistler"] * w["ion cyclotron"], kk**2 * va**2)
    np.testing.assert_allclose(w["whistler"] - w["ion cyclotron"], kk**2 * va * d)


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.1, np.pi / 2])
def test_hall_mhd_waves_against_linear_system(theta):
    k, va, cs, d = 2.3, 1.0, 0.7, 0.5
    expected = _ideal_mhd_frequencies(k, theta, va, cs, d)
    w = waves.hall_mhd_waves(k, theta, va, cs, d)
    got = np.array([w[name].real for name in ("slow", "intermediate", "fast")])
    got = got[got > 1e-9]
    np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_hall_mhd_waves_limits():
    k, theta = np.linspace(0.1, 5, 20), 0.7
    ideal = waves.mhd_waves(k, theta, 1.0, 0.6)
    hall = waves.hall_mhd_waves(k, theta, 1.0, 0.6, ion_inertial_length=0.0)
    for name, ideal_name in (("slow", "slow"), ("intermediate", "shear Alfvén"), ("fast", "fast")):
        np.testing.assert_allclose(hall[name], ideal[ideal_name], rtol=1e-9)
    parallel = waves.hall_mhd_waves(k, 0.0, 1.0, 0.6, 0.5)
    expected = waves.hall_mhd_parallel(k, 1.0, 0.5)
    sound = 0.6 * k
    got = np.sort(np.stack([parallel[n].real for n in parallel]), axis=0)
    want = np.sort(np.stack([expected["whistler"].real, expected["ion cyclotron"].real, sound]), axis=0)
    np.testing.assert_allclose(got, want, rtol=1e-8)


# ----------------------------------------------------------------------------------------------
# cold plasma


def _determinant(n2, theta, s):
    """Stix's determinant, and a scale of its terms to measure it against."""
    matrix = _dispersion_matrix(n2, theta, s)
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    S, D, P, R, L = (s[name] for name in "SDPRL")
    a, b, c = S * sin2 + P * cos2, R * L * sin2 + P * S * (1 + cos2), P * R * L
    scale = abs(a) * n2**2 + (abs(R * L) * sin2 + abs(P * S) * (1 + cos2)) * abs(n2) + abs(c) + abs(P * D**2)
    return abs(np.linalg.det(matrix)), scale


def _dispersion_matrix(n2, theta, s):
    """Stix's n × (n × E) + K·E matrix."""
    S, D, P = s["S"], s["D"], s["P"]
    sn, cn = np.sin(theta), np.cos(theta)
    return np.array(
        [[S - n2 * cn**2, -1j * D, n2 * sn * cn], [1j * D, S - n2, 0.0], [n2 * sn * cn, 0.0, P - n2 * sn**2]]
    )


SPECIES = {
    "electrons": [Species(0.8, -1.0)],
    "overdense electrons": [Species(2.5, -1.0)],
    "electron-ion": electron_ion(1.5, 1.0, mass_ratio=25.0),
    "three species": electron_ion(1.2, 1.0, mass_ratio=16.0) + [Species(0.1, 0.03)],
}


def test_species_input_forms():
    s1 = waves.stix(2.3, Species(1.0, -1.0))
    s2 = waves.stix(2.3, (1.0, -1.0))
    s3 = waves.stix(2.3, [(1.0, -1.0)])
    for name in s1:
        assert s1[name] == pytest.approx(s2[name]) == pytest.approx(s3[name])
    electrons, ions = electron_ion(2.0, -3.0, mass_ratio=50.0, charge=2)
    assert electrons == Species(2.0, -3.0)
    assert ions.plasma_frequency == pytest.approx(2.0 * np.sqrt(2 / 50))
    assert ions.cyclotron_frequency == pytest.approx(2 * 3.0 / 50)


def test_stix_relations():
    omega = np.linspace(0.3, 4, 17)
    s = waves.stix(omega, SPECIES["electron-ion"])
    np.testing.assert_allclose(s["S"], (s["R"] + s["L"]) / 2)
    np.testing.assert_allclose(s["R"] * s["L"], s["S"] ** 2 - s["D"] ** 2)
    np.testing.assert_allclose(s["P"], 1 - (1.5**2 + 1.5**2 / 25) / omega**2)


@pytest.mark.parametrize("name", list(SPECIES))
def test_refractive_index_solves_the_determinant(name):
    species = SPECIES[name]
    omega = np.array([0.05, 0.37, 0.83, 1.3, 2.2, 4.0])
    for theta in (0.0, 0.3, 1.0, np.pi / 2):
        s = waves.stix(omega, species)
        for n2 in waves.refractive_index(omega, theta, species):
            for i in range(len(omega)):
                si = {key: value[i] for key, value in s.items()}
                determinant, scale = _determinant(n2[i], theta, si)
                assert determinant <= 1e-10 * scale


def test_refractive_index_parallel_and_perpendicular_limits():
    species = SPECIES["electron-ion"]
    omega = np.array([0.1, 0.5, 1.7, 3.0])
    s = waves.stix(omega, species)
    parallel = np.sort(np.stack(waves.refractive_index(omega, 0.0, species)), axis=0)
    np.testing.assert_allclose(parallel, np.sort(np.stack([s["R"], s["L"]]), axis=0))
    perpendicular = np.sort(np.stack(waves.refractive_index(omega, np.pi / 2, species)), axis=0)
    np.testing.assert_allclose(perpendicular, np.sort(np.stack([s["P"], s["R"] * s["L"] / s["S"]]), axis=0))


def test_appleton_hartree_matches_stix():
    omega = np.array([0.4, 0.9, 1.3, 2.0, 3.5])
    for theta in (0.2, 0.8, np.pi / 2):
        ah = waves.appleton_hartree(omega, theta, plasma_frequency=1.2, cyclotron_frequency=0.8)
        stix_roots = waves.refractive_index(omega, theta, Species(1.2, -0.8))
        np.testing.assert_allclose(
            np.sort(np.stack([ah["O"], ah["X"]]), axis=0), np.sort(np.stack(stix_roots), axis=0), rtol=1e-9
        )
    perpendicular = waves.appleton_hartree(omega, np.pi / 2, 1.2, 0.8)
    np.testing.assert_allclose(perpendicular["O"], 1 - 1.44 / omega**2)


@pytest.mark.parametrize("name", list(SPECIES))
@pytest.mark.parametrize("theta", [0.25, 0.9, 1.4])
def test_cold_plasma_waves_roots(name, theta):
    species = SPECIES[name]
    k = np.array([0.05, 0.4, 1.0, 3.0, 10.0])
    w = waves.cold_plasma_waves(k, theta, species)
    assert len(w) == len(species) + 3
    frequencies = np.stack([w[b].real for b in w])
    assert np.all(np.diff(frequencies, axis=0) >= 0)
    assert np.all(frequencies > 0)
    for i, kk in enumerate(k):
        for omega in frequencies[:, i]:
            n2 = kk**2 / omega**2
            s = waves.stix(omega, species)
            roots = waves.refractive_index(omega, theta, species)
            assert min(abs(r - n2) for r in roots) <= 1e-7 * n2, (kk, omega)
    # the top branch becomes a light wave at large k
    assert w[f"branch {len(w)}"][-1].real == pytest.approx(10.0, rel=0.05)


def test_cold_plasma_waves_parallel_and_perpendicular():
    species = SPECIES["electron-ion"]
    k = np.array([0.2, 1.0, 2.5])
    for theta, relations in ((0.0, ("R", "L", "P=0")), (np.pi / 2, ("P", "X"))):
        w = waves.cold_plasma_waves(k, theta, species)
        for branch in w.values():
            for kk, omega in zip(k, branch.real):
                if omega < 1e-6:     # the degenerate ω = 0 root across B₀
                    continue
                s = waves.stix(omega, species)
                n2 = kk**2 / omega**2
                candidates = {
                    "R": s["R"] - n2, "L": s["L"] - n2, "P=0": s["P"], "P": s["P"] - n2,
                    "X": s["R"] * s["L"] / s["S"] - n2,
                }
                residual = min(abs(candidates[r]) / (1 + abs(n2)) for r in relations)
                assert residual < 1e-7, (theta, kk, omega)


def test_cold_plasma_waves_k_to_zero_are_cutoffs():
    species = SPECIES["electron-ion"]
    w = waves.cold_plasma_waves(1e-7, 0.6, species)
    at_zero = np.sort([o.real for o in w.values() if o.real > 1e-5])
    c = waves.cutoffs(species)
    expected = np.sort(np.concatenate([c["R"], c["L"], c["P"]]))
    np.testing.assert_allclose(at_zero, expected, rtol=1e-6)


@pytest.mark.parametrize("name", list(SPECIES))
def test_cutoffs_have_zero_refractive_index(name):
    species = SPECIES[name]
    c = waves.cutoffs(species)
    assert c["P"][0] == pytest.approx(np.sqrt(sum(s.plasma_frequency**2 for s in species)))
    for key in "RLP":
        assert len(c[key]) >= 1
        for omega in c[key]:
            assert waves.stix(omega, species)[key] == pytest.approx(0.0, abs=1e-9)
            for theta in (0.0, 0.7):
                roots = waves.refractive_index(omega, theta, species)
                if key != "P" or theta != 0.0:
                    assert min(abs(r) for r in roots) < 1e-8
    if name == "electrons":
        wp, wc = 0.8, 1.0
        assert c["R"][0] == pytest.approx((wc + np.sqrt(wc**2 + 4 * wp**2)) / 2)
        assert c["L"][0] == pytest.approx((-wc + np.sqrt(wc**2 + 4 * wp**2)) / 2)


@pytest.mark.parametrize("name", list(SPECIES))
@pytest.mark.parametrize("theta", [0.3, 1.0, np.pi / 2])
def test_resonances_have_infinite_refractive_index(name, theta):
    species = SPECIES[name]
    def a_of(omega):
        s = waves.stix(omega, species)
        return s["S"] * np.sin(theta) ** 2 + s["P"] * np.cos(theta) ** 2, abs(s["S"]) + abs(s["P"])

    for omega in waves.resonances(theta, species):
        a, scale = a_of(omega)
        assert abs(a) < 1e-8 * scale
        assert a_of(omega * (1 - 1e-6))[0] * a_of(omega * (1 + 1e-6))[0] < 0
        n2_near = max(abs(r) for r in waves.refractive_index(omega * (1 + 1e-10), theta, species))
        n2_far = max(abs(r) for r in waves.refractive_index(omega * (1 + 1e-3), theta, species))
        assert n2_near > 1e4 * n2_far


def test_resonance_values():
    wp, wce, mu = 1.5, 1.0, 25.0
    species = electron_ion(wp, wce, mass_ratio=mu)
    hybrid = waves.resonances(np.pi / 2, species)
    wci, wpi = wce / mu, wp / np.sqrt(mu)
    # S = 0: upper hybrid ≈ √(ω_pe² + Ω_e²), lower hybrid ≈ √(Ω_e Ω_i) for ω_pe ≫ Ω_e
    assert len(hybrid) == 2
    assert hybrid[1] == pytest.approx(np.sqrt(wp**2 + wce**2 + wpi**2), rel=0.01)
    lower_hybrid = np.sqrt((wpi**2 + wci**2) / (1 + wp**2 / wce**2))
    assert hybrid[0] == pytest.approx(lower_hybrid, rel=0.05)
    np.testing.assert_allclose(waves.resonances(0.0, species), [wci, wce])
    assert waves.resonances(np.pi / 2, Species(1.0, -1.0)) == pytest.approx([np.sqrt(2)])
    # the large-k limits of the branches are the resonances
    w = waves.cold_plasma_waves(1e4, 0.8, species)
    limits = np.sort([o.real for o in w.values()])[:-2]   # the two light-like branches keep growing
    np.testing.assert_allclose(limits, waves.resonances(0.8, species), rtol=1e-4)


def test_faraday_rotation():
    species = [Species(1.0, -0.5)]
    omega = np.array([50.0, 200.0, 1000.0])
    approximation = 1.0**2 * 0.5 * 10.0 / (2 * omega**2)
    np.testing.assert_allclose(waves.faraday_rotation(omega, 10.0, species), approximation, rtol=3 / omega[0])
    # with ions: Σ (−Ω_s) ω_ps² L/(2cω²)
    ei = electron_ion(1.0, 0.5, mass_ratio=4.0)
    approximation = sum(-s.cyclotron_frequency * s.plasma_frequency**2 for s in ei) * 10.0 / (2 * 1e3**2)
    assert waves.faraday_rotation(1e3, 10.0, ei) == pytest.approx(approximation, rel=1e-3)
    # exact: ω(√L − √R)L/(2c)
    s = waves.stix(3.0, species)
    assert waves.faraday_rotation(3.0, 2.0, species, c=2.0) == pytest.approx(3.0 * (np.sqrt(s["L"]) - np.sqrt(s["R"])) / 2)
    assert np.isnan(waves.faraday_rotation(0.5, 1.0, species))


def test_group_velocity():
    k = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(waves.group_velocity(waves.light_wave, k), 1.0, rtol=1e-8)
    w = waves.plasma_light_wave(k, 1.0, 2.0)
    vg = waves.group_velocity(lambda kk: waves.plasma_light_wave(kk, 1.0, 2.0), k)
    np.testing.assert_allclose(vg, 4 * k / w, rtol=1e-8)
    branches = waves.group_velocity(lambda kk: waves.mhd_waves(kk, np.pi / 2, 1.0, 0.5), k)
    np.testing.assert_allclose(branches["fast"], np.hypot(1.0, 0.5), rtol=1e-8)
    assert set(branches) == {"shear Alfvén", "slow", "fast"}


# ----------------------------------------------------------------------------------------------
# cavities


def _brute_force_modes(lengths, max_index):
    """Count independent standing-wave fields per index triple from the field components."""
    d = len(lengths)
    counts = {}
    for idx in np.ndindex(*(max_index + 1,) * d):
        k = np.pi * np.array(idx) / lengths
        # E_i ∝ cos(k_i x_i) Π_{j≠i} sin(k_j x_j): it vanishes if an index j ≠ i of a walled
        # direction is zero; a third, unwalled, direction (d = 2) has k = 0 and no sine.
        allowed = [i for i in range(3) if all(idx[j] != 0 for j in range(d) if j != i)]
        kk = np.concatenate([k, np.zeros(3 - d)])
        constraint = 1 if any(kk[i] != 0 for i in allowed) else 0
        dimension = len(allowed) - constraint if np.any(kk != 0) else 0
        if dimension > 0:
            counts[idx] = dimension
    return counts


@pytest.mark.parametrize("lengths", [(1.0, 1.3, 0.7), (1.0, 1.0, 1.0), (2.0, 0.5), (1.0, 1.0)])
def test_cavity_modes_against_brute_force(lengths):
    lengths = np.array(lengths)
    modes = waves.cavity_modes(lengths, c=2.0, max_index=5)
    counts = {}
    for idx in map(tuple, modes["indices"]):
        counts[idx] = counts.get(idx, 0) + 1
    assert counts == _brute_force_modes(lengths, 5)
    np.testing.assert_allclose(modes["omega"], 2.0 * np.pi * np.linalg.norm(modes["indices"] / lengths, axis=1))
    assert np.all(np.diff(modes["omega"]) >= 0)
    assert set(modes["kind"]) == {"TE", "TM"}


def test_cavity_modes_options():
    modes = waves.cavity_modes((1.0, 2.0, 3.0), max_index=10, max_frequency=5.0)
    assert np.all(modes["omega"] <= 5.0)
    full = waves.cavity_modes((1.0, 2.0, 3.0), max_index=10)
    assert len(modes["omega"]) == np.sum(full["omega"] <= 5.0)
    # the lowest mode of a box a > b > d: TM₁₁₀ at cπ√(1/a² + 1/b²) (d the shortest side)
    box = waves.cavity_modes((3.0, 2.0, 1.0), max_index=4)
    assert tuple(box["indices"][0]) == (1, 1, 0) and box["kind"][0] == "TM"
    with pytest.raises(ValueError, match="lengths"):
        waves.cavity_modes((1.0,))


# ----------------------------------------------------------------------------------------------
# drift waves


def test_drift_wave():
    ky = np.array([0.1, 1.0, 3.0])
    np.testing.assert_allclose(waves.drift_wave(ky), ky / (1 + ky**2))
    np.testing.assert_allclose(
        waves.drift_wave(ky, kx=0.5, diamagnetic_speed=2.0, rho_s=0.3), 2 * ky / (1 + (0.25 + ky**2) * 0.09)
    )


def _hw_matrix(kx, ky, alpha, kappa, nu):
    """d/dt (φ, n) of the linearized Hasegawa–Wakatani equations with ζ = −k²φ."""
    k2 = kx**2 + ky**2
    return np.array([[-alpha / k2 - nu * k2, alpha / k2], [alpha - 1j * kappa * ky, -alpha - nu * k2]])


def test_hasegawa_wakatani_against_linear_system():
    for kx, ky, alpha, kappa, nu in [(0.0, 1.0, 1.0, 1.0, 0.0), (0.3, 0.5, 0.1, 1.0, 0.01), (1.0, 2.0, 5.0, 0.5, 0.0),
                                     (0.0, -0.7, 0.5, 1.0, 0.02)]:
        expected = _omega_of_matrix(_hw_matrix(kx, ky, alpha, kappa, nu))
        w = waves.hasegawa_wakatani(ky, kx, alpha, kappa, nu)
        _assert_same_roots([w["drift wave"], w["damped"]], expected)
        assert w["drift wave"].imag >= w["damped"].imag
        if nu == 0:
            assert w["drift wave"].imag > 0     # unstable for every k_y ≠ 0
    ky = np.linspace(0.1, 3, 10)
    w = waves.hasegawa_wakatani(ky, adiabaticity=1e4, gradient=0.7)["drift wave"]
    np.testing.assert_allclose(w.real, waves.drift_wave(ky, diamagnetic_speed=0.7), rtol=1e-3)
    assert np.all((w.imag > 0) & (w.imag < 1e-3))


# ----------------------------------------------------------------------------------------------
# continua and the TAE


def test_alfven_and_slow_continua():
    r = np.linspace(0.05, 1, 20)
    q = 1 + 2 * r**2
    R0, va, cs = 3.0, 1.5, 0.8
    w = waves.alfven_continuum(r, 3, -2, lambda x: 1 + 2 * x**2, R0, va)
    np.testing.assert_allclose(w, np.abs(-2 + 3 / q) / R0 * va)       # Struphy: (n + m/q)/R₀
    np.testing.assert_allclose(waves.alfven_continuum(r, 3, -2, q, R0, va), w)
    # zero at the rational surface q = −m/n = 1.5
    assert waves.alfven_continuum(0.5, 3, -2, 1.5, R0, va) == pytest.approx(0.0)
    slow = waves.slow_continuum(r, 3, -2, q, R0, va, lambda x: cs + 0 * x)
    np.testing.assert_allclose(slow, w * cs / np.hypot(cs, va))
    np.testing.assert_allclose(waves.parallel_wavenumber(r, 3, -2, q, R0), (-2 + 3 / q) / R0)


def test_tae_gap():
    m, n, R0, va = 1, -1, 3.0, 1.0
    r = np.linspace(0.01, 1, 20001)

    def q(x):
        return 1 + x      # q = 1.5 at r = 0.5

    center = waves.tae_frequency(1.5, R0, va)
    assert center == pytest.approx(va / (2 * 1.5 * R0))
    # no coupling: the two cylindrical continua
    uncoupled = waves.toroidal_alfven_continuum(r, m, n, q, R0, va, coupling=0.0)
    a1, a2 = waves.alfven_continuum(r, m, n, q, R0, va), waves.alfven_continuum(r, m + 1, n, q, R0, va)
    np.testing.assert_allclose(uncoupled["lower"], np.minimum(a1.real, a2.real), atol=1e-14)
    np.testing.assert_allclose(uncoupled["upper"], np.maximum(a1.real, a2.real), atol=1e-14)
    widths = []
    for eps in (0.05, 0.1, 0.2, 0.3):
        w = waves.toroidal_alfven_continuum(r, m, n, q, R0, va, coupling=eps)
        top, bottom = np.max(w["lower"].real), np.min(w["upper"].real)
        # the gap is at q = (m + ½)/|n| = 1.5, where the frequencies are ω_TAE/√(1 ± ε)
        at_gap = waves.toroidal_alfven_continuum(0.5, m, n, q, R0, va, coupling=eps)
        assert at_gap["lower"].real == pytest.approx(center / np.sqrt(1 + eps), rel=1e-12)
        assert at_gap["upper"].real == pytest.approx(center / np.sqrt(1 - eps), rel=1e-12)
        # the extrema of the branches lie within O(ε) of it (ω_TAE ∝ 1/q varies across the gap)
        assert r[np.argmax(w["lower"].real)] == pytest.approx(0.5, abs=0.15 * eps)
        assert r[np.argmin(w["upper"].real)] == pytest.approx(0.5, abs=0.15 * eps)
        assert top == pytest.approx(center / np.sqrt(1 + eps), rel=0.01)
        assert bottom == pytest.approx(center / np.sqrt(1 - eps), rel=0.01)
        assert top < center < bottom
        assert (bottom - top) / center == pytest.approx(eps, rel=0.1)
        widths.append(bottom - top)
        # each point solves the 2×2 determinant
        x = w["lower"].real[::500] ** 2 / va**2
        k1, k2 = (waves.parallel_wavenumber(r[::500], mm, n, q, R0) for mm in (m, m + 1))
        np.testing.assert_allclose((x - k1**2) * (x - k2**2), eps**2 * x**2, atol=1e-12)
    assert np.all(np.diff(widths) > 0)
    # coupling as a profile ∝ r/R₀
    w = waves.toroidal_alfven_continuum(r, m, n, q, R0, va, coupling=lambda x: 2.5 * x / R0)
    at_gap = waves.toroidal_alfven_continuum(0.5, m, n, q, R0, va, coupling=lambda x: 2.5 * x / R0)
    assert at_gap["lower"].real == pytest.approx(center / np.sqrt(1 + 2.5 * 0.5 / R0), rel=1e-12)
    with pytest.raises(ValueError, match="coupling"):
        waves.toroidal_alfven_continuum(r, m, n, q, coupling=1.0)
