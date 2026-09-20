"""Tests for modal synthesis."""

import wave

import numpy as np
import pytest

from sonoform.audio import (
    modal_amplitudes,
    render,
    render_plate,
    structural_decay,
    volume_velocity,
    write_wav,
)
from sonoform.geometry import FourierShape
from sonoform.plate import BRASS, solve_plate, square_mesh
from sonoform.spectrum import solve_spectrum


@pytest.fixture(scope="module")
def disc():
    return solve_spectrum(FourierShape(1.0).mesh(10, 160), k=8, order=2)


def test_centre_strike_excites_only_radial_modes(disc):
    """Physics check: a centre strike cannot drive a mode with a nodal diameter.

    Every eigenfunction of a disc except the radially symmetric ones vanishes
    at the centre, so striking there should silence them.
    """
    amps = np.abs(modal_amplitudes(disc, (0.0, 0.0)))
    loud = amps > 1e-3
    assert loud.sum() <= 3, f"expected only radial modes, got {amps.round(3)}"
    assert loud[0], "the fundamental is radially symmetric and must sound"


def test_off_centre_strike_excites_more_modes(disc):
    centre = (np.abs(modal_amplitudes(disc, (0.0, 0.0))) > 1e-3).sum()
    off = (np.abs(modal_amplitudes(disc, (0.45, 0.0))) > 1e-3).sum()
    assert off > centre


def test_strike_outside_the_shape_is_rejected(disc):
    with pytest.raises(ValueError, match="outside the shape"):
        modal_amplitudes(disc, (5.0, 5.0))


def test_rendered_audio_contains_the_predicted_partials(disc):
    """End to end: the sound really has the frequencies the spectrum claims.

    Peaks are located rather than compared against a neighbouring band. The
    disc's partials are close enough together that any fixed "quiet" offset
    lands on another partial: 25 Hz above the 2.136 partial is essentially
    the 2.296 one.
    """
    rate, f0 = 44100, 220.0
    signal = render(
        disc,
        fundamental_hz=f0,
        duration=2.0,
        sample_rate=rate,
        decay=4.0,
        brightness=0.0,
    )
    freqs = np.fft.rfftfreq(signal.size, 1.0 / rate)
    power = np.abs(np.fft.rfft(signal))

    # local maxima carrying real energy
    interior = power[1:-1]
    is_peak = (interior > power[:-2]) & (interior > power[2:])
    strong = is_peak & (interior > 0.05 * power.max())
    peak_freqs = freqs[1:-1][strong]

    expected = f0 * np.unique(np.round(disc.ratios[:4], 4))
    for want in expected:
        assert np.min(np.abs(peak_freqs - want)) < 3.0, (
            f"no peak within 3 Hz of {want:.1f}; found {np.round(peak_freqs, 1)}"
        )


def test_shape_and_duration(disc):
    rate = 22050
    signal = render(disc, duration=0.5, sample_rate=rate)
    assert signal.shape == (int(0.5 * rate),)
    assert np.abs(signal).max() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(signal))


def test_clip_starts_and_ends_silent(disc):
    """Guards against the click you get from a hard cut."""
    signal = render(disc, duration=1.0)
    assert abs(signal[0]) < 1e-6
    assert abs(signal[-1]) < 1e-6


def test_brightness_shortens_the_tail(disc):
    def tail_energy(brightness):
        s = render(disc, duration=2.0, decay=1.5, brightness=brightness)
        return float(np.sum(s[-len(s) // 4 :] ** 2))

    assert tail_energy(2.0) < tail_energy(0.0)


def test_render_validates_its_arguments(disc):
    with pytest.raises(ValueError, match="duration must be positive"):
        render(disc, duration=0.0)
    with pytest.raises(ValueError, match="fundamental_hz must be positive"):
        render(disc, fundamental_hz=-1.0)


def test_partials_above_nyquist_are_dropped_not_aliased(disc):
    """A very high fundamental would fold upper partials down into the audio."""
    signal = render(disc, fundamental_hz=12000.0, duration=0.3, sample_rate=44100)
    assert np.all(np.isfinite(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / 44100)
    power = np.abs(np.fft.rfft(signal))
    # nothing meaningful below the fundamental, which is where aliases land
    below = power[freqs < 11000.0].max()
    assert below < 0.05 * power.max()


def test_wav_roundtrip(tmp_path, disc):
    signal = render(disc, duration=0.4, sample_rate=22050)
    path = write_wav(tmp_path / "out.wav", signal, sample_rate=22050)
    assert path.exists()
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22050
        assert handle.getnframes() == signal.size
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    np.testing.assert_allclose(data / 32767.0, signal, atol=2e-4)


@pytest.fixture(scope="module")
def brass_square():
    return solve_plate(square_mesh(), BRASS, k=6)


def test_flexible_modes_have_no_net_volume(brass_square):
    for index in range(len(brass_square)):
        assert volume_velocity(brass_square, index) == pytest.approx(0.0, abs=1e-8)


def test_higher_partials_decay_faster(brass_square):
    low = structural_decay(brass_square.frequencies[0], BRASS.loss_factor)
    high = structural_decay(brass_square.frequencies[-1], BRASS.loss_factor)
    assert high > low * 2.0
    _signal, coeffs = render_plate(brass_square, (0.04, 0.025), duration=1.0)
    assert coeffs[-1]["gamma"] > coeffs[0]["gamma"]


def test_the_recording_peaks_at_a_real_partial(brass_square):
    signal, coeffs = render_plate(brass_square, (0.04, 0.025), duration=2.5)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) == pytest.approx(1.0)
    weights = [abs(complex(*item["pressure"])) for item in coeffs]
    loudest = coeffs[int(np.argmax(weights))]["hz"]
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / 44100.0)
    assert freqs[int(np.argmax(spectrum))] == pytest.approx(loudest, abs=1.5)


def test_a_strike_outside_the_plate_is_refused(brass_square):
    with pytest.raises(ValueError, match="outside"):
        render_plate(brass_square, (1.0, 1.0), duration=0.2)
