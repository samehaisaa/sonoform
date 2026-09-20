"""Turn a spectrum into a sound.

An ideal membrane struck at a point rings as a sum of its modes,

``u(x, t) = sum_i a_i u_i(x) cos(omega_i t) exp(-t / tau_i)``

with ``omega_i = c sqrt(lambda_i)``. Two things make the result sound like a
drum rather than a chord on an organ.

The strike point decides *which* modes you get. A point impulse at ``p``
excites mode ``i`` in proportion to ``u_i(p)``, so a strike at the centre of a
disc is nearly silent in every mode with a nodal line through the middle.
Hitting a drum off-centre sounds different from hitting it in the middle for
exactly this reason, and the model reproduces it.

High modes decay faster than low ones. Without that the sound is a buzzy pad
that never resolves; with it you hear a bright attack settling onto the
fundamental, which is what makes a drum recognisable.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
from skfem import Basis

from sonoform.plate import PlateSpectrum
from sonoform.spectrum import _ELEMENTS, Spectrum

__all__ = [
    "AIR_DENSITY",
    "AIR_SPEED",
    "LISTENER_HEIGHT",
    "modal_amplitudes",
    "modal_velocity",
    "radiation_decay",
    "rayleigh_integral",
    "render",
    "render_plate",
    "structural_decay",
    "volume_velocity",
    "write_wav",
]

# Dry air at room temperature. The plate radiates into a half-space, which is
# the baffled Rayleigh model: the listener is above the plate and the lower
# hemisphere is not there.
AIR_DENSITY = 1.204
AIR_SPEED = 343.0
LISTENER_HEIGHT = 0.35


def modal_amplitudes(spec: Spectrum, strike) -> np.ndarray:
    """How strongly a point strike at ``strike`` excites each mode.

    The modal projection of a point impulse at ``p`` is ``u_i(p)``, so this
    evaluates every eigenfunction there. Values are normalised so the loudest
    mode is 1.

    Parameters
    ----------
    spec
        Spectrum of the domain.
    strike
        ``(x, y)`` in the same coordinates as the mesh.
    """
    basis = Basis(spec.mesh, _ELEMENTS[spec.order]())
    point = np.array([[float(strike[0])], [float(strike[1])]])
    try:
        probe = basis.probes(point)
    except ValueError as exc:  # outside the domain
        raise ValueError(f"strike point {tuple(strike)} is outside the shape") from exc

    amps = np.array(
        [float((probe @ spec.eigenvectors[:, i])[0]) for i in range(len(spec))]
    )
    peak = np.abs(amps).max()
    return amps / peak if peak > 0 else amps


def render(
    spec: Spectrum,
    fundamental_hz: float = 220.0,
    strike=None,
    duration: float = 2.0,
    sample_rate: int = 44100,
    decay: float = 1.6,
    brightness: float = 0.55,
) -> np.ndarray:
    """Synthesise the sound of striking this shape.

    Parameters
    ----------
    spec
        Spectrum of the domain. More modes give a richer sound; eight to
        sixteen is plenty.
    fundamental_hz
        Pitch to place the first partial at. The shape fixes the *ratios*
        between partials, not the absolute pitch, so this is free.
    strike
        Where the drum is hit, in mesh coordinates. Defaults to a point
        partway out from the centre, which excites both symmetric and
        antisymmetric modes and so actually reveals the chord. Striking dead
        centre silences every mode with a nodal line through it.
    duration
        Length of the clip in seconds.
    sample_rate
        Samples per second.
    decay
        Time constant of the fundamental, in seconds.
    brightness
        How much more quickly the upper partials die away, from 0 for uniform
        decay to about 1 for a sharp thud. Each mode gets
        ``tau_i = decay / (1 + brightness * (f_i / f_1 - 1))``.

    Returns
    -------
    Mono float array in [-1, 1].
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if fundamental_hz <= 0:
        raise ValueError("fundamental_hz must be positive")
    if brightness < 0.0:
        raise ValueError("brightness must be non-negative")

    if strike is None:
        # partway out along x, away from the centre and away from the rim
        xs = spec.mesh.p[0]
        strike = (0.45 * float(xs.max()), 0.0)

    amps = modal_amplitudes(spec, strike)
    ratios = spec.ratios
    freqs = fundamental_hz * ratios

    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Drop anything above Nyquist rather than let it alias into the mix.
    audible = freqs < 0.5 * sample_rate
    signal = np.zeros(n, dtype=np.float64)
    for freq, amp in zip(freqs[audible], amps[audible], strict=True):
        tau = decay / (1.0 + brightness * (freq / fundamental_hz - 1.0))
        signal += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-t / tau)

    peak = np.abs(signal).max()
    if peak > 0:
        signal /= peak

    # A few milliseconds of fade at each end, so the clip does not click.
    edge = min(int(0.004 * sample_rate), n // 2)
    if edge > 0:
        ramp = np.linspace(0.0, 1.0, edge)
        signal[:edge] *= ramp
        signal[-edge:] *= ramp[::-1]
    return signal


def structural_decay(frequency_hz: float, loss_factor: float) -> float:
    """Amplitude decay rate ``π η f`` from the structural loss factor.

    The envelope is ``exp(-π η f t)``. A higher partial dies sooner, and the
    rate is fixed by the material rather than by a brightness control.
    """
    if frequency_hz < 0.0 or loss_factor < 0.0:
        raise ValueError("frequency and loss factor must be non-negative")
    return float(np.pi * loss_factor * frequency_hz)


def radiation_decay(spec: PlateSpectrum, index: int) -> float:
    """Amplitude decay rate from the power the mode radiates into a half-space.

    A free plate's flexible modes are orthogonal to a pure heave, so the
    monopole vanishes and the compact ``(∫φ)²`` formula gives nothing. The
    power here is the self-power of the Rayleigh integral on the plate
    itself,

    ``P = (ρ ω / 4π) ∬ φ(x) φ(y) sin(k|x-y|) / |x-y| dS dS``,

    evaluated at element centroids. Dividing by twice the modal energy turns
    that power into the amplitude exponent. For a thin brass sheet this is a
    small correction on top of the structural loss; it is largest for the
    modes that actually move air.
    """
    omega = float(np.sqrt(spec.eigenvalues[index]))
    if omega == 0.0:
        return 0.0
    centroids, areas = _element_centroids(spec.mesh)
    phi = np.asarray(
        spec.basis.probes(centroids) @ spec.eigenvectors[:, index], dtype=float
    ).ravel()
    diff = centroids[:, None, :] - centroids[:, :, None]
    distance = np.sqrt(np.sum(diff * diff, axis=0))
    wave = omega / AIR_SPEED
    kernel = np.empty_like(distance)
    near = distance < 1e-12
    kernel[near] = wave
    kernel[~near] = np.sin(wave * distance[~near]) / distance[~near]
    weighted = phi * areas
    coupling = float(weighted @ kernel @ weighted)
    return float(AIR_DENSITY * omega / (4.0 * np.pi) * max(coupling, 0.0))


def volume_velocity(spec: PlateSpectrum, index: int) -> float:
    """Net volume displacement ``∫ φ dS`` of one mass-normalised mode."""
    return _integrate(spec, spec.eigenvectors[:, index], kernel=None)


def rayleigh_integral(spec: PlateSpectrum, vector, listener, wave_number: float):
    """``∫ φ exp(-i k R) / R dS`` from the plate to one listening point.

    ``listener`` is ``(x, y, z)`` in metres. ``vector`` is an Argyris
    coefficient vector, so this is the mode shape and not its vertex samples.
    """
    return _integrate(spec, vector, kernel=(listener, wave_number))


def modal_velocity(spec: PlateSpectrum, strike, impulse: float = 1.0) -> np.ndarray:
    """Initial modal velocities from a point impulse at ``strike``.

    Mass-normalised modes turn the impulse into ``q̇_i(0) = φ_i(p)``.
    The plate is at rest before the blow, so the modal displacement that
    follows is ``(q̇_i / ω_i) exp(-γ t) sin(ω_i t)``.
    """
    point = np.array([[float(strike[0])], [float(strike[1])]], dtype=float)
    try:
        probe = spec.basis.probes(point)
    except ValueError as exc:
        raise ValueError(f"strike point {tuple(strike)} is outside the plate") from exc
    phi = np.asarray(probe @ spec.eigenvectors, dtype=float).ravel()
    return impulse * phi


def render_plate(
    spec: PlateSpectrum,
    strike,
    duration: float = 2.5,
    sample_rate: int = 44100,
    impulse: float = 1.0,
    listener_height: float = LISTENER_HEIGHT,
) -> tuple[np.ndarray, list[dict]]:
    """Pressure at a point above the plate after a point strike.

    Each mode contributes the real part of its Rayleigh pressure, with
    structural decay and radiation damping on the same envelope. The returned
    signal is peak-normalised, so absolute loudness is not physical and the
    balance between partials is. The coefficient list keeps the un-normalised
    pressure phasor and the modal velocity, which is what an animation of the
    plate itself should use.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    velocities = modal_velocity(spec, strike, impulse=impulse)
    centroid = np.asarray(spec.mesh.p, dtype=float).mean(axis=1)
    listener = (float(centroid[0]), float(centroid[1]), float(listener_height))

    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = np.zeros(n, dtype=np.float64)
    coeffs: list[dict] = []

    for index, velocity in enumerate(velocities):
        omega = float(np.sqrt(spec.eigenvalues[index]))
        if omega <= 0.0:
            continue
        frequency = omega / (2.0 * np.pi)
        if frequency >= 0.5 * sample_rate:
            continue
        wave_number = omega / AIR_SPEED
        integral = rayleigh_integral(
            spec, spec.eigenvectors[:, index], listener, wave_number
        )
        gamma = structural_decay(frequency, spec.material.loss_factor)
        gamma += radiation_decay(spec, index)
        # v_n = Re[velocity * φ * exp(i ω t)], displacement lags by a quarter
        # cycle. The Rayleigh kernel already carries one factor of i ω.
        pressure = (1j * omega * AIR_DENSITY / (2.0 * np.pi)) * integral * velocity
        envelope = np.exp(-gamma * t)
        signal += envelope * (
            pressure.real * np.cos(omega * t) - pressure.imag * np.sin(omega * t)
        )
        coeffs.append(
            {
                "hz": frequency,
                "gamma": float(gamma),
                "velocity": float(velocity),
                "pressure": [float(pressure.real), float(pressure.imag)],
            }
        )

    edge = min(int(0.004 * sample_rate), n // 2)
    if edge > 0:
        ramp = np.linspace(0.0, 1.0, edge)
        signal[:edge] *= ramp
        signal[-edge:] *= ramp[::-1]
    peak = np.abs(signal).max()
    if peak > 0.0:
        signal /= peak
    return signal, coeffs


def _element_centroids(mesh):
    corners = np.asarray(mesh.p[:, mesh.t], dtype=float)
    centroids = corners.mean(axis=1)
    twice = (corners[0, 1] - corners[0, 0]) * (corners[1, 2] - corners[1, 0]) - (
        corners[0, 2] - corners[0, 0]
    ) * (corners[1, 1] - corners[1, 0])
    return centroids, 0.5 * np.abs(twice)


def _integrate(spec: PlateSpectrum, vector, kernel):
    field = spec.basis.interpolate(np.asarray(vector, dtype=float))
    phi = np.asarray(field.value)
    weight = np.asarray(spec.basis.dx)
    if kernel is None:
        return float(np.sum(phi * weight))
    listener, wave_number = kernel
    xy = np.asarray(spec.basis.global_coordinates().value)
    radius = np.sqrt(
        (xy[0] - listener[0]) ** 2
        + (xy[1] - listener[1]) ** 2
        + float(listener[2]) ** 2
    )
    green = np.exp(-1j * wave_number * radius) / radius
    return complex(np.sum(phi * green * weight))


def write_wav(path, signal: np.ndarray, sample_rate: int = 44100) -> Path:
    """Write a mono float signal to a 16-bit PCM wav file."""
    path = Path(path)
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{pcm.size}h", *pcm))
    return path
