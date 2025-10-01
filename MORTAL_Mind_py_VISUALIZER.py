#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import pandas as pd
from collections import Counter

print("Testing numpy exp:", np.exp(1j))  # Confirm np.exp works!

# === PARAMETERS ===
scenario = 5
L = 18
Nx, Ny = 256, 256
dx, dy = L / Nx, L / Ny
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

Nt = 4800
dt = 0.01
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
L_op = np.exp(-1j * k2 * dt / 2)

# === Scenario-dependent parameters ===
if scenario == 5:
    N_static, N_moving = 12, 11
    vx1, vy1 = 3, 10
    vx2, vy2 = -30, -1
    spacing_static = 2
else:
    raise ValueError("Use scenario=5 for this full model!")

# === Helper: place_grid ===
def place_grid(N, spacing, cx, cy):
    side = int(np.ceil(np.sqrt(N)))
    gx, gy = np.meshgrid(((np.arange(1, side+1) - (side+1)/2) * spacing),
                         ((np.arange(1, side+1) - (side+1)/2) * spacing))
    centers = np.column_stack((gx.ravel(), gy.ravel()))
    return centers[:N] + np.array([cx, cy])

offset_static = [-spacing_static*2, 0]
offset_moving = [spacing_static*2, 0]
static_pos = place_grid(N_static, spacing_static, *offset_static)
moving_pos = place_grid(N_moving, spacing_static, *offset_moving)

# === Build soliton lattices ===
def sech(r):
    return 1 / np.cosh(r)

psi1 = np.zeros((Nx, Ny), dtype=np.complex128)
for cx, cy in static_pos:
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    psi1 += sech(r)
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx * dy)
psi1 *= np.exp(1j * (vx1 * X + vy1 * Y))

psi2 = np.zeros((Nx, Ny), dtype=np.complex128)
for cx, cy in moving_pos:
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    psi2 += sech(r)
psi2 /= np.sqrt(np.sum(np.abs(psi2)**2) * dx * dy)
psi2 *= np.exp(1j * (vx2 * X + vy2 * Y))

psi = psi1 + psi2

# === Audio input ===
audio_amplitude = 0.85
def audio_callback(indata, frames, time, status):
    global audio_amplitude
    audio_amplitude = np.abs(indata[:, 0]).mean()
stream = sd.InputStream(callback=audio_callback)
stream.start()

# === Metrics storage ===
mass_series = np.zeros(Nt)
energy_series = np.zeros(Nt)
entropy_series = np.zeros(Nt)
annihilation_events = 0
creation_events = 0
prev_amp = np.abs(psi)
amp_drop_threshold = 0.6
amp_rise_threshold = 1.3

phase_series = np.zeros((Nt, Nx, Ny), dtype=np.float64)
amp_series = np.zeros((Nt, Nx, Ny), dtype=np.float64)
symbolic_series = []

metabolic_drive = 0.01
metabolic_decay = 0.02

plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
prev_phase = np.angle(psi)

start_time = time.time()

for t in range(Nt):
    # --- Audio modulation ---
    global_phase = 5.00185 * audio_amplitude
    psi *= np.exp(1j * global_phase)
    local_audio_phase = 5.0075 * audio_amplitude * (np.random.rand(Nx, Ny) - 0.5)
    psi *= np.exp(1j * local_audio_phase)

    # --- Metabolic "breath" ---
    psi *= np.exp(-metabolic_decay * dt)
    psi += metabolic_drive * (np.random.rand(Nx, Ny) - 0.5) * dt

    # --- Field evolution ---
    psi_hat = np.fft.fft2(psi)
    psi = np.fft.ifft2(L_op * psi_hat)
    psi *= np.exp(-1j * np.abs(psi)**2 * dt)
    psi_hat = np.fft.fft2(psi)
    psi = np.fft.ifft2(L_op * psi_hat)

    # --- Metrics ---
    amp = np.abs(psi)
    mass_series[t] = np.sum(amp**2) * dx * dy
    kinetic = np.sum(np.abs(np.fft.fft2(psi))**2 * k2) * (dx*dy) / (Nx*Ny)**2
    potential = -0.5 * np.sum(amp**4) * dx * dy
    energy_series[t] = kinetic + potential
    entropy_series[t] = -np.sum(amp**2 * np.log(amp**2 + 1e-12)) * dx * dy

    phase_series[t] = np.angle(psi)
    amp_series[t] = amp

    # --- Symbolic feedback ---
    if t % 10 == 0:
        symbolic_output = chr(65 + int((audio_amplitude * 10) % 4))
        symbolic_series.append(symbolic_output)
        print(f"Symbolic Output at t={t*dt:.2f}: {symbolic_output}")
        audio_amplitude += 0.01 * (ord(symbolic_output) % 4 - 1.5)

    # --- Mortality check ---
    if entropy_series[t] > 20.0 or np.sum(np.abs(psi)) < 1e-4:
        print(f"MIND DEATH at t={t*dt:.2f}")
        break

    # --- Annihilation & creation events ---
    ratio = amp / (prev_amp + 1e-12)
    annihilation_events += np.sum(ratio < amp_drop_threshold)
    creation_events += np.sum(ratio > amp_rise_threshold)
    prev_amp = amp

    # --- Visualization ---
    if t % 5 == 0:
        axs[0,0].cla()
        axs[0,0].imshow(amp, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='hot')
        axs[0,0].set_title(f"Amplitude | t={t*dt:.2f} | Audio Amp: {audio_amplitude:.3f}")

        axs[0,1].cla()
        axs[0,1].imshow(np.angle(psi), extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='hsv')
        axs[0,1].set_title("Phase")

        axs[1,0].cla()
        axs[1,0].imshow(amp, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='hot')
        peaks = (amp == np.maximum.reduce([amp,
                                           np.roll(amp, 1, 0), np.roll(amp, -1, 0),
                                           np.roll(amp, 1, 1), np.roll(amp, -1, 1)]))
        y_peaks, x_peaks = np.where(peaks)
        axs[1,0].scatter(x[x_peaks], y[y_peaks], color='cyan', s=10)
        axs[1,0].set_title("Amplitude + Peaks")

        axs[1,1].cla()
        axs[1,1].plot(range(t+1), energy_series[:t+1], 'k-')
        axs[1,1].set_title("Energy Evolution")
        plt.draw()
        plt.pause(0.001)

    # --- Emergent memory metrics inside loop ---
    if t % 50 == 0:
        # Symbolic sequence autocorr
        symbolic_numeric = np.array([ord(c)-65 for c in symbolic_series])
        ac = np.correlate(symbolic_numeric - np.mean(symbolic_numeric),
                          symbolic_numeric - np.mean(symbolic_numeric),
                          mode='full')
        ac = ac[ac.size//2:] / np.max(ac)
        plt.figure(100)
        plt.clf()
        plt.plot(ac)
        plt.title(f"Symbolic output autocorr | t={t*dt:.2f}")
        plt.xlabel("Lag")
        plt.ylabel("Normalized correlation")
        plt.draw()
        plt.pause(0.001)

        # Local phase change magnitude
        phase_change = np.angle(np.exp(1j*(np.angle(psi)-prev_phase)))
        plt.figure(101)
        plt.clf()
        plt.imshow(np.abs(phase_change), cmap='viridis', origin='lower')
        plt.title(f"Phase change magnitude | t={t*dt:.2f}")
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)
        prev_phase = np.angle(psi)
# === End of loop ===
end_time = time.time()
stream.stop()

# === Final results ===
print(f"Elapsed time: {end_time - start_time:.3f} seconds")
print(f"Final mass: {mass_series[t]:.4f}")
print(f"Final energy: {energy_series[t]:.4f}")
print(f"Final entropy: {entropy_series[t]:.4f}")
print(f"Total annihilation events: {annihilation_events}")
print(f"Total creation events: {creation_events}")

# --- Emergent memory post-loop metrics ---
if len(symbolic_series) == 0:
    print("\nNo symbolic outputs generated — skipping memory metrics.")
else:
    symbolic_numeric = np.array([ord(c)-65 for c in symbolic_series])

    # 1️⃣ Symbol distribution
    counts = Counter(symbolic_series)
    total_symbols = sum(counts.values())
    print("\nSymbol distribution (A-D):")
    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]} ({counts[k]/total_symbols:.2%})")

    plt.figure()
    plt.bar(counts.keys(), [counts[k] for k in sorted(counts.keys())])
    plt.title("Symbol distribution")
    plt.xlabel("Symbol")
    plt.ylabel("Count")
    plt.show()

    # 2️⃣ Transition matrix (NumPy-based, no Pandas)
    symbols = sorted(set(symbolic_series))
    if len(symbols) > 1:
        idx = {s: i for i, s in enumerate(symbols)}
        matrix = np.zeros((len(symbols), len(symbols)), float)

        for prev, curr in zip(symbolic_series[:-1], symbolic_series[1:]):
            matrix[idx[prev], idx[curr]] += 1

        # Normalize rows to probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix /= row_sums

        print("\nTransition probability matrix:")
        for i, s in enumerate(symbols):
            row = " ".join(f"{matrix[i,j]:.2f}" for j in range(len(symbols)))
            print(f"{s}: {row}")

        # Plot
        plt.figure()
        plt.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(label="Transition probability")
        plt.xticks(range(len(symbols)), symbols)
        plt.yticks(range(len(symbols)), symbols)
        plt.title("Symbol Transition Matrix")
        plt.xlabel("Next Symbol")
        plt.ylabel("Current Symbol")
        plt.show()
    else:
        print("\nOnly one symbol observed — no transitions to compute.")

    # 3️⃣ Autocorrelation of symbolic sequence
    if len(symbolic_numeric) > 1:
        def autocorr(x):
            result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
            return result[result.size//2:] / np.max(result)

        symbolic_ac = autocorr(symbolic_numeric)
        plt.figure()
        plt.plot(symbolic_ac, 'o-')
        plt.title("Symbolic output autocorrelation (post-loop)")
        plt.xlabel("Lag")
        plt.ylabel("Normalized correlation")
        plt.show()

    # 4️⃣ Correlation of symbolic output with mean phase change magnitude
    phase_change_magnitude = np.zeros(len(symbolic_series))
    prev_phase = np.angle(psi)
    for i, t_idx in enumerate(range(0, Nt, 10)):
        current_phase = phase_series[t_idx]
        phase_change_magnitude[i] = np.mean(
            np.abs(np.angle(np.exp(1j*(current_phase - prev_phase))))
        )
        prev_phase = current_phase

    plt.figure()
    plt.plot(phase_change_magnitude, symbolic_numeric[:len(phase_change_magnitude)], 'o')
    plt.xlabel("Mean phase change magnitude")
    plt.ylabel("Symbol index (A=0, B=1, ...)")
    plt.title("Correlation: symbolic output vs phase change")
    plt.show()


