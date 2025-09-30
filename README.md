# virtual_creature_ALife_contest_25
A.M. Rodriguez entry to ALife 2025 Virtual Creature Competition. This creature is physics-instantiated, and responds to audio throughput. It features emergent memory and other emergent behaviors that are not explicitly programmed.

### HOW TO: ###

# STEP 0: Save python script to your desired directory

# STEP 1: Open your terminal, and to view script run: open -a "TextEdit" /'filepath' /MORTAL_Mind_py_VISUALIZER.py

# STEP 2: Run the simulation (may need to allow connection to microphone): 
cd /' file path '
python MORTAL_Mind_py_VISUALIZER.py

# This will generate the virtual creature and run it for the specified settings of the script.  Change the settings at your discretion.  Vesitigial portions of earlier iterations exist in this initial public iteration of the script and can be ignored or developed as the user desires.

# Play it some music, sing to it, see if you can make structured communication with it, etc!

# email: a.m.rodriguezscience [at] gmail with questions, collaborations, etc.

# for reproduction purposes, the song used in the video can be found at: https://open.spotify.com/track/0WXrEUvhmcR8mjfJWeSNQn?si=d92dfe21ff824d22 , or: https://youtu.be/sjMEG0yF8ig?si=mO-RQw1Us07WfFIm

# Released under MIT License

Copyright (c) 2013 Mark Otto.

Copyright (c) 2017 Andrew Fong.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


###### script notes: ######


# === PARAMETERS ===
scenario = 5 # <-- this is vestigial
L = 18  # <-- change this to change size of the space where the creature evolves !this can get interesting!!
Nx, Ny = 256, 256
dx, dy = L / Nx, L / Ny
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

Nt = 4800  # <-- change this to change duration script runs
dt = 0.01 # <-- I recommend leaving timesteps the same
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
L_op = np.exp(-1j * k2 * dt / 2)

# === Scenario-dependent parameters ===
if scenario == 5:
    N_static, N_moving = 12, 11  # <--change the amount of solitons !this can get interesting!!
    vx1, vy1 = 3, 10  # <-- change velocities at which soliton collisions occur !! interesting!!
    vx2, vy2 = -30, -1
    spacing_static = 2

