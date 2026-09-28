from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from numba import njit
from skimage.measure import points_in_poly
from tkinter import *
from tkinter import ttk

from neural.model import PlateNet
from neural.shapes import make_morph_contour

fs = 48000.0                # samplerate
n_Modes = 256               # max. number of modes
MAX_STATE_MAG = 10.0        # used to clamp mag to avoid exploding nonlineary modecpouling

params = {
    "size": 1.0,            # determines size/tune of plate
    "aspect": 1.0,          # aspect ratio of plate (0.5-4.0 (only for the 256 modes model, otherwise 0.5-2.0))
    "morph": 0.0,           # morphs between square, circle, triangle

    "alpha_g": 0.3322,      # damping factor
    "alpha_r": 4e-5,        # damping tilt (higher f are damped stronger)

    "tau": 1.0,             # nonlinearity threshold (higher means less nonlinearities)
    "eta": 0.01,            # mode coupling
    "lamb": 0.01,           # nonlinearity coupling strength

    "D": 18300.0,           # Material Parameters
    "rho": 7800.0, 
    "H": 0.01,

    "modes": 256,           # nubmer of modes in use

    "N_ex": 192,            # excitation signal length (hard/softer strike)
    "A": 0.5,               # excitation signal gain

    "x_e": -0.3,            # excitation and pickup positions 
    "y_e": 0.3,
    "x_l": 0.3, 
    "y_l": 0.3,
    "x_r": 0.3, 
    "y_r": 0.3,

    "impact_start": None,   # start sample of last strike
    "changed": False,       # flag to show parameter change
}

# create PlateNet model and load trained weights from saved model
plate_net = PlateNet(n_modes=n_Modes)
plate_net.load_state_dict(torch.load(
    Path(__file__).parent / "models" / "plate_nn.pt",
    map_location="cpu", weights_only=True,
))
plate_net.eval()

# predict frequency factors and gains for strike and pickups from model, convert into physical frequencies
# using material properties and size and ouptut frequencies/gains for strike and right/left pickups
@torch.no_grad()
def plate_model():
    geometry = torch.tensor([[params["morph"], params["aspect"]]], dtype=torch.float32)
    points = torch.tensor([[
        [params["x_e"], params["y_e"]],
        [params["x_l"], params["y_l"]],
        [params["x_r"], params["y_r"]],
    ]], dtype=torch.float32)

    mu, gains = plate_net(geometry, points)
    mu = mu[0].numpy().astype(np.float64)
    gains = gains[0].numpy().astype(np.float64)
    freqs = mu * np.sqrt(params["D"] / (params["rho"] * params["H"])) / (2 * np.pi * params["size"] ** 2)
    return freqs, gains[0], gains[1], gains[2]

# create distribution matrix for nonlinear mode coupling using frequency proximity and the parameters 
# eta, tau and lambda. coupled filter formulation Eq. 20 and Eq. 26from Poirot et al, 2023 
def distribution_matrix(freqs):
    diff = np.abs(freqs[:, None] - freqs[None, :])
    a = 1.0 - diff / np.mean(freqs)
    denom = a.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return np.ascontiguousarray(
        params["eta"] * params["lamb"] * a / denom - params["lamb"] * np.eye(len(freqs)),
        dtype=np.float64,
    )

# build runtime parameters from current parameters (freqs, strike, left/right pickup), calculates 
# damping factors and frequency factors for each mode and creates distribution matrix
# uses Eq. 5, Eq 25 from Poirot et al, 2023
def build_runtime():
    freqs, strike, left, right = plate_model()
    n = int(params["modes"])
    freqs, strike, left, right = freqs[:n], strike[:n], left[:n], right[:n]

    alphas = np.exp(np.minimum(params["alpha_g"] + params["alpha_r"] * freqs, 700.0))
    Z = np.exp(-alphas / fs) * np.exp(1j * 2 * np.pi * freqs / fs)

    return (
        freqs,
        np.ascontiguousarray(Z, dtype=np.complex128),
        distribution_matrix(freqs),
        np.ascontiguousarray(strike, dtype=np.float64),
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
    )

# generates excitation signal for strike position, uses Eq. 24 from Poirot et al, 2023
def excitation_signal(pos, gains):
    u = np.zeros_like(pos, dtype=np.float64)
    inside = pos < params["N_ex"]
    u[inside] = (
        params["A"] * 2.0 / params["N_ex"]
        * np.sin(np.pi * pos[inside] / params["N_ex"]) ** 2
    )
    return np.ascontiguousarray(gains[:, None] * u[None, :])

# process modal filter bank sample by sample, compute excess
@njit(cache=True, fastmath=True)
def process_block(states, Z, M, u, left, right, tau, max_state_mag, out):

    n = len(states)

    # temp. buffers for excess power above threshold and power transfer to other modes
    rectified = np.empty(n)
    T = np.empty(n)

    # process each sample in block
    for i in range(u.shape[1]):

        # compute modal power and keep only amount above threshold, Eq. 18 from Poirot et al, 2023
        for k in range(n):
            zr, zi = states[k].real, states[k].imag
            r = 0.5 * (zr * zr + zi * zi) - tau
            rectified[k] = r if r > 0.0 else 0.0

        # distribute excess power to other modes using distribution matrix
        for k in range(n):
            acc = 0.0
            for j in range(n):
                acc += M[k, j] * rectified[j]
            T[k] = acc #if acc > 0.0 else 0.0

        # update modal states using Eq. 12 from Poirot et al, 2023
        for k in range(n):
            zr, zi = states[k].real, states[k].imag
            mag2 = zr * zr + zi * zi

            # change mode amplitude with transferred energy, if-clause used for initially silent mode
            if mag2 < 1e-20:
                new_z = np.sqrt(max(2.0 * T[k], 0.0)) * Z[k] + u[k, i]
            else:
                scale2 = 1.0 + 2.0 * T[k] / mag2
                scale2 = max(scale2, 0.0)  # numerical safety
                new_z = np.sqrt(scale2) * Z[k] * states[k] + u[k, i]

            # additional clamp to avoid exploding states from nonlinear mode coupling
            mag = np.sqrt(new_z.real ** 2 + new_z.imag ** 2)
            if mag > max_state_mag:
                new_z *= max_state_mag * np.tanh(mag / max_state_mag) / mag
            states[k] = new_z

        # compute output for left/right pickup positions
        l = r = 0.0
        for k in range(n):
            value = states[k].imag
            l += left[k] * value
            r += right[k] * value

        # write output to stereo buffer
        out[i, 0], out[i, 1] = l, r

# Init modal filter bank, build current runtime parameters and process a dummy block to precompile Numba 
# before actual audio starts
states = np.zeros(n_Modes, dtype=np.complex128)
runtime = build_runtime()
_, Z0, M0, strike0, left0, right0 = runtime

process_block(
    states.copy(), Z0, M0,
    np.zeros((len(strike0), 4)),
    left0, right0,
    params["tau"], MAX_STATE_MAG,
    np.zeros((4, 2)),
)

print("Numba-Kernel kompiliert.")

# 
def callback(outdata, frames, time, status):
    global runtime

    # print warnings
    if status:
        print(status)

    # rebuild runtime parameters if parameter change
    if params["changed"]:
        runtime = build_runtime()
        params["changed"] = False

    _, Z, M, strike, left, right = runtime

    # number of modes in use
    n = len(Z)

    # sample indices of current audio block
    pos = callback.pos + np.arange(frames)

    # generate excitation for current block
    u = (
        np.zeros((n, frames), dtype=np.float64)
        if params["impact_start"] is None
        else excitation_signal(pos - params["impact_start"], strike)
    )

    # stereo output buffer for current block
    out = np.empty((frames, 2))

    # run realtime filter bank, update states and compute left/right output for current block
    process_block(
        states[:n], Z, M, u,
        left, right,
        params["tau"], MAX_STATE_MAG, out,
    )

    # advance sample position 
    callback.pos += frames

    # normalize output, replace NaN/Inf values with 0 and clip to output range [-1, 1] 
    out = np.nan_to_num(
        out / np.sqrt(n),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    outdata[:] = np.clip(out, -1.0, 1.0)

# sample counter for strike timing
callback.pos = 0

############
# GUI Part #
############

root = Tk()
root.title("Resonator Filter GUI (Numba)")
canvas_size = 300

ttk.Label(
    root,
    text="Left: strike | Right: pickup L | Middle / Shift+Right: pickup R",
).pack()

canvas = Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="white",
)

canvas.pack(pady=6)

shape_id = canvas.create_polygon(
    0, 0,
    fill="",
    outline="black",
    width=2,
)

def contour():
    return make_morph_contour(params["morph"], params["aspect"])

def point_inside(x, y):
    return points_in_poly(np.array([[x, y]]), contour())[0]

def draw():
    c = contour()

    xy = np.column_stack((
        (c[:, 0] + 1) * 0.5 * canvas_size,
        (1 - c[:, 1]) * 0.5 * canvas_size,
    ))

    canvas.coords(shape_id, *xy.ravel())

    for tag, x, y, color in (
        ("strike", params["x_e"], params["y_e"], "red"),
        ("left", params["x_l"], params["y_l"], "green"),
        ("right", params["x_r"], params["y_r"], "blue"),
    ):
        canvas.delete(tag)

        px = (x + 1) * 0.5 * (canvas_size - 1)
        py = (1 - y) * 0.5 * (canvas_size - 1)

        canvas.create_oval(
            px - 6, py - 6,
            px + 6, py + 6,
            fill=color,
            tags=tag,
        )

def set_param(name, value):
    params[name] = int(float(value)) if name == "modes" else float(value)

    if name == "modes":
        states[:] = 0.0

    if name in ("morph", "aspect"):
        for x, y in (
            ("x_e", "y_e"),
            ("x_l", "y_l"),
            ("x_r", "y_r"),
        ):
            if not point_inside(params[x], params[y]):
                params[x] = params[y] = 0.0

        draw()

    params["changed"] = True

def set_point(event, x_name, y_name):
    x = 2 * np.clip(event.x, 0, canvas_size - 1) / (canvas_size - 1) - 1
    y = 1 - 2 * np.clip(event.y, 0, canvas_size - 1) / (canvas_size - 1)

    if point_inside(x, y):
        params[x_name], params[y_name] = x, y
        params["changed"] = True
        draw()


def add_slider(label, name, lo, hi, model_param=True):
    ttk.Label(root, text=label).pack()

    command = (
        (lambda v: set_param(name, v))
        if model_param
        else (lambda v: params.__setitem__(name, float(v)))
    )

    s = ttk.Scale(
        root,
        from_=lo,
        to=hi,
        orient="horizontal",
        command=command,
    )

    s.set(params[name])
    s.pack(fill="x")

ttk.Button(
    root,
    text="Strike",
    command=lambda: params.__setitem__("impact_start", callback.pos),
).pack(pady=6)

add_slider("Size", "size", 0.1, 3.0)
add_slider("Morph", "morph", 0.0, 1.0)
add_slider("Shape", "aspect", 0.5, 4.0)

add_slider("Damping", "alpha_g", 0.0000000001, 5.0)
add_slider("Damping Tilt", "alpha_r", 0.0001, 0.01)

add_slider("NonLin Threshold Tau", "tau", 0.0001, 2.0)
add_slider("NonLin Odd Coupling Strength Eta", "eta", 0.0, 0.8)
add_slider("NonLin Coupling StrengthLambda", "lamb", 0.0001, 1)

add_slider("Excitation Length", "N_ex", 2, 192, False)
add_slider("Excitation Gain", "A", 0.0, 2.0, False)

ttk.Label(root, text="Modes").pack()

s = Scale(
    root,
    from_=1,
    to=n_Modes,
    orient=HORIZONTAL,
    resolution=1,
    command=lambda v: set_param("modes", v),
)

s.set(params["modes"])
s.pack(fill="x")

# Left click / drag: move left pickup
canvas.bind("<Button-1>", lambda e: set_point(e, "x_l", "y_l"))
canvas.bind("<B1-Motion>", lambda e: set_point(e, "x_l", "y_l"))

# Right click / drag: move right pickup
canvas.bind("<Button-3>", lambda e: set_point(e, "x_r", "y_r"))
canvas.bind("<B3-Motion>", lambda e: set_point(e, "x_r", "y_r"))

# Shift + left click / drag: move strike position
canvas.bind("<Shift-Button-1>", lambda e: set_point(e, "x_e", "y_e"))
canvas.bind("<Shift-B1-Motion>", lambda e: set_point(e, "x_e", "y_e"))

draw()

print("Initial frequencies:", np.round(runtime[0], 2))

with sd.OutputStream(
    channels=2,
    samplerate=int(fs),
    callback=callback,
):
    root.mainloop()
