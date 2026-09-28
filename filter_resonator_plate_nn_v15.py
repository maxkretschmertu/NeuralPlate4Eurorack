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

fs = 48000.0
n_Modes = 256
MAX_STATE_MAG = 10.0

params = {
    "alpha_g": 0.3322, "alpha_r": 4e-5,
    "tau": 1.0, "eta": 0.01, "lamb": 0.01, "modes": 256,
    "size": 1.0, "aspect": 1.0, "morph": 0.0,
    "D": 18300.0, "rho": 7800.0, "H": 0.01,
    "x_e": -0.3, "y_e": 0.3,
    "x_l": 0.3, "y_l": 0.3,
    "x_r": 0.3, "y_r": 0.3,
    "N_ex": 192, "A": 0.5,
    "impact_start": None, "changed": False,
}

plate_net = PlateNet(n_modes=n_Modes)
plate_net.load_state_dict(torch.load(
    Path(__file__).parent / "models" / "plate_nn.pt",
    map_location="cpu", weights_only=True,
))
plate_net.eval()

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

def distribution_matrix(freqs):
    diff = np.abs(freqs[:, None] - freqs[None, :])
    a = 1.0 - diff / np.mean(freqs)
    np.fill_diagonal(a, 0.0)
    denom = a.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return np.ascontiguousarray(
        params["eta"] * params["lamb"] * a / denom - params["lamb"] * np.eye(len(freqs)),
        dtype=np.float64,
    )

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

def excitation_signal(pos, gains):
    u = np.zeros_like(pos, dtype=np.float64)
    inside = pos < params["N_ex"]
    u[inside] = (
        params["A"] * 2.0 / params["N_ex"]
        * np.sin(np.pi * pos[inside] / params["N_ex"]) ** 2
    )
    return np.ascontiguousarray(gains[:, None] * u[None, :])

@njit(cache=True, fastmath=True)
def process_block(states, Z, M, u, left, right, tau, max_state_mag, out):

    n = len(states)
    rectified = np.empty(n)
    T = np.empty(n)

    for i in range(u.shape[1]):
        for k in range(n):
            zr, zi = states[k].real, states[k].imag
            r = 0.5 * (zr * zr + zi * zi) - tau
            rectified[k] = r if r > 0.0 else 0.0

        for k in range(n):
            acc = 0.0
            for j in range(n):
                acc += M[k, j] * rectified[j]
            T[k] = acc if acc > 0.0 else 0.0

        for k in range(n):
            zr, zi = states[k].real, states[k].imag
            mag2 = zr * zr + zi * zi

            if mag2 < 1e-20:
                new_z = np.sqrt(2.0 * T[k]) * Z[k] + u[k, i]
            else:
                new_z = np.sqrt(1.0 + 2.0 * T[k] / mag2) * Z[k] * states[k] + u[k, i]

            mag = np.sqrt(new_z.real ** 2 + new_z.imag ** 2)
            if mag > max_state_mag:
                new_z = max_state_mag * np.tanh(mag / max_state_mag) / mag
            states[k] = new_z

        l = r = 0.0
        for k in range(n):
            value = states[k].imag
            l += left[k] * value
            r += right[k] * value

        out[i, 0], out[i, 1] = l, r

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

def callback(outdata, frames, time, status):
    global runtime

    if status:
        print(status)

    if params["changed"]:
        runtime = build_runtime()
        params["changed"] = False

    _, Z, M, strike, left, right = runtime

    n = len(Z)
    pos = callback.pos + np.arange(frames)

    u = (
        np.zeros((n, frames), dtype=np.float64)
        if params["impact_start"] is None
        else excitation_signal(pos - params["impact_start"], strike)
    )

    out = np.empty((frames, 2))

    process_block(
        states[:n], Z, M, u,
        left, right,
        params["tau"], MAX_STATE_MAG, out,
    )

    callback.pos += frames

    out = np.nan_to_num(
        out / np.sqrt(n),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    outdata[:] = np.clip(out, -1.0, 1.0)

callback.pos = 0

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
add_slider("Damping Tilt", "alpha_r", 0.0, 0.01)

add_slider("NonLin Threshold Tau", "tau", 0.0001, 2.0)
add_slider("NonLin Odd Coupling Strength Eta", "eta", 0.0, 0.8)
add_slider("NonLin Coupling StrengthLambda", "lamb", 0.0001, 10)

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

canvas.bind("<Button-1>", lambda e: set_point(e, "x_e", "y_e"))
canvas.bind("<B1-Motion>", lambda e: set_point(e, "x_e", "y_e"))

canvas.bind("<Button-3>", lambda e: set_point(e, "x_l", "y_l"))
canvas.bind("<B3-Motion>", lambda e: set_point(e, "x_l", "y_l"))

canvas.bind("<Button-2>", lambda e: set_point(e, "x_r", "y_r"))
canvas.bind("<B2-Motion>", lambda e: set_point(e, "x_r", "y_r"))

canvas.bind("<Shift-Button-3>", lambda e: set_point(e, "x_r", "y_r"))
canvas.bind("<Shift-B3-Motion>", lambda e: set_point(e, "x_r", "y_r"))

draw()

print("Initial frequencies:", np.round(runtime[0], 2))

with sd.OutputStream(
    channels=2,
    samplerate=int(fs),
    callback=callback,
):
    root.mainloop()
