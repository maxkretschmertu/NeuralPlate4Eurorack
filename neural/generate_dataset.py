from pathlib import Path
import numpy as np

from shapes import make_morph_contour
from plate_reference import mesh_from_contour, solve_plate

# Generates dataset for FEM reference of plate geometries, frequency factors and gains at sample points.
# Uses shapes.py and plate_reference.py to output a compressed npz file for training the NN

N_GEOMETRIES = 1000             # number of geometries
N_MODES = 256                   # number of modes 
N_POINTS = 128                  # number of sample points for gain calculation on plate
OUTPUT = Path(__file__).parent / "plate_dataset.npz"

# generate sample geometries for later FEM-Reference:
# samples morph and aspect evenly over their range and randomly combines them
def sample_geometries(n, rng):
    morph = (np.arange(n) + rng.random(n)) / n
    aspect = 0.5 + 3.5 * (np.arange(n) + rng.random(n)) / n
    rng.shuffle(aspect)
    order = rng.permutation(n)
    return morph[order], aspect[order]

# sample point coordinates from a meshed plate geometry, 
# prioritises larger triangles to avoid many close points
def sample_points(mesh, n, rng):
    t = mesh.p[:, mesh.t]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    area = 0.5 * np.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    ids = rng.choice(area.size, n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return ((1 - r1) * a[:, ids] + r1 * (1 - r2) * b[:, ids] + r1 * r2 * c[:, ids]).T

# generate FEM reference dataset for training
if __name__ == "__main__":

    # randomizers for geometry, point and train/validation split
    geometry_rng = np.random.default_rng(0)
    point_rng = np.random.default_rng(1)
    split_rng = np.random.default_rng(2)

    # generate sampled plate geometries, prepare storage arrays for frequency factors, sample points and gains
    morphs, aspects = sample_geometries(N_GEOMETRIES, geometry_rng)
    factors = np.empty((N_GEOMETRIES, N_MODES), np.float32)
    points = np.empty((N_GEOMETRIES, N_POINTS, 2), np.float32)
    point_gains = np.empty((N_GEOMETRIES, N_POINTS, N_MODES), np.float32)

    # build FEM-Mesh, sample points, solve FEM and store frequency factor results
    for i, (morph, aspect) in enumerate(zip(morphs, aspects)):
        mesh = mesh_from_contour(make_morph_contour(morph, aspect))
        points[i] = sample_points(mesh, N_POINTS, point_rng)
        factors[i], point_gains[i] = solve_plate(mesh, points[i], N_MODES)

        #progress output
        if i == 0 or (i + 1) % 25 == 0:
            print(f"{i + 1}/{N_GEOMETRIES}")

    # randomly split train/validation geometries 
    indices = np.arange(N_GEOMETRIES)
    split_rng.shuffle(indices)
    split = round(0.8 * N_GEOMETRIES)

    # save dataset
    np.savez_compressed(
        OUTPUT,
        factors=factors,
        points=points,
        point_gains=point_gains,
        morphs=morphs.astype(np.float32),
        aspects=aspects.astype(np.float32),
        train_indices=np.sort(indices[:split]),
        val_indices=np.sort(indices[split:]),
    )
    print("Saved:", OUTPUT)
