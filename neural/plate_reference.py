import numpy as np
import triangle as tr
from scipy.sparse.linalg import eigsh
from skfem import MeshTri, Basis, ElementTriMorley, BilinearForm, asm
from skfem.helpers import dd, ddot, trace

POISSON = 0.3


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    Hu, Hv = dd(u), dd(v)
    return (1 - POISSON) * ddot(Hu, Hv) + POISSON * trace(Hu) * trace(Hv)


def mesh_from_contour(contour, max_area=0.00025):
    vertices = np.asarray(contour, dtype=np.float64)
    if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]

    i = np.arange(len(vertices))
    result = tr.triangulate(
        {"vertices": vertices, "segments": np.column_stack((i, np.roll(i, -1)))},
        f"pq28a{max_area}",
    )
    return MeshTri(
        np.ascontiguousarray(result["vertices"].T),
        np.ascontiguousarray(result["triangles"].T),
    )


def solve_plate(mesh, points, n_modes=32):
    basis = Basis(mesh, ElementTriMorley())
    K, M = asm(stiffness, basis), asm(mass, basis)

    fixed = basis.get_dofs().nodal["u"]
    free = np.setdiff1d(np.arange(basis.N), fixed)
    Mf = M[free][:, free]

    values, modes = eigsh(K[free][:, free], k=n_modes, M=Mf, sigma=0.0, which="LM")
    order = np.argsort(values)
    values, modes = values[order], modes[:, order]
    modes /= np.sqrt(np.sum(modes * (Mf @ modes), axis=0))[None, :]

    full = np.zeros((basis.N, n_modes))
    full[free] = modes
    gains = np.asarray(basis.probes(np.asarray(points).T) @ full)

    t = mesh.p[:, mesh.t]
    area = 0.5 * np.abs(
        (t[0, 1] - t[0, 0]) * (t[1, 2] - t[1, 0])
        - (t[0, 2] - t[0, 0]) * (t[1, 1] - t[1, 0])
    ).sum()

    return np.sqrt(values) * area, gains
