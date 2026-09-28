import numpy as np

MAX_HALF_SIZE = 0.9

# generate morphing contour of plate geometry from morph and aspect parameters by interpolating the radial distance 
# between the middle point and the shape boundary for each angle

def make_morph_contour(morph, aspect=1.0, n_points=512):
    if aspect <= 0:
        raise ValueError("aspect must be positive")

    w, h = ((MAX_HALF_SIZE, MAX_HALF_SIZE / aspect)
            if aspect >= 1 else (MAX_HALF_SIZE * aspect, MAX_HALF_SIZE))

    # sample angles add angles to avoid missing corners of rectangles/triangles
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    critical = np.array([[w, h], [-w, h], [-w, -h], [w, -h], [0, h]])
    theta = np.sort(np.unique(np.r_[theta, np.mod(np.arctan2(critical[:, 1], critical[:, 0]), 2 * np.pi)]))

    # compute direction vector for each angle
    c, s = np.cos(theta), np.sin(theta)

    # radial distance from center for rectangle
    rectangle = np.minimum(w / np.maximum(np.abs(c), 1e-12),
                           h / np.maximum(np.abs(s), 1e-12))

    # radial distance from center for ellipse
    ellipse = 1 / np.sqrt((c / w) ** 2 + (s / h) ** 2)

    # radial distance from center for triangle
    coeff = np.vstack((2 * c / w + s / h, -2 * c / w + s / h, -s / h))
    triangle = np.full_like(coeff, np.inf)
    np.divide(1.0, coeff, out=triangle, where=coeff > 1e-12)
    triangle = triangle.min(axis=0)

    # morph rectangle --> ellipse for 0-0.5, ellipse --> triangle for 0.5-1
    morph = float(np.clip(morph, 0, 1))
    if morph <= 0.5:
        t = 2 * morph
        radius = (1 - t) * rectangle + t * ellipse
    else:
        t = 2 * morph - 1
        radius = (1 - t) * ellipse + t * triangle

    # convert polar back to x,y coordinates
    return np.column_stack((radius * c, radius * s))
