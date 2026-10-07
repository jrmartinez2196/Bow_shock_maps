"""Geometry of the layered bow-shock shell, in coordinates tied to the surface.

The post-shock layers (hot/cold layer behind the reverse shock, hot/cold
layer behind the forward shock) have thicknesses measured *along the local
normal* to the shock surface. The natural way to say "is this point inside
layer k, and what are the gas conditions there?" is therefore in coordinates
attached to the surface rather than in polar angle from the star:

* the *foot point*: the point of the reverse-shock curve nearest to the
  query point. It identifies which part of the shock generated the layer
  the point sits in, so the layer's density, temperature, ... are those of
  that point of the shock;
* the signed distance ``d`` from the surface along its outward normal. The
  layers are the intervals ``[0, d1]``, ``[d1, d2]``, ``[d2, d3]`` and
  ``[d3, d4]`` of ``d``, with ``d_k`` the cumulative thickness at the foot
  point.

Polar angle from the star is not a good label for this: displacing a point
along the normal also moves it to a *smaller* polar angle, so the polar
angle of a point inside a layer is not the polar angle of the part of the
shock that produced it. Using it to look up the shock properties
evaluates them at the wrong place (for BD+43's forward shock the density is
off by ~20% near the edge of the modeled arc), and building the layer
boundaries as functions of polar angle requires reparametrizations,
extrapolations and special cases near the edge. In normal coordinates none
of that is needed.

The sky-plane/inclination transform of the line of sight is unchanged: only
the cylindrical radius ``rho`` and the height ``z`` along the symmetry axis
of each line-of-sight sample enter here (the shell is axisymmetric).
"""

import numpy as np
from scipy.spatial import cKDTree


class ShellGeometry:
    """Reverse-shock curve plus the cumulative thickness of the layers on it.

    Parameters
    ----------
    theta : array
        Polar angle grid [rad] of the reverse-shock curve, increasing, from
        ~0 (apex) to the end of the modeled arc. This is the model's own
        parameter along the surface (the angle at which the shock
        properties are tabulated), not the polar angle of a query point.
        It should be dense enough that neighbouring samples are much closer
        than a layer thickness.
    rr : array
        Normalized radius r/R0 of the curve at ``theta``.
    R0_phys : float
        Standoff distance R0 [cm].
    H_RS_hot, H_RS_cold, H_FS_cold, H_FS_hot : array
        Thickness [cm], measured along the local normal, of each layer at
        ``theta``. Layers are stacked outward in that order, starting at
        the reverse shock.

    Attributes
    ----------
    theta : array
        The input grid; sample ``j`` of every other array refers to it.
    d1, d2, d3, d4 : array
        Cumulative distance [cm] from the surface to the outer edge of the
        RS hot layer, RS cold layer (the contact discontinuity), FS cold
        layer and FS hot layer (the forward shock).
    """

    def __init__(self, theta, rr, R0_phys, H_RS_hot, H_RS_cold, H_FS_cold, H_FS_hot):
        theta = np.asarray(theta, dtype=float)
        rr = np.asarray(rr, dtype=float)
        self.theta = theta
        self.n_arc = theta.size

        R = rr * R0_phys
        self.C_rho = R * np.sin(theta)  # curve, cylindrical radius [cm]
        self.C_z = R * np.cos(theta)  # curve, height on the symmetry axis [cm]

        # Unit tangent (increasing theta) and outward unit normal of the
        # curve, in the (rho, z) plane, taken from the curve itself. This
        # is the same normal that thermodynamics.AA gives and the layer
        # thicknesses are measured along (checked numerically: they agree
        # to ~1e-3, the finite-difference noise), but computing it here
        # keeps this class self-contained and exactly consistent with the
        # curve that is being tested against.
        t_rho = np.gradient(self.C_rho, theta)
        t_z = np.gradient(self.C_z, theta)
        t_norm = np.hypot(t_rho, t_z)
        self.t_rho, self.t_z = t_rho / t_norm, t_z / t_norm
        self.n_rho, self.n_z = -self.t_z, self.t_rho  # tangent rotated by +90 deg: outward

        self.d1 = np.asarray(H_RS_hot, dtype=float)
        self.d2 = self.d1 + np.asarray(H_RS_cold, dtype=float)
        self.d3 = self.d2 + np.asarray(H_FS_cold, dtype=float)
        self.d4 = self.d3 + np.asarray(H_FS_hot, dtype=float)

        self._tree = cKDTree(np.column_stack((self.C_rho, self.C_z)))

        # Anything farther from the curve than the outermost layer is
        # outside the shell. A bounding box lets us skip the nearest-point
        # search for the (many) line-of-sight samples that are nowhere
        # near it.
        pad = float(np.nanmax(self.d4)) if self.d4.size else 0.0
        self._rho_hi = self.C_rho.max() + pad
        self._z_lo = self.C_z.min() - pad
        self._z_hi = self.C_z.max() + pad

    def _envelope(self):
        """(rho, z) [cm] of the boundary of every layer: the reverse shock
        itself and the outer edge of each of the four layers."""
        rho, z = [], []
        for d in (0.0 * self.d1, self.d1, self.d2, self.d3, self.d4):
            rho.append(self.C_rho + d * self.n_rho)
            z.append(self.C_z + d * self.n_z)
        return np.concatenate(rho), np.concatenate(z)

    def sky_extent(self, inclination):
        """Extent of the shell projected on the sky, in the intrinsic frame.

        The shell is a surface of revolution, so a point at azimuth phi of
        the boundary point (rho, z) is at (X, Y, Z) = (rho cos phi, rho sin
        phi, z), and on the sky (see `los_projection_vectorized`)
        x = ci*X - si*Z and y = Y, with ci, si the cosine and sine of the
        inclination angle used there. Maximizing over phi gives, over the
        boundary of the layers:

            x in [min(-ci*rho - si*z), max(ci*rho - si*z)],  |y| <= max(rho).

        The star (the origin) is always included.

        Parameters
        ----------
        inclination : float
            Inclination angle [rad], as passed to `los_projection_vectorized`.

        Returns
        -------
        x_min, x_max, y_half : float
            [cm]; the shell fits in x_min <= x <= x_max, |y| <= y_half.
        """
        ci, si = np.cos(inclination), np.sin(inclination)
        rho, z = self._envelope()
        x_min = min((-ci * rho - si * z).min(), 0.0)
        x_max = max((ci * rho - si * z).max(), 0.0)
        return x_min, x_max, rho.max()

    def los_reach(self, inclination):
        """Largest |z| along the line of sight [cm] at which the shell can be.

        With z the coordinate along the line of sight, z = si*X + ci*Z, so
        |z| <= si*rho + ci*|z_shell| over the boundary of the layers. A
        line of sight needs only to be integrated over |z| <= this.
        """
        ci, si = np.cos(inclination), np.sin(inclination)
        rho, z = self._envelope()
        return (si * rho + ci * np.abs(z)).max()

    def thinnest_layer(self, densities, min_weight=0.05):
        """Characteristic thickness [cm] of the thinnest layer that matters.

        Sets how finely the line of sight must be sampled: the error of the
        integral through a layer goes as (step / thickness). Layers that
        carry a negligible part of the emission must not set it (BD+43's
        forward-shock hot layer is ~0.02 arcsec thick and irrelevant), so
        each layer is weighted by its emission measure, the integral of
        n^2 over its volume, and only layers with at least ``min_weight`` of
        the total count. Sampling a layer worth less than that badly costs
        at most that fraction of the flux, so it is not worth the (much
        smaller) step. The thickness of a layer is its median over the part
        of the shell where it exists (thickness > 0).

        Parameters
        ----------
        densities : 4 arrays
            Density [cm^-3] along the shell (like ``theta``) of the RS hot,
            RS cold, FS cold and FS hot layers.
        min_weight : float
            Fraction of the total emission measure a layer needs to count
            (default 5%).

        Returns
        -------
        float or None
            The thickness [cm], or None if the shell has no layers.
        """
        thickness = [self.d1, self.d2 - self.d1, self.d3 - self.d2, self.d4 - self.d3]
        edges = [0.0 * self.d1, self.d1, self.d2, self.d3, self.d4]
        dl = np.hypot(np.gradient(self.C_rho), np.gradient(self.C_z))  # arc length elements

        weights = []
        for k, (h, n) in enumerate(zip(thickness, densities)):
            rho_mid = self.C_rho + 0.5 * (edges[k] + edges[k + 1]) * self.n_rho
            n = np.nan_to_num(np.asarray(n, dtype=float))
            weights.append(np.sum(2.0 * np.pi * np.abs(rho_mid) * np.maximum(h, 0.0) * n**2 * dl))
        total = sum(weights)
        if not total > 0:
            return None

        scales = [
            np.median(h[h > 0])
            for h, w in zip(thickness, weights)
            if w >= min_weight * total and (h > 0).any()
        ]
        return min(scales) if scales else None

    @property
    def bbox(self):
        """Bounding box of the shell in the (rho, z) plane: (rho_hi, z_lo, z_hi) [cm].

        The shell lies in rho in [0, rho_hi] and z in [z_lo, z_hi]; a
        point outside it is in no layer.
        """
        return self._rho_hi, self._z_lo, self._z_hi

    def locate(self, rho, z):
        """Foot point and signed normal distance of points in the (rho, z) plane.

        Parameters
        ----------
        rho, z : 1D arrays [cm]
            Cylindrical radius (>= 0) and height along the symmetry axis.

        Returns
        -------
        idx : int array
            Index (into ``theta`` and the other tables) of the nearest curve
            sample, i.e. the part of the shock that generated the layer the
            point would sit in.
        d : float array
            Signed distance [cm] along the outward normal at that sample;
            positive on the ISM side. ``-inf`` for points that were not
            searched because they lie outside the bounding box of the
            shell.
        valid : bool array
            False for points that cannot belong to the shell: those outside
            the bounding box, and those beyond the end of the modeled arc
            (their nearest sample is the last one but they lie past it
            along the tangent -- the shell ends there, with an end cap
            along the normal at the last sample).
        """
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)
        idx = np.zeros(rho.shape, dtype=np.intp)
        d = np.full(rho.shape, -np.inf)
        valid = np.zeros(rho.shape, dtype=bool)

        cand = np.flatnonzero((rho <= self._rho_hi) & (z >= self._z_lo) & (z <= self._z_hi))
        if cand.size == 0:
            return idx, d, valid

        points = np.column_stack((rho[cand], z[cand]))
        try:
            _, j = self._tree.query(points, workers=-1)
        except TypeError:  # scipy < 1.6 has no `workers`
            _, j = self._tree.query(points)

        dr = rho[cand] - self.C_rho[j]
        dz = z[cand] - self.C_z[j]
        d[cand] = dr * self.n_rho[j] + dz * self.n_z[j]
        t = dr * self.t_rho[j] + dz * self.t_z[j]
        idx[cand] = j
        valid[cand] = ~((j == self.n_arc - 1) & (t > 0.0))
        return idx, d, valid

    def layers(self, idx, d, valid):
        """Which layer each located point is in.

        Returns four boolean arrays: RS hot layer, RS cold layer, FS cold
        layer, FS hot layer. A point outside every layer (inside the
        reverse shock, or beyond the forward shock) is in none of them.
        """
        d1, d2, d3, d4 = self.d1[idx], self.d2[idx], self.d3[idx], self.d4[idx]
        outside_rs = valid & (d >= 0.0)
        rs_hot = outside_rs & (d <= d1)
        rs_cold = outside_rs & (d > d1) & (d <= d2)
        fs_cold = outside_rs & (d > d2) & (d <= d3)
        fs_hot = outside_rs & (d > d3) & (d <= d4)
        return rs_hot, rs_cold, fs_cold, fs_hot
