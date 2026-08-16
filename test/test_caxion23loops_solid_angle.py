import os
import sys
import unittest

import numpy as np
from scipy.integrate import quad


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import caxion23loops as loops


def boundary_integral(rho, z, radius):
    """Independent one-dimensional solid-angle reference."""
    if z == 0:
        if rho < radius:
            return 2 * np.pi
        if rho > radius:
            return 0.0
        return 0.0

    def integrand(phi):
        distance = np.sqrt(radius**2 + rho**2 + z**2
                           - 2 * radius * rho * np.cos(phi))
        return (radius * (radius - rho * np.cos(phi)) /
                (distance * (distance + z)))

    return quad(integrand, 0, 2 * np.pi, epsabs=2e-12,
                epsrel=2e-12, limit=500)[0]


class CircularDiskSolidAngleTest(unittest.TestCase):
    def test_axis_and_plane_limits(self):
        radius = 16.0
        self.assertAlmostEqual(loops.solid_angle_disk(8.0, 0.0, radius),
                               2 * np.pi)
        self.assertAlmostEqual(loops.solid_angle_disk(17.0, 0.0, radius), 0.0)
        for z in (0.1, 1.0, 20.0):
            expected = 2 * np.pi * (1-z/np.sqrt(z*z+radius*radius))
            self.assertAlmostEqual(loops.solid_angle_disk(0.0, z, radius),
                                   expected, places=13)

    def test_off_axis_against_independent_integral(self):
        radius = 16.0
        for rho, z in ((8.0, 2.0), (15.9, 0.2), (16.0, 0.2),
                       (16.1, 0.2), (30.0, 3.0), (100.0, 50.0)):
            expected = boundary_integral(rho, z, radius)
            actual = loops.solid_angle_disk(rho, z, radius)
            self.assertAlmostEqual(actual, expected, places=9)

    def test_integer_radius_has_finite_zero_modulus_core(self):
        radius = 16.0
        theta = loops.thetaics_solid_angle(33, 16, radius)
        phi = loops.phiics(theta, 0.5, R=radius)
        self.assertTrue(np.isfinite(phi).all())
        self.assertAlmostEqual(theta[0, int(radius)], 0.0)
        self.assertAlmostEqual(np.hypot(*phi[0, int(radius)]), 0.0,
                               places=12)


if __name__ == '__main__':
    unittest.main()
