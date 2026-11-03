"""
H2 mesh-convergence diagnostic.

Purpose
-------
Checks whether the three segment-pressure-drop results admit a physically
meaningful positive Richardson apparent order using

    phi(h) = phi_ext + C*h**p

with h proportional to N**(-1/3).

If p <= 0, the three grids are not demonstrably in the asymptotic Richardson
range and a classical GCI percentage is not reported.

This script intentionally DOES NOT force a positive p or manufacture a GCI.
"""

import math
from scipy.optimize import root_scalar

N_COARSE = 20120
N_MEDIUM = 90420
N_FINE = 221409

H_COARSE = N_COARSE ** (-1/3)
H_MEDIUM = N_MEDIUM ** (-1/3)
H_FINE = N_FINE ** (-1/3)

R21 = H_MEDIUM / H_FINE
R32 = H_COARSE / H_MEDIUM

def fitted_order(coarse, medium, fine):
    target = (coarse - medium) / (medium - fine)

    def equation(p):
        numerator = H_COARSE**p - H_MEDIUM**p
        denominator = H_MEDIUM**p - H_FINE**p
        return numerator / denominator - target

    # Search for a root on a broad interval while avoiding p = 0.
    intervals = [(-20, -1e-6), (1e-6, 20)]
    roots = []
    for a, b in intervals:
        # Scan to locate sign changes.
        steps = 2000
        x_prev = a
        f_prev = equation(x_prev)
        for i in range(1, steps + 1):
            x = a + (b-a) * i / steps
            f = equation(x)
            if math.isfinite(f_prev) and math.isfinite(f) and f_prev * f < 0:
                roots.append(root_scalar(equation, bracket=(x_prev, x)).root)
                break
            x_prev, f_prev = x, f

    return roots

cases = {
    "0.10 m/s segment dP": (155.90174, 147.75285, 140.17735),
    "0.26 m/s segment dP": (590.17564, 578.13760, 561.06906),
}

print(f"r21 (medium/fine) = {R21:.6f}")
print(f"r32 (coarse/medium) = {R32:.6f}")

for name, values in cases.items():
    roots = fitted_order(*values)
    print(f"\n{name}")
    if not roots:
        print("  No Richardson apparent-order root found.")
        print("  Classical GCI should not be reported.")
        continue

    for p in roots:
        print(f"  fitted p = {p:.6f}")

    positive = [p for p in roots if p > 0]
    if not positive:
        print("  No physically meaningful positive apparent order.")
        print("  The three grids are not demonstrably in the asymptotic Richardson range.")
        print("  Classical GCI should not be reported from these three solutions alone.")
