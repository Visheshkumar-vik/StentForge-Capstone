"""Render the deformed-shape PNG from an existing combined results file, in a
FRESH process that does ONLY the render (open model + warp + screenshot).  Doing
the render in isolation -- with no prior heavy DPF QoI extraction in the same
process -- avoids the pyvista+DPF native-lib crash seen when both run together.
The caller retries this whole process until it exits 0 (the crash is intermittent).

Usage:  python scripts/_render_png.py <case>
Title (pressure, %reduction) is read from the already-written pressure_diameter.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stentforge import fea_crimp as fc          # noqa: E402
from ansys.dpf import core as dpf               # noqa: E402


def main(case):
    out_dir = ROOT / "results_fea" / case
    rst = out_dir / "_mapdl" / "file.rst"
    csv = out_dir / "pressure_diameter.csv"
    if not rst.exists():
        raise SystemExit(f"no .rst at {rst}")

    # title fields from the curve CSV (no DPF needed for these)
    p_max, red = float("nan"), float("nan")
    if csv.exists():
        rows = [r.split(",") for r in csv.read_text().splitlines()[1:] if r.strip()]
        if rows:
            P = np.array([float(r[0]) for r in rows])
            D = np.array([float(r[1]) for r in rows])
            p_max = float(P[-1])
            red = (D[0] - D[-1]) / D[0] * 100.0

    model = dpf.Model(str(rst))
    _, crimp_set, _, _ = fc._set_map_pressure(model)
    title = (f"{case}  pressure crimp  P={p_max:.3g} MPa  "
             f"OD -{red:.1f}%  (SimpleStent)")
    png = out_dir / "deformed_shape.png"
    ok = fc._deformed_png(model, crimp_set, png, title=title, method="pressure")
    print(f"RENDER_OK={ok} crimp_set={crimp_set} {png}")
    del model
    import gc
    gc.collect()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "default")
