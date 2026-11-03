"""Write the ParaView deformed-result VTU(s) from a finished pressure run's .rst,
in a FRESH process that does ONLY the DPF read + meshio write.

Isolating the DPF open on the large distributed .rst -- with no other heavy DPF /
pyvista work in the same process -- avoids the intermittent native-lib crash seen
when it runs in-process (the same reason scripts/_render_png.py is its own
process).  The caller (fea_crimp._export_pressure_vtu) retries this whole process
until it prints a VTU_WRITTEN= marker.

Usage:  python scripts/_export_vtu.py <case> [release|norelease]

Writes results_fea/<case>/deformed_result.vtu (full crimp, end of LS1) and, when
the second arg is "release" (the default) AND the release load step produced a
set, released_result.vtu -- on the ORIGINAL wrapped_mesh.vtu nodes, with point
fields displacement, disp_mag, von_mises_MPa, eqv_plastic_strain.  Pass
"norelease" for an UNSTABLE pressure release (post-buckling snap): its final set
has no readable stress and a blown-up shape, so released_result.vtu is skipped.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stentforge import fea_crimp as fc          # noqa: E402
from ansys.dpf import core as dpf               # noqa: E402


def main(case, write_release=True):
    out_dir = ROOT / "results_fea" / case
    rst = out_dir / "_mapdl" / "file.rst"
    mesh_vtu = out_dir / "wrapped_mesh.vtu"
    if not rst.exists():
        raise SystemExit(f"no .rst at {rst}")
    if not mesh_vtu.exists():
        raise SystemExit(f"no mesh at {mesh_vtu}")

    model = dpf.Model(str(rst))
    # crimp = last LS1 substep (full crimp); release = last set if LS2 ran (None else)
    _, crimp_set, release_set, _ = fc._set_map_pressure(model)
    written = fc.export_result_vtus(model, out_dir, crimp_set, release_set,
                                    mesh_vtu=mesh_vtu, write_release=write_release)
    # machine-readable marker the parent parses; never crash on an empty result.
    print("VTU_WRITTEN=" + ",".join(p.name for p in written))
    if written:
        print(fc._paraview_hint())
    # The files are written and stdout is about to be flushed; exit HARD with
    # os._exit so DPF's native atexit teardown (a known intermittent segfault
    # source on this distributed .rst) can never crash *after* a good run and
    # cost us the marker.  Flush first -- os._exit skips Python's buffer flush.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    _case = sys.argv[1] if len(sys.argv) > 1 else "default"
    _write_release = not (len(sys.argv) > 2 and sys.argv[2].lower() == "norelease")
    main(_case, _write_release)
