"""Isaac regression: release layouts, ten targets, optical ownership and legacy objects."""

import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
from revosim.launch import configure

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--variant", choices=["MT", "MX"], default="MT")
p.add_argument("--mode", choices=["sweep", "single", "multi"], default="sweep")
p.add_argument("--output", type=Path, default=Path("outputs/variant-check"))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(configure(a), multi_gpu=False).app
try:
    import numpy as np, torch
    from scipy.spatial.transform import Rotation
    from revosim.world import World
    from revosim.resources import ASSETS, SIDES, FINGERS, release_layout, pressure_count
    from revosim.sweep_motion import Sweep

    names = (
        [f"ball_{s}_{f}" for s in SIDES for f in FINGERS]
        if a.mode == "sweep"
        else ["sphere"]
        if a.mode == "single"
        else ["sphere", "cube", "polyhedron"]
    )
    w = World(
        names,
        device=a.device,
        variant=a.variant,
        probe_radius=0.004 if a.mode == "sweep" else 0.007,
        observation_hz=10,
        hand_spacing=0.32,
    )
    report = {"variant": a.variant, "mode": a.mode, "phases": {}}
    if a.mode == "sweep":
        plan = json.loads((ASSETS / "sweep_routes.json").read_text())[a.variant]
        for side, hand in w.hands.items():
            q = hand.data.default_joint_pos.clone()
            for n, value in plan[side]["splay"].items():
                q[0, hand.joint_names.index(n)] = value
            hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            hand.set_joint_position_target(q)
        for _ in range(10):
            w.step()

        def transform(side, link):
            h = w.hands[side]
            pose = (
                h.data.body_link_state_w[0, h.body_names.index(link), :7].cpu().numpy()
            )
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            T[:3, 3] = pose[:3]
            return T

        motion = Sweep(plan, transform)
        for phase, routes in motion.routes.items():
            for i, (n, r) in enumerate(zip(names, routes)):
                k = len(r["points"]) // 2
                point = r["points"][k]
                normal = r["normals"][k]
                w.reset_object(point + normal * 0.006, object_name=n)
                w.drive_object(point + normal * motion.gap(phase), object_name=n)
            w.spring_stiffness = {n: 180.0 if phase == "tip" else 360.0 for n in names}
            peak = np.zeros(10)
            for _ in range(12):
                w.step()
                v = w.observe()
                assert v["pressure"].shape == (2, pressure_count(a.variant))
                for h, side in enumerate(SIDES):
                    for i, f in enumerate(FINGERS):
                        g = next(
                            g
                            for g in release_layout(side, a.variant)
                            if g["region"] == plan[side]["phases"][phase][f]["region"]
                        )
                        raw = (
                            w.sensors[side].banks[i].get_pressure()[0, 0].cpu().numpy()
                        )
                        peak[h * 5 + i] = max(
                            peak[h * 5 + i], float(raw[g["start"] : g["stop"]].max())
                        )
                    sensor = w.sensors[side]
                    if _ == 11:
                        sensor.audit_gate()
                if phase == "tip":
                    active = v["optical_object_index"] >= 0
                    assert (
                        v["optical_object_index"][active]
                        == np.arange(10).reshape(2, 5)[active]
                    ).all()
            assert (peak > 1e-5).all(), (phase, peak)
            report["phases"][phase] = peak.tolist()
            print(phase, peak.tolist(), flush=True)
    else:
        for i, n in enumerate(names):
            w.reset_object((i * 0.03, 0, 0.8), object_name=n)
            w.drive_object((i * 0.03, 0, 0.8), object_name=n)
        target = "sphere" if a.mode == "single" else "cube"
        expected = names.index(target)
        point, normal = w.surface_point("right", "index", "tip")
        w.reset_object(point + normal * 0.015, object_name=target)
        w.drive_object(point + normal * 0.004, object_name=target)
        peak = 0.0
        for _ in range(15):
            w.step()
            v = w.observe()
            peak = max(peak, float(v["depth"][1, 1].max()))
            assert v["pressure"].shape == (2, pressure_count(a.variant))
            active = v["optical_object_index"] >= 0
            assert (v["optical_object_index"][active] == expected).all()
        assert peak > 1e-5
        report["phases"]["right_index_tip_depth_m"] = peak
    for i, n in enumerate(names):
        w.reset_object((i * 0.03, 0, 0.8), object_name=n)
        w.drive_object((i * 0.03, 0, 0.8), object_name=n)
    for _ in range(5):
        w.step()
        v = w.observe()
    assert v["pressure"].max() < 1e-6 and v["depth"].max() < 1e-6
    report["release_zero"] = True
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print("PASS", a.variant, a.mode, flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
