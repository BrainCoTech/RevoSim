#!/usr/bin/env python3
import argparse, os, json, time, math
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    from revosim.launch import configure

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["MT", "MX"], default=None)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--max-time", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.enable_cameras = True
    app = AppLauncher(configure(a), multi_gpu=False, renderer="RaytracedLighting").app
    try:
        import numpy as np, torch
        from scipy.spatial.transform import Rotation
        from PIL import Image, ImageDraw
        from pxr import UsdGeom, Gf
        from revosim.world import World
        from revosim.dashboard import Dashboard
        from revosim.recording import Recorder
        from revosim.resources import (
            ASSETS,
            SIDES,
            FINGERS,
            resolve_variant,
            release_layout,
        )
        from revosim.sweep_motion import Sweep

        np.random.seed(7)
        torch.manual_seed(7)
        VARIANT = resolve_variant(a.variant)
        names = [f"ball_{s}_{f}" for s in SIDES for f in FINGERS]
        plan = json.loads((ASSETS / "sweep_routes.json").read_text())[VARIANT]
        w = World(
            names,
            device=a.device,
            observation_hz=a.fps,
            hand_spacing=0.32,
            variant=VARIANT,
            probe_radius=0.004,
        )
        w.shape = "10 x 4 mm radius probes / equal finger spacing"
        w.spring_stiffness = {n: 360.0 for n in names}
        w.force_caps = {n: 3.0 for n in names}
        for side, hand in w.hands.items():
            q = hand.data.default_joint_pos.clone()
            for n, value in plan[side]["splay"].items():
                q[0, hand.joint_names.index(n)] = value
            hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            hand.set_joint_position_target(q)
        for _ in range(a.fps):
            w.step()

        def transform(side, link):
            hand = w.hands[side]
            pose = (
                hand.data.body_link_state_w[0, hand.body_names.index(link), :7]
                .cpu()
                .numpy()
            )
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            T[:3, 3] = pose[:3]
            return T

        motion = Sweep(plan, transform)
        for i, n in enumerate(names):
            w.reset_object(motion.home[i], object_name=n)
            w.drive_object(motion.home[i], object_name=n)
        for _ in range(a.fps):
            w.step()
        w.initial_poses = {
            s: h.data.body_link_state_w[0, :, :7].cpu().numpy().copy()
            for s, h in w.hands.items()
        }
        a.output.mkdir(parents=True, exist_ok=True)
        dash = Dashboard(w)
        rec = (
            None
            if a.pilot
            else Recorder(a.output, fps=a.fps, save_data=False, variant=VARIANT)
        )
        settings = dict(
            variant=VARIANT,
            fps=a.fps,
            duration_s=motion.duration,
            probe_radius_m=0.004,
            stiffness_N_m=360.0,
            force_cap_N=3.0,
            source_plan=plan,
            phases=motion.segments,
        )
        for name, obj in [
            ("settings", settings),
            ("physics", w.physics_manifest),
            ("sensor_layout", w.sensor_layout()),
        ]:
            (a.output / (name + ".json")).write_text(
                json.dumps(obj, default=lambda x: x.tolist(), indent=2)
            )
        camera_x = 0.0
        records = []
        start_wall = time.monotonic()
        lastphase = None
        frame = 0
        phase_indices = {}
        groups = {
            s: {g["region"]: g for g in release_layout(s, VARIANT)} for s in SIDES
        }
        duration = min(motion.duration, a.max_time) if a.max_time else motion.duration
        while app.is_running() and frame < math.ceil(duration * a.fps):
            t = frame / a.fps
            state = motion.update(min(t, motion.duration - 1e-8))
            target, key, action, press, u = state
            phase = key + " / " + action
            for i, n in enumerate(names):
                w.spring_stiffness[n] = 180.0 if key == "tip" else 360.0
                w.drive_object(target[i], object_name=n)
            w.step()
            v = w.observe()
            palm_values = [
                float(
                    v["pressure"][
                        i, groups[s]["palm"]["start"] : groups[s]["palm"]["stop"]
                    ].max()
                )
                for i, s in enumerate(SIDES)
            ]
            bank_peak = []
            bank_any = []
            sample_count = []
            target_regions = []
            for si, s in enumerate(SIDES):
                for fi, f in enumerate(FINGERS):
                    bank = w.sensors[s].banks[fi]
                    raw = bank.get_pressure()[0, 0].cpu().numpy()
                    bank_any.append(float(raw.max()))
                    route = plan[s]["phases"].get(key, {}).get(f)
                    region = route["region"] if route else None
                    target_regions.append(region)
                    if region:
                        g = groups[s][region]
                        vals = raw[g["start"] : g["stop"]]
                        bank_peak.append(float(vals.max()))
                        sample_count.append(int((vals > 1e-5).sum()))
                    else:
                        bank_peak.append(0.0)
                        sample_count.append(0)
            forces = np.linalg.norm(v["probe_contact_forces"], axis=-1)
            active = int((forces > 0.02).sum())
            responsive = int((np.array(bank_peak) > 1e-5).sum())
            tip_active = int(
                (
                    (v["depth"].max(axis=(-2, -1)) > 1e-5)
                    & (np.linalg.norm(v["wrench"][..., :3], axis=-1) > 0.01)
                ).sum()
            )
            # Each phase transition checks the conservative optimization against the full sensor code.
            if phase != lastphase:
                for sensor in w.sensors.values():
                    sensor.audit_gate()
            render = not a.pilot or frame % a.fps == 0 or phase != lastphase
            if render:
                desired = (
                    0.35
                    * np.sin(
                        2
                        * np.pi
                        * (
                            t
                            - next(
                                seg["start"]
                                for seg in motion.segments
                                if seg["key"] == key and seg["kind"] == "contact"
                            )
                        )
                        / 3.0
                    )
                    if "side" in key and press
                    else 0.0
                )
                camera_x += 0.35 * (desired - camera_x)
                UsdGeom.Xformable(dash.camera).GetOrderedXformOps()[0].Set(
                    Gf.Matrix4d()
                    .SetLookAt(
                        Gf.Vec3d(camera_x, -0.32, 1.08),
                        Gf.Vec3d(0, -0.005, 0.18),
                        Gf.Vec3d(0, 0, 1),
                    )
                    .GetInverse()
                )
                view = dict(v)
                view["time"] = t
                im = dash.draw(view, phase)
                img = Image.fromarray(im)
                d = ImageDraw.Draw(img)
                d.text(
                    (785, 900),
                    f"PROBES: contact {active}/10 | target taxels {responsive}/10",
                    font=dash.font,
                    fill="white",
                )
                d.text(
                    (785, 926),
                    f"Palm L {palm_values[0]:.3f} / R {palm_values[1]:.3f}  |  largest contact {forces.max():.2f} N",
                    font=dash.small,
                    fill="#aabfd4",
                )
                if rec:
                    rec.append(v, np.array(img), phase)
                if frame % a.fps == 0 or phase != lastphase:
                    (a.output / "snapshots").mkdir(exist_ok=True)
                    img.save(a.output / "snapshots" / f"{frame:04d}.jpg", quality=90)
            if phase not in phase_indices:
                phase_indices[phase] = len(phase_indices)
            records.append(
                dict(
                    time=t,
                    phase=phase_indices[phase],
                    is_press=press,
                    progress=u,
                    pressure=v["pressure"].astype(np.float32),
                    depth_max=v["depth"].max(axis=(-2, -1)),
                    marker=v["marker"],
                    wrench=v["wrench"],
                    contact=v["contact"],
                    object_poses=v["object_poses"],
                    joint_pos=v["joint_pos"],
                    applied_forces=np.stack([w.applied_forces[n] for n in names]),
                    targets=target,
                    probe_contact_forces=v["probe_contact_forces"],
                    palm_contact_forces=v["palm_contact_forces"],
                    palm_response=np.array(palm_values),
                    bank_target_peak=np.array(bank_peak),
                    bank_any_peak=np.array(bank_any),
                    bank_target_count=np.array(sample_count),
                    contact_count=active,
                    response_count=responsive,
                    tip_active=tip_active,
                    optical_object_index=v["optical_object_index"],
                )
            )
            if frame % a.fps == 0:
                print(
                    "FRAME",
                    frame,
                    "t",
                    round(t, 2),
                    key,
                    action,
                    "contact",
                    active,
                    "taxels",
                    responsive,
                    "peaks",
                    np.round(bank_peak, 3).tolist(),
                    "maxF",
                    round(float(forces.max()), 2),
                    "wall",
                    round(time.monotonic() - start_wall, 1),
                    flush=True,
                )
                np.savez_compressed(
                    a.output / "responses.partial.npz",
                    **{k: np.stack([r[k] for r in records]) for k in records[0]},
                )
                (a.output / "phases.json").write_text(
                    json.dumps(phase_indices, indent=2)
                )
            lastphase = phase
            frame += 1
        if rec:
            rec.close()
        dash.close()
        data = {k: np.stack([r[k] for r in records]) for k in records[0]}
        np.savez_compressed(a.output / "responses.npz", **data)
        report = dict(
            variant=VARIANT,
            frames=frame,
            fps=a.fps,
            duration_s=frame / a.fps,
            wall_s=time.monotonic() - start_wall,
            gate_audit_max_error={s: w.sensors[s].audit_max_error for s in SIDES},
            gated_reads={s: w.sensors[s].gated_reads for s in SIDES},
            max_optical_targets_per_pad={
                s: w.sensors[s].max_simultaneous_targets_per_pad for s in SIDES
            },
            phases={},
        )
        for phase, i in phase_indices.items():
            m = data["phase"] == i
            report["phases"][phase] = dict(
                frames=int(m.sum()),
                max_contacts=int(data["contact_count"][m].max()),
                max_responding_probes=int(data["response_count"][m].max()),
                target_peak=data["bank_target_peak"][m].max(0).tolist(),
                contact_force_peak_N=np.linalg.norm(
                    data["probe_contact_forces"][m], axis=-1
                )
                .max(0)
                .tolist(),
            )
        report["release_pressure_peak"] = float(data["pressure"][-1].max())
        report["release_depth_peak"] = float(data["depth_max"][-1].max())
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        print("COMPLETE", VARIANT, frame, flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
