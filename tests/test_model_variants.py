"""Release geometry, IDs and model selection contracts, without an Isaac runtime."""

import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from revosim.resources import (
    ASSETS,
    VARIANTS,
    release_layout,
    pressure_count,
    pressure_regions,
    resolve_variant,
    urdf_path,
    sensor_paths,
)


def transform(origin):
    T = np.eye(4)
    if origin is not None:
        T[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return T


def fk(root):
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = {x.get("name") for x in root.findall("link")} - children
    assert len(roots) == 1
    worlds = {roots.pop(): np.eye(4)}
    pending = list(joints)
    while pending:
        old = len(pending)
        for j in pending[:]:
            parent = j.find("parent").get("link")
            if parent in worlds:
                worlds[j.find("child").get("link")] = worlds[parent] @ transform(
                    j.find("origin")
                )
                pending.remove(j)
        assert len(pending) < old, "Cycle or disconnected joint"
    return worlds


@pytest.mark.parametrize("variant,count,palm", [("MT", 519, 71), ("MX", 387, 117)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_release_geometry_and_sampling_contract(variant, count, palm, side):
    path = urdf_path(side, variant)
    root = ET.parse(path).getroot()
    manifest = json.loads((ASSETS / "model_variants.json").read_text())
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == manifest["variants"][variant][side]["sha256"]
    )
    for tag in ["joint", "link"]:
        names = [x.get("name") for x in root.findall(tag)]
        assert len(names) == len(set(names))
    assert len(root.findall("joint[@type='revolute']")) == 21
    for joint in root.findall("joint[@type='fixed']"):
        assert joint.find("mimic") is None
        assert joint.find("axis") is None
    for mesh in root.findall(".//mesh"):
        assert (path.parent / mesh.get("filename")).is_file()
    worlds = fk(root)
    old = ET.parse(ASSETS / f"hands/source_urdf/revo3_{side}.urdf").getroot()
    source_worlds = fk(old)
    for link, T in source_worlds.items():
        release_name = link.replace("_plam_", "_palm_").replace("_tubber_", "_rubber_")
        if release_name in worlds:
            np.testing.assert_allclose(worlds[release_name], T, atol=1e-12)
    ids = []
    start = 0
    for group in release_layout(side, variant):
        points = np.asarray(group["points_local_m"])
        normals = np.asarray(group["normals_local"])
        centres = np.asarray(group["centers_local_m"])
        assert group["start"] == start
        start = group["stop"]
        assert start - group["start"] == len(group["ids"])
        assert np.isfinite(points).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
        # 1.5 mm below skin + 0.2 mm runtime sampling offset.
        np.testing.assert_allclose(points - centres, 0.0017 * normals, atol=1e-9)
        for name, centre in zip(group["ids"], centres):
            link = root.find(f"link[@name='{name}']")
            assert link is not None and link.find("collision") is None
            visual = link.find("visual")
            sphere = visual.find("geometry/sphere")
            assert float(sphere.get("radius")) == 0.0018
            local = (
                np.linalg.inv(source_worlds[group["link"]])
                @ worlds[name]
                @ transform(visual.find("origin"))
            )
            np.testing.assert_allclose(local[:3, 3], centre, atol=1e-10)
        ids.extend(group["ids"])
    assert start == count == pressure_count(variant)
    assert len(ids) == len(set(ids))
    sl = pressure_regions(variant)["palm"]
    assert sl.stop - sl.start == palm
    masses = json.loads((ASSETS / "release_masses.json").read_text())[side]
    for name, mass in masses.items():
        assert (
            float(root.find(f"link[@name='{name}']/inertial/mass").get("value")) == mass
        )


def test_variant_selection_is_explicit_and_validated(monkeypatch):
    monkeypatch.setenv("REVO_VARIANT", "MX")
    assert resolve_variant() == "MX" and resolve_variant("MT") == "MT"
    assert pressure_count("MT") == 519 and pressure_count("MX") == 387
    with pytest.raises(ValueError):
        release_layout("left", "typo")
    with pytest.raises(ValueError):
        release_layout("up", "MT")


def test_left_marker_display_matches_right_vertical_convention():
    data = [
        np.load(sensor_paths(s)["vtac"] / "marker_positions.npz")
        for s in ["left", "right"]
    ]
    rects = [
        json.loads(
            (sensor_paths(s)["vtac"] / "camera_ray_rectangles_320x240.json").read_text()
        )["rectangles"]
        for s in ["left", "right"]
    ]
    for f in ["thumb", "index", "middle", "ring", "pinky"]:
        uv = []
        for d, r in zip(data, rects):
            corners = np.array(r[f]["corners_camera_m"])[:, :2]
            uv.append(
                (d[f + "_points_camera_m"][:, :2] - corners.min(0))
                / np.ptp(corners, axis=0)
            )
        uv[0][:, 1] = 1 - uv[0][:, 1]
        np.testing.assert_allclose(uv[0], uv[1], atol=1e-10)


def test_sweep_routes_use_existing_point_ids():
    routes = json.loads((ASSETS / "sweep_routes.json").read_text())
    for v in VARIANTS:
        for side in ["left", "right"]:
            for phase in routes[v][side]["phases"].values():
                for r in phase.values():
                    if r is None:
                        continue
                    g = next(
                        g for g in release_layout(side, v) if g["region"] == r["region"]
                    )
                    idx = [g["ids"].index(i) for i in r["ids"]]
                    expected = (
                        np.array(g["points_local_m"])[idx]
                        - 0.0002 * np.array(g["normals_local"])[idx]
                    )
                    np.testing.assert_allclose(r["points"], expected, atol=1e-12)
