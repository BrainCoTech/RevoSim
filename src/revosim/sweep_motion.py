"""Ten synchronized, taxel-supported surface trajectories; contact-free transfers."""

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("left", "right")
RADIUS = 0.004
PHASES = [
    ("tip", 1.2, 4.0),
    ("pip", 1.4, 3.0),
    ("mcp", 1.4, 4.0),
    ("pip_side_A", 1.4, 2.5),
    ("pip_side_B", 1.4, 2.5),
    ("mcp_side", 1.4, 3.0),
    ("palm", 2.0, 10.0),
]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def at(route, u):
    p = route["points"]
    n = route["normals"]
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    c = np.r_[0, ds.cumsum()]
    if c[-1] < 1e-9:
        tangent = np.cross(n[0], [0, 0, 1])
        if np.linalg.norm(tangent) < 0.1:
            tangent = np.cross(n[0], [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        return p[0] + tangent * 0.0004 * np.sin(u * 2 * np.pi), n[0]
    z = u * c[-1]
    k = int(np.clip(np.searchsorted(c, z, side="right") - 1, 0, len(p) - 2))
    a = (z - c[k]) / (c[k + 1] - c[k])
    nn = n[k] * (1 - a) + n[k + 1] * a
    return p[k] * (1 - a) + p[k + 1] * a, nn / np.linalg.norm(nn)


class Sweep:
    def __init__(self, plan, transform):
        self.names = [f"ball_{s}_{f}" for s in SIDES for f in FINGERS]
        self.routes = {}
        self.segments = []
        self.expected = {}
        for key, _, _ in PHASES:
            arr = []
            for side in SIDES:
                for f in FINGERS:
                    src = plan[side]["phases"][key][f]
                    if src is None:
                        arr.append(None)
                        continue
                    T = transform(side, src["link"])
                    a = dict(src)
                    a["points"] = np.array(src["points"]) @ T[:3, :3].T + T[:3, 3]
                    a["normals"] = np.array(src["normals"]) @ T[:3, :3].T
                    arr.append(a)
            self.routes[key] = arr
        home = np.array([at(r, 0)[0] + [0, 0, 0.045] for r in self.routes["tip"]])
        self.home = home
        last = home.copy()
        start = 0.0
        for key, travel, hold in PHASES:
            routes = self.routes[key]
            entry = np.array(
                [
                    at(r, 0)[0] + at(r, 0)[1] * (0.0055 if "side" in key else 0.012)
                    if r
                    else last[i] + [0, 0, 0.02]
                    for i, r in enumerate(routes)
                ]
            )
            self.segments.append(
                dict(
                    kind="travel",
                    key=key,
                    start=start,
                    duration=travel,
                    source=last.copy(),
                    target=entry,
                )
            )
            start += travel
            self.segments.append(
                dict(kind="contact", key=key, start=start, duration=hold)
            )
            start += hold
            gap = self.gap(key)
            last = np.array(
                [
                    at(r, 0)[0] + at(r, 0)[1] * gap if r else entry[i]
                    for i, r in enumerate(routes)
                ]
            )
        self.segments.append(
            dict(
                kind="release",
                key="release",
                start=start,
                duration=1.8,
                source=last.copy(),
                target=last + [0, 0, 0.045],
            )
        )
        self.duration = start + 1.8

    @staticmethod
    def gap(key):
        return 0.0008 if key == "tip" else -0.0018 if key == "palm" else 0.0002

    def update(self, t):
        if t >= self.duration:
            return None
        seg = next(
            s for s in self.segments if s["start"] <= t < s["start"] + s["duration"]
        )
        q = (t - seg["start"]) / seg["duration"]
        key = seg["key"]
        if seg["kind"] == "travel":
            a = seg["source"]
            b = seg["target"]
            z = np.maximum(a[:, 2], b[:, 2]) + 0.045 + np.tile(np.arange(5), 2) * 0.012
            pa = a.copy()
            pa[:, 2] = z
            pb = b.copy()
            pb[:, 2] = z
            if q < 0.25:
                target = a + (pa - a) * smooth(q / 0.25)
            elif q < 0.75:
                target = pa + (pb - pa) * smooth((q - 0.25) / 0.5)
            else:
                target = pb + (b - pb) * smooth((q - 0.75) / 0.25)
            return target, key, "transfer - lifted over gaps", False, 0.0
        if seg["kind"] == "release":
            return (
                seg["source"] + (seg["target"] - seg["source"]) * smooth(q),
                key,
                "release",
                False,
                0.0,
            )
        tau = t - seg["start"]
        u = 0.5 - 0.5 * np.cos(
            2
            * np.pi
            * (2 if key == "palm" else 1)
            * np.clip((tau - 0.6) / (seg["duration"] - 0.6), 0, 1)
        )
        hover = 0.0055 if "side" in key else 0.012
        gap = hover + (self.gap(key) - hover) * smooth(tau / 0.6)
        targets = []
        for i, r in enumerate(self.routes[key]):
            if r is None:
                targets.append(self.home[i])
                continue
            p, n = at(r, u)
            targets.append(p + n * gap)
        return np.array(targets), key, "press + back-and-forth slide", True, float(u)
