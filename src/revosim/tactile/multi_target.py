"""Independent target backends for disjoint simultaneous fingertip contacts.

Never add RGB images or normalized, already-diffused pressure fields. Optical
pads must have at most one active target; overlapping targets raise an error.
"""

import torch
from . import make_tactile_cfg, TactileSensor


class MultiTargetTactile:
    def __init__(
        self, robot_path, side, target_paths, *, variant=None, target_radii=None
    ):
        self.side = side
        self.banks = [
            TactileSensor(
                make_tactile_cfg(
                    robot_prim_path=robot_path,
                    target_prim_paths=(path,),
                    hands=(side,),
                    object_sample_count=8192,
                    pressure_diffusion=False,
                    variant=variant,
                    target_bounding_radius_m=None
                    if target_radii is None
                    else target_radii[i],
                )
            )
            for i, path in enumerate(target_paths)
        ]
        self._children = self.banks[0]._children
        self.cache = {}
        self.max_simultaneous_targets_per_pad = 0
        self.cooldown = [0] * len(self.banks)
        self.candidates = [True] * len(self.banks)
        self.defaults = {}
        self.gated_reads = 0
        self.audit_max_error = 0.0

    def set_optical_candidates(self, flags):
        self.cooldown = [
            3 if near else max(c - 1, 0) for c, near in zip(self.cooldown, flags)
        ]
        self.candidates = [c > 0 for c in self.cooldown]

    def _bank_read(self, index, name):
        if not self.candidates[index] and name in self.defaults:
            self.gated_reads += 1
            return self.defaults[name]
        value = getattr(self.banks[index], "get_" + name)()
        if name not in self.defaults and not self.candidates[index]:
            if name == "rgb":
                self.defaults[name] = value.clone()
            else:
                if float(value.abs().max()) > 1e-8:
                    raise RuntimeError("Nonzero outside optical mesh bounds")
                self.defaults[name] = torch.zeros_like(value)
        return value

    def audit_gate(self):
        # Compare skipped branches to the original backend at phase boundaries.
        from .backends import observations as obs

        for bank in self.banks:
            fast = bank.get_pressure()[:, 0]
            ctx = bank._contexts[self.side]
            skipped = getattr(ctx, "_pressure_skip_names", set())
            ctx._pressure_skip_names = set()
            full = obs.ours_rl_pressure_obs(ctx, bank._components[self.side])
            ctx._pressure_skip_names = skipped
            err = float((fast - full).abs().max())
            self.audit_max_error = max(self.audit_max_error, err)
            if err > 1e-6:
                raise RuntimeError(f"Pressure broad phase mismatch {err}")
        for i, near in enumerate(self.candidates):
            if near:
                continue
            for name in ["depth", "rgb", "marker_displacement"]:
                if name not in self.defaults:
                    continue
                raw = getattr(self.banks[i], "get_" + name)()
                err = float((raw - self.defaults[name]).abs().max())
                self.audit_max_error = max(self.audit_max_error, err)
                if err > 1e-6:
                    raise RuntimeError(f"Optical gate mismatch {name}: {err}")

    def update(self, dt):
        for bank in self.banks:
            bank.update(dt)
        self.cache.clear()

    def reset(self):
        for bank in self.banks:
            bank.reset()
        self.cache.clear()

    def _owners(self):
        if "owners" not in self.cache:
            depth = torch.stack(
                [self._bank_read(i, "depth") for i in range(len(self.banks))]
            )
            strength = depth.amax(dim=(-2, -1))
            count = (strength > 1e-5).sum(0)
            self.max_simultaneous_targets_per_pad = max(
                self.max_simultaneous_targets_per_pad, int(count.max())
            )
            if int(count.max()) > 1:
                raise RuntimeError(
                    "Two targets touch one optical pad; this demo requires disjoint optical contacts"
                )
            self.cache["owners"] = strength.argmax(0)
            self.cache["depth_banks"] = depth
        return self.cache["owners"]

    def _optical(self, name):
        if name not in self.cache:
            owner = self._owners()
            values = (
                self.cache["depth_banks"]
                if name == "depth"
                else torch.stack(
                    [self._bank_read(i, name) for i in range(len(self.banks))]
                )
            )
            index = owner[None]
            while index.ndim < values.ndim:
                index = index.unsqueeze(-1)
            self.cache[name] = values.gather(0, index.expand(1, *values.shape[1:]))[0]
        return self.cache[name]

    def get_depth(self):
        return self._optical("depth")

    def get_rgb(self):
        return self._optical("rgb")

    def get_marker_displacement(self):
        return self._optical("marker_displacement")

    def get_pressure(self):
        if "pressure" not in self.cache:
            from .backends.observations import (
                _pressure_rl_diffusion_kernel_for_sensor,
                diffuse_pressure_rl_values,
            )

            # Union of per-target raw taxel responses, followed by one diffusion.
            raw = torch.stack([bank.get_pressure() for bank in self.banks]).amax(0)
            first = self.banks[0]
            cfg = first._components[self.side]
            context = first._contexts[self.side]
            chunks, start = [], 0
            for name, size in zip(
                cfg["pressure_sensor_names"], cfg["pressure_taxel_counts"]
            ):
                values = raw[:, 0, start : start + size]
                child = first._children[self.side][name]
                kernel = _pressure_rl_diffusion_kernel_for_sensor(
                    context,
                    name,
                    child,
                    values,
                    sigma_m=cfg["pressure_diffusion_sigma_m"],
                    radius_sigma=cfg["pressure_diffusion_radius_sigma"],
                    normal_power=cfg["pressure_diffusion_normal_power"],
                )
                chunks.append(
                    diffuse_pressure_rl_values(
                        values, kernel, blend=cfg["pressure_diffusion_blend"]
                    )
                )
                start += size
            self.cache["pressure"] = torch.cat(chunks, dim=-1)[:, None]
        return self.cache["pressure"]
