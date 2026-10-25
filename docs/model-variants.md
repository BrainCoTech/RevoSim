# MT and MX model variants

MT is the default. Choose `World(variant="MX")`, set `REVO_VARIANT=MX` for existing examples, or use `examples/sweep_response.py --variant MX`. Explicit arguments override the environment. Unknown names raise an error. When recording an explicitly selected world, pass `Recorder(..., variant=world.variant)` so metadata uses the same layout.

| Variant | Left URDF | Right URDF | Samples / hand |
|---|---|---|---:|
| MT | [revo3_left_MT.urdf](../src/revosim/assets/hands/variants/urdf/revo3_left_MT.urdf) | [revo3_right_MT.urdf](../src/revosim/assets/hands/variants/urdf/revo3_right_MT.urdf) | 519 |
| MX | [revo3_left_MX.urdf](../src/revosim/assets/hands/variants/urdf/revo3_left_MX.urdf) | [revo3_right_MX.urdf](../src/revosim/assets/hands/variants/urdf/revo3_right_MX.urdf) | 387 |

Copy the entire `src/revosim/assets/hands/variants` directory when using these URDFs outside RevoSim: the four files share the adjacent `meshes` directory. Mesh paths are relative. The package includes all referenced files. Point IDs are preserved; side subpoints remain distinct IDs rather than being collapsed into their original parent channel.

## Frames and sampling

Each tactile sphere has its own fixed frame. The URDFs contain unique joint/link names, 21 revolute joints per hand, and no mimic constraints or zero axes on fixed joints. Sphere radius is 1.8 mm. Sphere centres are 1.5 mm below the selected skin surface, including fingertips, pads, sides and palm. Spheres are visual markers, without collision shapes or added inertia.

`release_layout(side, variant)` returns groups in observation order. Each group contains:

- `ids`: original URDF point-link names.
- `centers_local_m`: preserved sphere centres in the owning mechanical link frame.
- `normals_local`: unit outward surface normals in that frame.
- `points_local_m`: skin contact sampling positions, offset 0.2 mm outward from the selected surface.
- `link`, `start`, `stop`: owning frame and slice in `pressure`.

Consequently, `points_local_m - centers_local_m = 0.0017 * normals_local`. Embedded marker centres and surface contact samples serve different purposes. Runtime sampling does not move or rewrite the URDF centres. These counts describe simulation samples, not independent calibrated hardware channels.

The runtime uses the shared PBR mechanical USDs. Their mechanical joint definitions, centre-of-mass and inertia definitions match the release source; release masses are applied on scene creation, including the updated left-hand bodies. The JSON layouts carry the new tactile frames into the corresponding USD link coordinates. URDFs, meshes, layout data and SHA-256 provenance are bundled; see `assets/model_variants.json`. The runtime USD retains its historical `plam` spelling; release URDFs use `palm`. Layout link names refer to the USD frames, and tests verify their transforms against the URDF aliases. Older `source_urdf` and pressure XML files remain as source/calibration references, not the active MT/MX layout.

## Ten-probe demonstration

`examples/sweep_response.py` records 41 seconds by default, at 15 fps with 240 Hz PhysX steps. The four long-finger axes are separated by approximately 22.276 degrees. The thumb retains its independent outward posture.

Ten 4 mm-radius probes test fingertips, middle pads, proximal pads, both populated middle-pad sides, the populated proximal side, and five tracks on each palm. Paths follow existing point IDs. Transfers lift over gaps; isolated side points receive a small local stroke. Five separate palm tracks move back and forth twice. A virtual spring drives each probe, with a 3 N applied-force cap, 180 N/m fingertip stiffness and 360 N/m elsewhere.

The README GIFs are 2× previews of the accompanying full-speed recordings. The normalized SDF/spring response is allowed to fall between sparse points even when PhysX reports contact. There is no left-hand gain correction. Both variants use the same RevoSim optical model; these videos compare spatial coverage and simulated response, not calibrated MT/MX hardware sensitivity. Contact is a rigid/compliant PhysX approximation, not finite-element silicone deformation.

The left optical sampling window is mirrored correctly. Displayed depth/RGB and marker vectors share the right-hand vertical convention; raw marker displacement remains in its physical link frame.

## Validation

```bash
python -m pytest tests -q
python tools/check_variants.py --variant MT --mode sweep --headless --device cuda:0 --output outputs/check-MT
python tools/check_variants.py --variant MX --mode sweep --headless --device cuda:0 --output outputs/check-MX
python tools/check_variants.py --variant MT --mode single --headless --device cuda:0 --output outputs/check-single
python tools/check_variants.py --variant MX --mode multi --headless --device cuda:0 --output outputs/check-multi
```

Static tests verify graph integrity, mesh resolution, unique names, preserved mechanical transforms, sphere centres, depth offsets, original IDs, counts and mirrored marker projection. Isaac checks cover all sweep regions, object ownership, single/multiple general targets and zero response after release. Conservative bounding tests are compared with the full tactile backend during sweep checks.
