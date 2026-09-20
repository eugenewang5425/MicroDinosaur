# Third-party notices and reference boundaries

## Microduck digital blueprint

- Source: `pollen-robotics/microduck_rl`, commit `29e887ecfbf5d37144759e5a9f8a176dfb83d547`.
- The repository code is Apache-2.0. Its README separately states that the 3D model files are Creative Commons BY-SA-NC, without naming a CC version; this project does not invent one.
- The current `MicroDinosaur_v1.blender` and previews retain Microduck-derived 3D geometry. The original non-commercial/share-alike model notice is preserved in [MODEL_LICENSE_NOTE.md](source/microduck_rl/MODEL_LICENSE_NOTE.md); the separate code license is in [LICENSE](source/microduck_rl/LICENSE).
- This project preserves the Microduck hierarchy and source meshes, then adds original DuckRex structural shells, clearance envelopes, controls, reports and renders. Raw downloaded source files are unchanged.

## User-provided mecha T-rex reference

- `机甲霸王龙_P1.3mf` and its embedded assembly guide were used only to understand proportions and assembly language: articulated jaw, armored skull, continuous dorsal line and tapered tail.
- No reference mesh was copied into DuckRex. Instructions in embedded documents were not treated as user requests or executed.
- The original 3MF remains separate and is not redistributed by this project; its own license continues to govern that file.

## Legacy Open Duck Mini variant

- Earlier DuckRex v0.2–v0.4 outputs were based on `apirrone/Open_Duck_Mini`, commit `b23317a485b3cec7d8417f352478778b3475173c`, under Apache-2.0.
- Those old outputs are retained only as rollback/history artifacts and are no longer the active DuckRex basis.

## Blender

- Portable Blender 4.5.9 LTS is unmodified and remains local under `tools/`; it and its installation files are not uploaded in this repository.
- Official Windows archive SHA256 recorded at setup: `41da973b9bf95bb312cbeff4d1982feb13259b43c821686b9bafea4dfe5477cf`.

The local-only `source/manifest.json` records source URLs, revisions, byte counts and SHA256 values for retrieved upstream files. User reference files and retired CAD assemblies remain local.

## Training snapshot added on 2026-09-20

- `simulation/training/` includes the local training implementation based on `pollen-robotics/microduck_rl`; its pinned upstream base and original source hashes are recorded in `simulation/SOURCE_SNAPSHOT.json`.
- The upstream Apache-2.0 code license is preserved in `simulation/training/LICENSE`. Mesh assets in `simulation/training_assets.zip` retain their original model-license boundaries; inclusion here does not relicense those meshes as Apache-2.0.
- The archive retains models required by the current source registry, including compatibility variants. They are not additional current CAD designs. The current CAD and its model notice remain the authoritative mechanical source.
- Selected policies, reports, and videos show simulation work only. Original upstream work and local modifications must not be represented as a wholly original robot or training framework.
