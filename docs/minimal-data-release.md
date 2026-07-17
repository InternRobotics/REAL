# Minimal Data Release Plan

This document defines the public data release for the seven processed GRScenes
scenes used by REAL. It exercises the demo, procedural generation, and physics
verification without uploading a second copy of MesaTask object USD files.

It is a release specification, not a claim that the proposed Hugging Face
artifacts already exist.

## Design rules

1. **Do not redistribute MesaTask object USDs.** Publish only the required
   basenames and resolve them through `MESATASK_USD_ROOT`.
2. **Release all seven experiment scenes.** These are the processed interaction
   stages used by REAL, not the original GRScenes stages with the same IDs.
3. **Keep data outside the MIT code release.** MesaTask and GRScenes are
   published under CC BY-NC-SA 4.0; a derived scene bundle must preserve their
   attribution and data-license terms.
4. **No machine-specific paths.** Every USD/JSON/YAML reference in a public
   artifact must be relative to the artifact root or be an explicit
   environment-variable reference.
5. **Publish manifests, not opaque archives.** Pin upstream revisions and list
   every file's path, byte size, SHA-256, origin, and license.

## Recommended artifacts

### 1. `real-metadata-v1` — required, small

Publish the 17 files actually read by the generator and placement checker:

| Content | Files | Measured raw size | Purpose |
|---|---:|---:|---|
| Three global JSON files | 3 | 4.38 MiB | asset sizes, furniture/object mapping, purpose pairs |
| Scene furniture libraries | 7 | 0.08 MiB | generation and simulator furniture registry |
| Occupancy maps | 7 | 55.45 MiB | static placement checks and navigation |
| **Total** | **17** | **59.91 MiB** | all seven supported scenes |

The three global files are:

```text
category_pairs_same_purpose.json
consolidated_asset_library_with_size.json
furniture_pair_object_mapping.json
```

`Asset_annotation.json` is not read by current REAL code and is deliberately
excluded. This artifact is sufficient for procedural generation when the user
also has the MesaTask object directory. The occupancy maps can be a separate
optional download, but the complete 59.91 MiB artifact is small enough that
one archive is simpler and less error-prone. A one-scene generation profile for
the default ABY8 scene is only 12.32 MiB (five files).

### 2. `real-processed-grscenes-7-v1` — required scene bundle

Publish the union of the exact dependency closures of all seven processed
`<scene_id>_usd/scene.usd` entries, plus their occupancy maps, furniture
libraries, and three global generator metadata files. Do not include
`assets/objects/`.

An audit of the current staging scene found:

| Metric | Value |
|---|---:|
| USD layers | 776 |
| Existing unique dependency files | 1,327 |
| Existing dependency bytes | 1,313,811,121 (about 1.31 GB) |
| Scene metadata bytes | 8,320,034 (7.93 MiB) |

The existing 9.79 GiB staging bundle is not publishable unchanged: it contains
unused raw-stage dependencies and machine-local paths. The release builder
prunes the InternUtopia seed to the processed seven-scene union.

A clean local build on 2026-07-15 produced the following release-candidate
evidence (this is not a claim that the remote artifact has already been
uploaded):

| Gate | Result |
|---|---:|
| Processed scene entries | 7 |
| Model layers | 4,531 |
| Material/texture/MDL files | 1,647 |
| Payload files | 6,205 |
| Payload bytes | 8,575,426,239 (7.99 GiB) |
| Authored USD asset paths checked | 158,520 |
| MDL modules / transitive resources checked | 152 / 79 |
| Unresolved or external dependencies | 0 / 0 |
| Symlinks / files containing private path markers | 0 / 0 |
| Checksum failures | 0 |

The two default demo objects must be listed in `mesa_required.txt`, not copied:

```text
c13555900d7f413bad3caec2086d3874.usd
052fedd7-cb75-43dd-9685-bd85a0e1619b.usd
```

Their texture references are satisfied by the `textures/` directory shipped
with the MesaTask download.

### 3. `real-benchmark-v1` — task definitions only

Publish the exact 241 REAL-Bench task definitions used by the paper, with:

```text
benchmark/
  README.md
  tasks/
    FDP/<task_id>.yaml
    FODP/<task_id>.yaml
    FDO/<task_id>.yaml
    SUL/<task_id>.yaml
  mesa_required.txt
  manifest.yaml
```

Each task file must be a complete eval-server configuration with `scene_id`,
portable `paths`, the runtime `objects` registry, and exactly one episode. The
release gate loads all 241 files through the eval-server config parser, emits a
basename-only object lock, and asserts both total and per-family counts.

The task artifact contains metadata and object basenames only. It must not
contain MesaTask USDs, replay PKLs, model responses, or rendered trajectories.

### 4. Optional artifacts — release separately

The following are useful but are not part of the minimum runnable release:

- SFT images and annotations;
- evaluation trajectories;
- model checkpoints.

Keeping them separate lets users download scene/runtime data without pulling a
full training corpus or checkpoints.

## Portable scene layout

The builder's output directory is the artifact root. Name that directory
`assets` (or move it to `REAL/assets`) when installing it into a checkout; its
stable layout is:

```text
assets/
  metadata/
    <scene_id>/
      occupancy.npy
      scene_furniture_library.json
  scenes/
    <scene_id>_usd/
      scene.usd
  models/                 # exact scene dependency closure only
  Materials/              # exact scene dependency closure only
```

Runtime USD references should point directly to relative paths in this tree.
Do not depend on `/cpfs/...`, `/shared/...`, a source checkout, or a user-owned
symlink outside the artifact.

## Scene packaging procedure

The exact processed inputs are expected at
`$REAL_PROCESSED_SCENES_ROOT/<scene_id>_usd/scene.usd`. Six are byte-identical
to their processed `start_result_interaction_noMDL_move.usd` source. The ABY8
entry is the experiment artifact that additionally removes `demo_cam_1` and
`demo_cam_2`; do not replace it with the current same-named GRScenes source.

Run the release builder in a Python environment that can import OpenUSD:

```bash
python scripts/data/export_processed_grscenes.py \
  --source-root "$GRSCENES_HOME_ROOT" \
  --processed-scenes-root "$REAL_PROCESSED_SCENES_ROOT" \
  --metadata-root "$REAL_METADATA_ROOT" \
  --toolkit-exporter "$INTERNUTOPIA_ROOT/toolkits/grscenes_scripts/export_scenes.py" \
  --output output/real-processed-grscenes-7-v1
```

The builder deliberately runs InternUtopia's exporter in its supported
raw-stage mode to collect a dependency superset. The unmodified upstream helper
cannot safely consume these processed roots and does not rewrite paths. The
builder then replaces every entry with the hash-pinned experiment stage,
rewrites all USD asset paths to relative files, validates recursive MDL
resources, removes symlinks, prunes unused seed files, verifies all seven
composed closures, compares prim path/type fingerprints, and writes
`manifest.json` plus `SHA256SUMS`.

Both inputs are pinned separately: `internutopia_seed_sha256` records the
`start_result_raw.usd` layer actually consumed by the upstream exporter, while
`processed_sha256` records the experiment entry that replaces it. This prevents
the raw dependency seed from being mistaken for the released scene.

The pre-existing staging tree is **not ready to publish unchanged**. Its audit
found unresolved default-material textures and references that resolve outside
the artifact. Only the validated builder output is a release candidate.

## Required validation gates

Run all gates from a clean machine or container where the internal source tree
is not mounted.

### File and path integrity

- every manifest SHA-256 and byte size matches;
- no file content or USD asset path contains `/cpfs/`, `/shared/`, credentials,
  or a private hostname;
- every resolved scene dependency remains inside the extracted artifact;
- `UsdUtils.ComputeAllDependencies` returns no unresolved dependency;
- all YAML `episodes[*].placements` keys exist in the corresponding `objects`;
- every `mesa_required.txt` basename is present under `MESATASK_USD_ROOT`.

### Functional smoke tests

1. Generate all five task types for the demo scene with
   `--verify-placement`; require `articulation.yaml` and no `store.yaml` or
   `retrieve.yaml`.
2. Physics-check at least one episode per released task family with the default
   settle-step count.
3. Start the MCP demo with no OpenAI credentials, list all 11 tools, and execute
   exact `find_objects`, `pick`, `place`, `open`, `close`, and interactive
   `ask` calls.
4. Run the merge stage and assert zero dangling placement references.
5. Validate benchmark total/per-family counts against the paper split.

## Data that must not be uploaded

Exclude all of the following from every public artifact:

```text
.env
embedding_cache/
eval_output/
replay/*.pkl
simuser_logs/
internal service responses
private cluster paths or hostnames
checkpoints and optimizer state
```

## License and provenance

- Keep repository code under the existing MIT license.
- Put data in a separate Hugging Face dataset repository with a dedicated
  dataset card and data license.
- Preserve [GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes)
  and [MesaTask-10K](https://huggingface.co/datasets/InternRobotics/MesaTask-10K)
  attribution and CC BY-NC-SA 4.0 terms. Preserve any third-party notices
  carried by scene materials as well.
- Record the exact upstream repository revision for every derived file.
- State clearly that MesaTask is gated and downloaded separately; REAL does
  not re-host its object USD payload.

This split keeps the seven processed scenes and their runtime metadata together,
while MesaTask objects, task definitions, training data, and checkpoints remain
separate artifacts.
