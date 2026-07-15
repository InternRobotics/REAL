#!/usr/bin/env bash
# 物理验证 proc_datagen/configs 中生成的四类任务（串行）
# 输入为 per-scene per-type YAML 文件，由 task_generator.py 生成。
# 任务已含 computed_placements，直接跳过静态过滤阶段。
#
# 用法:
#   ./scripts/filter/batch_filter_proc.sh               # 验证所有场景，然后合并
#   ./scripts/filter/batch_filter_proc.sh --stage physics  # 只跑验证
#   ./scripts/filter/batch_filter_proc.sh --stage merge    # 只合并已完成的结果

set -euo pipefail

# ================================================================
# 1. 参数解析
# ================================================================
STAGE="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! "$STAGE" =~ ^(all|physics|merge)$ ]]; then
    echo "ERROR: --stage must be one of: all, physics, merge"
    exit 1
fi

# ================================================================
# 2. 配置区域
# ================================================================
SCENES=(
    'MVUCSQAKTKJ5EAABAAAAABQ8'
    'MVUCSQAKTKJ5EAABAAAAAAQ8'
    'MVUCSQAKTKJ5EAABAAAAABA8'
    'MVUCSQAKTKJ5EAABAAAAACA8'
    'MVUCSQAKTKJ5EAABAAAAAAI8'
    'MV7J6NIKTKJZ2AABAAAAAEI8'
    'MVUCSQAKTKJ5EAABAAAAABY8'
)

TASK_TYPES=(
    "articulation"
    "interactive"
    "distractor"
    "gather"
)

# Isaac Sim's conda activation exports a synthetic BASH_SOURCE value.  This
# launcher is executed (not sourced), so $0 is the reliable script location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_SRC_DIR="${TASK_SRC_DIR:-$ROOT/proc_datagen/configs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/proc_datagen/verify_results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/proc_datagen/verify_proc.py"

# ================================================================
# 3. 初始化输出目录
# ================================================================
for TASK_TYPE in "${TASK_TYPES[@]}"; do
    for SCENE_ID in "${SCENES[@]}"; do
        mkdir -p "$OUTPUT_ROOT/$TASK_TYPE/$SCENE_ID"
    done
done

# ================================================================
# Phase 1: 物理验证（各场景串行）
# ================================================================
run_physics() {
    local TOTAL=${#SCENES[@]}
    echo ""
    echo "=========================================================="
    echo " Physics Verification  [${TOTAL} scenes × ${#TASK_TYPES[@]} task types]"
    echo "=========================================================="

    for i in "${!SCENES[@]}"; do
        SCENE_ID=${SCENES[$i]}
        echo ""
        echo "----------------------------------------"
        echo "[$(( i + 1 ))/$TOTAL] Scene: $SCENE_ID"
        echo "----------------------------------------"

        for TASK_TYPE in "${TASK_TYPES[@]}"; do
            YAML_FILE="$TASK_SRC_DIR/$SCENE_ID/$TASK_TYPE.yaml"
            OUT_PATH="$OUTPUT_ROOT/$TASK_TYPE/$SCENE_ID"

            echo "  [$TASK_TYPE]"

            if [ ! -f "$YAML_FILE" ]; then
                echo "  WARNING: $YAML_FILE not found — skipping"
                continue
            fi

            TASK_SOURCE_PATH=$YAML_FILE \
            OUTPUT_PATH=$OUT_PATH \
            "$PYTHON_BIN" "$SCRIPT" 2>&1 | tee "$OUT_PATH/physics_log.txt"
        done

        echo "-> Done: $SCENE_ID ($(( i + 1 ))/$TOTAL)"
    done

    echo ""
    echo "All scenes done."
}

# ================================================================
# Phase 2: 合并结果
# ================================================================
run_merge() {
    echo ""
    echo "=========================================================="
    echo " Merging Physics Verification Results"
    echo "=========================================================="

    for TASK_TYPE in "${TASK_TYPES[@]}"; do
        TASK_OUTPUT_ROOT="$OUTPUT_ROOT/$TASK_TYPE"

        echo ""
        echo "  [$TASK_TYPE]"
        for SCENE_ID in "${SCENES[@]}"; do
            OUT_PATH="$TASK_OUTPUT_ROOT/$SCENE_ID"
            if [ -f "$OUT_PATH/physics_passed.yaml" ]; then
                PASSED=$(python -c "
import yaml
with open('$OUT_PATH/physics_passed.yaml') as f:
    doc = yaml.safe_load(f)
print(len(doc.get('episodes', [])))
" 2>/dev/null || echo "?")
                FAILED=$(python -c "
import yaml
with open('$OUT_PATH/physics_failed.yaml') as f:
    doc = yaml.safe_load(f)
print(len(doc.get('failed_episodes', [])))
" 2>/dev/null || echo "?")
                echo "    $SCENE_ID: passed=$PASSED  failed=$FAILED"
            else
                echo "    $SCENE_ID: NOT DONE"
            fi
        done
    done

    "$PYTHON_BIN" << 'MERGE_SCRIPT'
import yaml
import json
import os

task_src_dir = os.environ.get("TASK_SRC_DIR", "TASK_SRC_DIR_PLACEHOLDER")
output_root  = os.environ.get("OUTPUT_ROOT", "OUTPUT_ROOT_PLACEHOLDER")

task_types = ["articulation", "interactive", "distractor", "gather"]

scenes = [
    "MVUCSQAKTKJ5EAABAAAAABQ8",
    "MVUCSQAKTKJ5EAABAAAAAAQ8",
    "MVUCSQAKTKJ5EAABAAAAABA8",
    "MVUCSQAKTKJ5EAABAAAAACA8",
    "MVUCSQAKTKJ5EAABAAAAAAI8",
    "MV7J6NIKTKJZ2AABAAAAAEI8",
    "MVUCSQAKTKJ5EAABAAAAABY8",
]


class _FlowList(list):
    pass

def _represent_flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

class _CustomDumper(yaml.Dumper):
    pass
_CustomDumper.add_representer(_FlowList, _represent_flow_list)


total_stats = {}

for task_type in task_types:
    task_output_root = os.path.join(output_root, task_type)
    export_path = os.path.join(task_output_root, "physics_valid.yaml")

    print(f"\n{'='*55}\nProcessing: {task_type}\n{'='*55}")

    # Merge all passed episodes across scenes
    merged_objects = {}
    merged_episodes = []
    scene_stats = {}
    paths_info = {}

    for scene_id in scenes:
        passed_file = os.path.join(task_output_root, scene_id, "physics_passed.yaml")
        if os.path.exists(passed_file):
            with open(passed_file) as f:
                doc = yaml.safe_load(f)
            episodes = doc.get("episodes", [])
            objects = doc.get("objects", {})

            if not paths_info:
                paths_info = doc.get("paths", {})

            # Add scene_id to each episode for traceability
            for ep in episodes:
                ep["scene_id"] = scene_id
                ep["placements"] = {
                    f"{scene_id}_{obj_key}": placement
                    for obj_key, placement in ep.get("placements", {}).items()
                }
                merged_episodes.append(ep)

            # Merge objects (prefix with scene_id to avoid key collisions)
            for obj_key, obj_data in objects.items():
                merged_key = f"{scene_id}_{obj_key}"
                merged_objects[merged_key] = obj_data

            scene_stats[scene_id] = len(episodes)
        else:
            scene_stats[scene_id] = 0

    # Count total source episodes from input YAML files
    n_total = 0
    for scene_id in scenes:
        src_file = os.path.join(task_src_dir, scene_id, f"{task_type}.yaml")
        if os.path.exists(src_file):
            with open(src_file) as f:
                src_doc = yaml.safe_load(f)
            n_total += len(src_doc.get("episodes", []))

    n_valid = len(merged_episodes)
    rate = f"{n_valid / n_total * 100:.1f}%" if n_total else "N/A"
    total_stats[task_type] = {"physics_valid": n_valid, "total": n_total, "rate": rate}

    # Save merged YAML
    merged_doc = {
        "task_type": task_type,
        "objects": merged_objects,
        "episodes": merged_episodes,
    }

    with open(export_path, "w") as f:
        yaml.dump(merged_doc, f, Dumper=_CustomDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False, width=120)

    print(f"  Total             : {n_total}")
    print(f"  Physics valid     : {n_valid}  ({rate})")
    print(f"  Saved: {export_path}")
    for sid, cnt in scene_stats.items():
        print(f"    {sid}: {cnt}")

print(f"\n{'='*55}\nFINAL SUMMARY\n{'='*55}")
for tb, s in total_stats.items():
    print(f"  {tb:30s}: {s['physics_valid']}/{s['total']}  ({s['rate']})")

summary_file = os.path.join(output_root, "physics_final_summary.yaml")
with open(summary_file, "w") as f:
    yaml.dump(total_stats, f, default_flow_style=False)
print(f"\nSummary: {summary_file}")
MERGE_SCRIPT

    echo ""
    echo "Done! Results in: $OUTPUT_ROOT"
}

# ================================================================
# 4. 执行
# ================================================================
# Export paths for the embedded Python merge script
export TASK_SRC_DIR
export OUTPUT_ROOT

case "$STAGE" in
    physics) run_physics ;;
    merge)   run_merge ;;
    all)     run_physics; run_merge ;;
esac
