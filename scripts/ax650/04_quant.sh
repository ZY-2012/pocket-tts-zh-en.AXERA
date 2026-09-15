#!/usr/bin/env bash
# M2: Pulsar2 quantization for the zh-en static subgraphs.
# Usage: CHECK_LEVEL=0 bash scripts/ax650/04_quant.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export WORK_TMP="${WORK_TMP:-${ROOT_DIR}/.work_tmp}"
export TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

CHECK_LEVEL="${CHECK_LEVEL:-0}"
MODELS="${MODELS:-flow_net_step mimi_conv_step}"
RUN_NAME="${RUN_NAME:-pulsar2_zh_en_check${CHECK_LEVEL}}"
MODEL_CONVERT_DIR="${MODEL_CONVERT_DIR:-${ROOT_DIR}/model_convert/${RUN_NAME}}"
AXMODEL_DIR="${AXMODEL_DIR:-${ROOT_DIR}/model_convert/axmodels_zh_en_${CHECK_LEVEL}}"

if ! command -v pulsar2 >/dev/null 2>&1; then
    set +u
    source "${NPU_DEV_ENV:-/data/huyuan/npu-codebase/script/npu_dev}"
    set -u
fi

PRECISION_ARGS=()
if [[ "${CHECK_LEVEL}" -ge 2 ]]; then
    PRECISION_ARGS=(--precision-analysis)
fi

python "${ROOT_DIR}/python/make_quant_configs.py" \
    --models ${MODELS} \
    --check-level "${CHECK_LEVEL}" \
    --run-name "${RUN_NAME}" \
    --output-dir "${MODEL_CONVERT_DIR}" \
    "${PRECISION_ARGS[@]}"

mkdir -p "${AXMODEL_DIR}"
export QUANT_MANIFEST="${MODEL_CONVERT_DIR}/quant_manifest_check${CHECK_LEVEL}.json"

python - <<'PY' | while IFS=$'\t' read -r config input_shapes build_dir output_name; do
import json, os
from pathlib import Path
manifest = json.loads(Path(os.environ["QUANT_MANIFEST"]).read_text())
for row in manifest:
    print(row["config"] + "\t" + row["input_shapes"] + "\t" + row["build_dir"] + "\t" + row["output_name"])
PY
    echo "=== pulsar2 build ${output_name} ==="
    pulsar2 build --config "${config}" --input_shapes "${input_shapes}" 2>&1 | tee "${build_dir}.log"
    cp "${build_dir}/${output_name}" "${AXMODEL_DIR}/"
done

cp "${QUANT_MANIFEST}" "${AXMODEL_DIR}/quant_manifest.json"
echo "axmodels in ${AXMODEL_DIR}:"
find "${AXMODEL_DIR}" -maxdepth 1 -type f | sort
