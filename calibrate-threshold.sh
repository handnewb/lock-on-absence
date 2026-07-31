#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  calibrate-threshold.sh — barre o threshold de reconhecimento com dados reais
#
#  Existe porque `--sweep threshold=...` NAO funciona: --sweep varia campos de
#  Config (a maquina de estados), e o threshold vive no reconhecedor, aplicado
#  antes de a Observation ser construida. Barrer o threshold exige re-rodar a
#  visao, entao e um laco de shell, nao uma flag.
#
#  Uso:  ./calibrate-threshold.sh mesa.mp4 mesa.csv face_model.yml
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail
VIDEO="${1:?uso: $0 <video> <labels.csv> <model.yml> [valores]}"
LABELS="${2:?falta o CSV de rotulos}"
MODEL="${3:?falta o modelo}"
VALUES="${4:-40,45,50,55,60,65,70,75,80,85}"
FAR_WINDOW="${FAR_WINDOW:-8}"
OUT="${OUT:-calibration}"

command -v lock-on-absence-replay >/dev/null \
  || { echo "instale o pacote primeiro: pip install -e ."; exit 1; }
mkdir -p "$OUT"

printf "video=%s labels=%s model=%s\n\n" "$VIDEO" "$LABELS" "$MODEL"
printf "%10s %8s %8s %10s %9s %8s\n" threshold FAR FRR "TTL med" "TTL p90" spur/h
printf "%10s %8s %8s %10s %9s %8s\n" "---------" "-------" "-------" "---------" "--------" "-------"

IFS=',' read -ra THRESHOLDS <<< "$VALUES"
for t in "${THRESHOLDS[@]}"; do
  json="$OUT/t${t}.json"
  lock-on-absence-replay --video "$VIDEO" --labels "$LABELS" --model "$MODEL" \
      --threshold "$t" --far-window "$FAR_WINDOW" --json "$json" >/dev/null 2>&1 || {
    printf "%10s %s\n" "$t" "(falhou)"; continue; }
  python3 - "$json" "$t" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); t = sys.argv[2]
f = lambda v, s="%": "     n/a" if v is None else f"{v*100:6.1f}{s}"
g = lambda v: "     n/a" if v is None else f"{v:7.1f}s"
print(f"{t:>10} {f(d['far'])} {f(d['frr'])} {g(d['time_to_lock_median'])} "
      f"{g(d['time_to_lock_p90'])} "
      f"{'    n/a' if d['spurious_locks_per_hour'] is None else format(d['spurious_locks_per_hour'],'7.2f')}")
PY
done

cat <<'MSG'

Como escolher:
  * FAR e a falha de SEGURANCA  (intruso aceito). FRR e a falha de USABILIDADE
    (trava na sua cara) -- e a que faz as pessoas desinstalarem.
  * Escolha o menor threshold que mantenha FRR aceitavel, nao o que minimiza a
    soma. As duas colunas nao tem o mesmo custo.
  * Se FAR e FRR forem ambos 0% em toda a faixa, o dataset e facil demais:
    faltam cenarios adversos (contraluz, angulo, segunda pessoa parecida).
  * Publique a tabela no README. Um numero sem a tabela volta a ser chute.

Depois de escolher, grave no metadata (e o agente recalcula nada -- o digest
cobre o .yml, nao o .json):
    python3 - <<'PY'
    import json, pathlib
    p = pathlib.Path("face_model.json"); m = json.loads(p.read_text())
    m["threshold"] = 65            # <-- seu valor escolhido
    p.write_text(json.dumps(m, indent=2))
    PY
MSG
