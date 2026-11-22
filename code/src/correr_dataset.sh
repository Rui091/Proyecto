#!/bin/bash
# Corre el pipeline completo CON validación cruzada: M1 a M7.
# Hace lo mismo que correr_pipeline.sh y después corre M7.
#
# Uso (desde cualquier carpeta):
#   ./correr_dataset.sh ampliadas2        # folds de 26 semanas
#   ./correr_dataset.sh originales 13     # folds de 13 semanas (segundo argumento opcional)
set -e
cd "$(dirname "$0")"

dataset="${1:-originales}"
resultados="../../dataset/resultados"

./correr_pipeline.sh "$dataset"

log="$resultados/log_${dataset}_m7_validacion_cruzada.txt"
echo "Corriendo m7_validacion_cruzada.py ..."
# ${2:+"$2"} pasa el tamaño de fold solo si se dio
if ! python m7_validacion_cruzada.py ${2:+"$2"} > "$log" 2>&1; then
  echo "FALLÓ m7_validacion_cruzada.py (log: $log)"
  tail -5 "$log"
  exit 1
fi
echo "  ok"
cp ../../dataset/processed/07_*.csv "$resultados/$dataset/processed/"
echo "Listo: validación cruzada de '$dataset' en $resultados/$dataset"
