#!/bin/bash
# Corre el pipeline completo SIN validación cruzada: M1 a M6.
#
# Uso (desde cualquier carpeta):
#   ./correr_pipeline.sh               # ventas originales
#   ./correr_pipeline.sh ampliadas     # dataset/raw/ventas_ampliadas.csv
#   ./correr_pipeline.sh ampliadas2    # dataset/raw/ventas_ampliadas2.csv
#
# Los resultados quedan en dataset/processed/ y modelos/, y además se copian en
# dataset/resultados/<dataset>/. La salida de cada módulo queda en dataset/resultados/log_<dataset>_<módulo>.txt.
# Si un módulo falla, el script se detiene y muestra las últimas líneas de su log.
set -e
cd "$(dirname "$0")"

dataset="${1:-originales}"
resultados="../../dataset/resultados"
mkdir -p "$resultados"

correr() {
  # $1: script; el resto: argumentos del script
  local script="$1"; shift
  local log="$resultados/log_${dataset}_${script%.py}.txt"
  echo "Corriendo $script ..."
  if ! python "$script" "$@" > "$log" 2>&1; then
    echo "FALLÓ $script (log: $log)"
    tail -5 "$log"
    exit 1
  fi
  echo "  ok"
}

correr m1_ingesta_limpieza.py "$dataset"
correr m2_serie_semanal.py
correr m3_ventanas_modelos.py
correr m4_entrenamiento.py
correr m5_evaluacion.py
correr m6_pronostico_menu.py

# Copia de los resultados de este dataset
rm -rf "$resultados/$dataset"
mkdir -p "$resultados/$dataset"
cp -r ../../dataset/processed "$resultados/$dataset/processed"
cp -r ../../modelos "$resultados/$dataset/modelos"
# Los 07_*.csv de dataset/processed son de una validación cruzada anterior (otro dataset o corrida),
# no de esta: no se guardan. correr_dataset.sh copia los nuevos después de correr M7.
rm -f "${resultados:?}/${dataset:?}/processed/"07_*.csv
echo "Listo: resultados de '$dataset' en $resultados/$dataset"
