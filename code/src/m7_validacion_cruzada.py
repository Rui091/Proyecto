"""
M7. Validación cruzada de series de tiempo (origen móvil, «rolling origin»).

Entrada: dataset/processed/03_ventanas_mm4.csv, 03_ventanas_xgboost.csv, 03_ventanas_lstm.csv (salida de M3)
         dataset/processed/04_entrenamiento.csv  (hiperparámetros ganadores de M4)
Salida:  dataset/processed/07_validacion_cruzada.csv  (métricas de cada modelo en cada fold)
         dataset/processed/07_resumen_cv.csv          (promedio y desviación de los folds)

Idea: en vez de medir los modelos en una sola prueba de 26 semanas, se mide en N_FOLDS bloques seguidos de
26 semanas, de atrás hacia adelante. En cada fold el modelo se entrena SOLO con las semanas anteriores al
bloque y se evalúa en el bloque (nunca se mezcla el futuro con el pasado, por eso no se usa KFold normal).
Los hiperparámetros son los ganadores de M4 y no se vuelven a buscar; solo se reentrena en cada fold.

  fold 1:  entrena [................]  prueba [26 sem]
  fold 2:  entrena [..........]        prueba [26 sem]   ...  (el fold 1 es el último bloque, igual que M5)

Aviso: los hiperparámetros se eligieron con las semanas de validación de M4, que caen dentro de los folds
más recientes; esos folds son algo optimistas. Los folds antiguos son los más honestos.

Uso (desde code/src, después de M4):
  python3 m7_validacion_cruzada.py              # folds de 26 semanas, hasta 5
  python3 m7_validacion_cruzada.py 13           # folds de 13 semanas (para datasets cortos)
"""

import os
import sys
import ast
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score # type: ignore
from sklearn.ensemble import HistGradientBoostingRegressor # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore
import tensorflow as tf # type: ignore

tf.config.experimental.enable_op_determinism()

SEMILLA = 42
SEMILLAS_LSTM = [42, 43, 44, 45, 46]
# Hasta 5 folds; si el dataset es corto se usan menos (los que quepan con el entrenamiento mínimo)
# Semanas de cada fold (26 por defecto) y máximo de folds, opcionales por línea de comandos
SEMANAS_FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 26
MAX_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
# Un fold necesita al menos un año de entrenamiento (igual que M3), para que cada semana del año aparezca
# al menos una vez y la LSTM y el gradient boosting tengan con qué aprender
SEMANAS_MIN_ENTRENAMIENTO = 52
HISTORIA_LSTM = 8
VARIABLES_LSTM = ['ur', 'precio', 'sin', 'cos']
NIVEL_MINIMO = 0.1
UMBRAL_VENTA = 0.5

ruta_mm4 = '../../dataset/processed/03_ventanas_mm4.csv'
ruta_xgboost = '../../dataset/processed/03_ventanas_xgboost.csv'
ruta_lstm = '../../dataset/processed/03_ventanas_lstm.csv'
ruta_entrenamiento = '../../dataset/processed/04_entrenamiento.csv'
ruta_salida = '../../dataset/processed/07_validacion_cruzada.csv'
ruta_resumen = '../../dataset/processed/07_resumen_cv.csv'


def leer_valor(texto):
    # '0.03' -> 0.03, '(1, 0, 0)' -> tupla, 'absolute_error' -> texto
    try:
        return ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return texto


def leer_hiperparametros(texto):
    return {parte.split('=')[0]: leer_valor(parte.split('=')[1]) for parte in texto.split('; ')}


def smape(y_real, y_pred):
    denominador = np.abs(y_real) + np.abs(y_pred)
    terminos = np.where(denominador == 0, 0, 2 * np.abs(y_real - y_pred) / np.where(denominador == 0, 1, denominador))
    return 100 * terminos.mean()


def metricas(y_real, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    vende_real, vende_pred = y_real > 0, y_pred >= UMBRAL_VENTA
    return {'mae': mean_absolute_error(y_real, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_real, y_pred)),
            'smape': smape(y_real, y_pred),
            'recall': recall_score(vende_real, vende_pred, zero_division=0),
            'f1': f1_score(vende_real, vende_pred, zero_division=0),
            'sesgo': 100 * (y_pred.sum() - y_real.sum()) / y_real.sum() if y_real.sum() > 0 else np.nan}


def a_cubo(tabla, columnas_fijas):
    pasos = np.stack([tabla[[f'{v}_t{k}' for k in range(HISTORIA_LSTM, 0, -1)]].to_numpy()
                      for v in VARIABLES_LSTM], axis=2)
    fijas = np.repeat(tabla[columnas_fijas].to_numpy()[:, np.newaxis, :], HISTORIA_LSTM, axis=1)
    return np.concatenate([pasos, fijas], axis=2).astype('float32')


def crear_lstm(parametros, n_variables, semilla):
    tf.keras.utils.set_random_seed(semilla)
    modelo = tf.keras.Sequential([tf.keras.Input(shape=(HISTORIA_LSTM, n_variables))])
    for capa in range(parametros['capas']):
        modelo.add(tf.keras.layers.LSTM(parametros['unidades'], return_sequences=capa != parametros['capas'] - 1))
        modelo.add(tf.keras.layers.Dropout(parametros['dropout']))
    modelo.add(tf.keras.layers.Dense(1))
    modelo.compile(optimizer=tf.keras.optimizers.Adam(parametros['learning_rate']), loss='mae')
    return modelo


def armar_sarimax(serie, orden, estacional):
    tendencia = 'c' if orden[1] == 0 and estacional[1] == 0 else 'n'
    return SARIMAX(serie, order=orden, seasonal_order=estacional, trend=tendencia)


# ---------------------------------------------------------------------------
# 1. Ingesta y hiperparámetros ganadores de M4
# ---------------------------------------------------------------------------

mm4 = pd.read_csv(ruta_mm4, parse_dates=['semana_objetivo'])
xgb = pd.read_csv(ruta_xgboost)
lstm = pd.read_csv(ruta_lstm)
entrenamiento = pd.read_csv(ruta_entrenamiento)

assert mm4['fila_id'].equals(xgb['fila_id']) and mm4['fila_id'].equals(lstm['fila_id']), 'Archivos desalineados'

ganadores = entrenamiento[entrenamiento['mejor_del_modelo'] == 'sí'].set_index('modelo')['hiperparametros']
params_gb = leer_hiperparametros(ganadores['GradientBoosting'])
params_lstm = leer_hiperparametros(ganadores['LSTM_promedio'])
epocas_lstm = params_lstm.pop('epocas')
params_arima = leer_hiperparametros(ganadores['ARIMA'])
params_sarima = leer_hiperparametros(ganadores['SARIMA'])
print('M7 Validación cruzada de series de tiempo')
print(f'Ganadores de M4: GB {params_gb}; LSTM {params_lstm} x {epocas_lstm} épocas; '
      f'ARIMA {params_arima["orden"]}; SARIMA {params_sarima["orden"]}{params_sarima["estacional"]}')

columnas_comunes = ['fila_id', 'plato', 'semana_objetivo', 'conjunto', 'y_real']
# Nivel reciente (igual que M4 y M5): media de las 8 semanas anteriores + 0,1
media_8 = lstm[[f'u_t{k}' for k in range(1, HISTORIA_LSTM + 1)]].mean(axis=1)
nivel = (media_8 + NIVEL_MINIMO).to_numpy()
y_relativa = (mm4['y_real'] / nivel).to_numpy()
xgb['media_8'] = media_8
variables_gb = [c for c in xgb.columns if c not in columnas_comunes]
columnas_plato = [c for c in lstm.columns if c.startswith('plato_')]

lstm_rel = lstm.copy()
for k in range(1, HISTORIA_LSTM + 1):
    lstm_rel[f'ur_t{k}'] = lstm_rel[f'u_t{k}'] / nivel
lstm_rel['nivel'] = nivel
lstm_rel['y_relativa'] = y_relativa

semanas = np.sort(mm4['semana_objetivo'].unique())
N_FOLDS = min(MAX_FOLDS, (len(semanas) - SEMANAS_MIN_ENTRENAMIENTO) // SEMANAS_FOLD)
assert N_FOLDS >= 2, f'Con {len(semanas)} semanas no caben 2 folds de {SEMANAS_FOLD} con {SEMANAS_MIN_ENTRENAMIENTO} de entrenamiento'
print(f'{len(semanas)} semanas objetivo: {N_FOLDS} folds de {SEMANAS_FOLD} semanas')

resultados = []

# ---------------------------------------------------------------------------
# 2. Folds
# Reglas
# V1: fold k prueba el bloque de 26 semanas que termina k-1 bloques antes de la última semana
# V2: entrena con TODAS las filas cuya semana objetivo es anterior al bloque (las ventanas solo miran el pasado)
# V3: la escala de la LSTM, los parámetros de ARIMA/SARIMA y los árboles se calculan solo con el entrenamiento del fold
# ---------------------------------------------------------------------------

for fold in range(1, N_FOLDS + 1):
    fin = len(semanas) - SEMANAS_FOLD * (fold - 1)
    semanas_prueba = semanas[fin - SEMANAS_FOLD:fin]
    es_prueba = mm4['semana_objetivo'].isin(semanas_prueba).to_numpy()
    es_train = (mm4['semana_objetivo'] < semanas_prueba[0]).to_numpy()
    assert not (es_train & es_prueba).any() and es_train.any() and es_prueba.any()
    assert len(np.unique(mm4.loc[es_train, 'semana_objetivo'])) >= SEMANAS_MIN_ENTRENAMIENTO - HISTORIA_LSTM
    y_prueba = mm4.loc[es_prueba, 'y_real'].to_numpy()
    nivel_prueba = nivel[es_prueba]
    print(f'Fold {fold}: prueba {pd.Timestamp(semanas_prueba[0]).date()} a {pd.Timestamp(semanas_prueba[-1]).date()} '
          f'({es_prueba.sum()} filas); entrenamiento {es_train.sum()} filas', flush=True)

    preds = {}
    preds['MM4'] = mm4.loc[es_prueba, ['u_t1', 'u_t2', 'u_t3', 'u_t4']].mean(axis=1).to_numpy()
    preds['Siempre 0'] = np.zeros(es_prueba.sum())

    # Gradient boosting: mismos hiperparámetros y número de árboles que ganaron en M4
    gb = HistGradientBoostingRegressor(**params_gb, early_stopping=False, random_state=SEMILLA)
    gb.fit(xgb.loc[es_train, variables_gb], y_relativa[es_train])
    preds['GradientBoosting'] = gb.predict(xgb.loc[es_prueba, variables_gb]) * nivel_prueba

    # LSTM: escala 0-1 con el entrenamiento del fold, 5 semillas y promedio
    escala = {}
    for variable in VARIABLES_LSTM:
        columnas = [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
        valores = lstm_rel.loc[es_train, columnas].to_numpy()
        escala[variable] = (valores.min(), valores.max())
    escala['nivel'] = (nivel[es_train].min(), nivel[es_train].max())
    escalada = lstm_rel.copy()
    for variable, (minimo, maximo) in escala.items():
        columnas = ['nivel'] if variable == 'nivel' else [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
        escalada[columnas] = (escalada[columnas] - minimo) / (maximo - minimo)
    minimo_ur, maximo_ur = escala['ur']
    y_escalada = ((lstm_rel['y_relativa'] - minimo_ur) / (maximo_ur - minimo_ur)).to_numpy()
    columnas_fijas = columnas_plato + ['nivel']
    X_train = a_cubo(escalada[es_train], columnas_fijas)
    X_prueba = a_cubo(escalada[es_prueba], columnas_fijas)
    pred_semillas = []
    for semilla in SEMILLAS_LSTM:
        modelo = crear_lstm(params_lstm, X_train.shape[2], semilla)
        modelo.fit(X_train, y_escalada[es_train], epochs=epocas_lstm, batch_size=params_lstm['batch_size'], verbose=0)
        relativa = modelo.predict(X_prueba, verbose=0).ravel() * (maximo_ur - minimo_ur) + minimo_ur
        pred_semillas.append(relativa * nivel_prueba)
    preds['LSTM'] = np.mean(pred_semillas, axis=0)

    # ARIMA y SARIMA: parámetros de cada plato con su entrenamiento del fold; pronóstico de una semana
    # adelante en el bloque de prueba con las unidades reales anteriores (igual que M4 y M5)
    for nombre, ajustes in [('ARIMA', params_arima), ('SARIMA', params_sarima)]:
        serie_pred = pd.Series(np.nan, index=mm4.index)
        for plato, filas in mm4[es_train | es_prueba].sort_values('semana_objetivo').groupby('plato'):
            n = int(es_train[filas.index].sum())
            y = filas['y_real'].to_numpy(dtype=float)
            if n < 20:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    modelo = armar_sarimax(y[:n], ajustes['orden'], ajustes['estacional'])
                    parametros = modelo.fit(disp=False, maxiter=50).params
                    pronostico = armar_sarimax(y, ajustes['orden'], ajustes['estacional']).filter(parametros).fittedvalues
                if np.isfinite(parametros).all() and np.isfinite(pronostico).all():
                    serie_pred[filas.index[n:]] = pronostico[n:]
            except Exception:
                pass
        pred = serie_pred[es_prueba].to_numpy().copy()
        # Un plato que no se pudo estimar usa MM4, como en M4
        sin_valor = ~np.isfinite(pred)
        pred[sin_valor] = preds['MM4'][sin_valor]
        preds[nombre] = pred
        print(f'  {nombre}: {sin_valor.sum()} filas con MM4 por falta de parámetros', flush=True)

    for nombre, pred in preds.items():
        resultados.append({'fold': fold, 'modelo': nombre, 'semana_inicio': pd.Timestamp(semanas_prueba[0]).date(),
                           'semana_fin': pd.Timestamp(semanas_prueba[-1]).date(), 'n': int(es_prueba.sum()),
                           'unidades_reales': int(y_prueba.sum()), **metricas(y_prueba, pred)})

# ---------------------------------------------------------------------------
# 3. Resumen y exportación
# ---------------------------------------------------------------------------

folds = pd.DataFrame(resultados)
metricas_resumen = ['mae', 'rmse', 'smape', 'recall', 'f1', 'sesgo']
resumen = folds.groupby('modelo')[metricas_resumen].agg(['mean', 'std'])
resumen.columns = [f'{m}_{e}' for m, e in resumen.columns]
# En cuántos folds el modelo tiene menor MAE que MM4 (la comparación que importa)
mae_mm4 = folds[folds['modelo'] == 'MM4'].set_index('fold')['mae']
folds['gana_a_mm4'] = folds['mae'] < folds['fold'].map(mae_mm4)
resumen['folds_que_supera_mm4'] = folds.groupby('modelo')['gana_a_mm4'].sum()
resumen = resumen.sort_values('mae_mean')

# Controles
assert len(folds) == N_FOLDS * 6, 'Faltan resultados de algún modelo en algún fold'
assert folds[metricas_resumen[:-1]].notna().all().all(), 'Hay métricas vacías'

print(resumen.round(4).to_string())
folds.drop(columns='gana_a_mm4').to_csv(ruta_salida, index=False, float_format='%.4f')
resumen.to_csv(ruta_resumen, float_format='%.4f')
print(f'{len(folds)} filas -> {ruta_salida}; resumen -> {ruta_resumen}')
