"""
Gráficas: qué pronostica cada modelo frente a lo que realmente se vendió.

No es un módulo del pipeline: se corre después de M5 y M7 y solo lee sus salidas.

Entrada: <carpeta>/05_predicciones.csv        (salida de M5: una fila por plato y semana de prueba)
         <carpeta>/05_metricas.csv            (salida de M5)
         <carpeta>/07_validacion_cruzada.csv  (salida de M7: métricas de cada modelo en cada fold)
Salida:  figs/fig6_01_total_semanal_real_vs_modelos<sufijo>.png   (prueba única de M5)
         figs/fig6_02_platos_real_vs_modelos<sufijo>.png         (prueba única de M5)
         figs/fig6_03_total_por_semestre_cv<sufijo>.png          (validación cruzada de M7, si existe)

Uso (desde code/src):
  python graficas_predicciones.py <carpeta processed> <sufijo> "<nombre del dataset para el título>" [carpeta de salida] [modelos excluidos]
  python graficas_predicciones.py ../../dataset/resultados/ampliadas/processed _ampliadas "Ampliadas 2020-2025"
  python graficas_predicciones.py ../../dataset/resultados/originales/processed _originales_sin_tsb "Originales" ../../figs TSB
Los modelos excluidos van separados por coma (p. ej. "TSB,ARIMA") y no aparecen en ninguna de las tres figuras.
Sin argumentos usa dataset/processed, sin sufijo y sin nombre en el título.
"""

import os
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib  # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.dates as mdates  # type: ignore

carpeta = sys.argv[1] if len(sys.argv) > 1 else '../../dataset/processed'
sufijo = sys.argv[2] if len(sys.argv) > 2 else ''
# Nombre del dataset al inicio de cada título, para no confundir las figuras de distintos datasets
nombre = f'{sys.argv[3]}: ' if len(sys.argv) > 3 else ''
# Títulos: solo el dataset, «predicción vs ventas reales» y si es con o sin validación cruzada
TITULO_SIN_CV = f'{nombre}predicción vs ventas reales (sin validación cruzada)'
TITULO_CON_CV = f'{nombre}predicción vs ventas reales (con validación cruzada)'
if not nombre:
    TITULO_SIN_CV, TITULO_CON_CV = TITULO_SIN_CV.capitalize(), TITULO_CON_CV.capitalize()
# Cuarto argumento opcional: carpeta de salida (por defecto figs/)
carpeta_figuras = sys.argv[4] if len(sys.argv) > 4 else '../../figs'
os.makedirs(carpeta_figuras, exist_ok=True)
# Quinto argumento opcional: modelos que no se grafican, separados por coma
excluidos = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else set()

# Colores (paleta de referencia validada: 3 series categóricas, todas las parejas distinguibles con daltonismo)
SUPERFICIE = '#fcfcfb'
TINTA = '#0b0b0b'
TINTA_SECUNDARIA = '#52514e'
TINTA_TENUE = '#898781'
GRILLA = '#e1e0d9'
EJE = '#c3c2b7'
AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
REAL = TINTA_SECUNDARIA  # lo real va en tinta neutra; los modelos llevan el color

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 9,
    'axes.facecolor': SUPERFICIE, 'figure.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
    'axes.edgecolor': EJE, 'axes.linewidth': 0.8,
    'axes.labelcolor': TINTA_SECUNDARIA, 'xtick.color': TINTA_TENUE, 'ytick.color': TINTA_TENUE,
    'axes.titlecolor': TINTA, 'axes.titlesize': 10, 'axes.titleweight': 'semibold',
    'axes.grid': True, 'grid.color': GRILLA, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'legend.fontsize': 9,
})

MODELOS = [('MM4', 'pred_mm4'), ('GradientBoosting', 'pred_gradient_boosting'), ('LSTM', 'pred_lstm'),
           ('ARIMA', 'pred_arima'), ('SARIMA', 'pred_sarima'), ('TSB', 'pred_tsb')]

predicciones = pd.read_csv(os.path.join(carpeta, '05_predicciones.csv'), parse_dates=['semana_objetivo'])
metricas = pd.read_csv(os.path.join(carpeta, '05_metricas.csv')).set_index('modelo')
MODELOS = [(m, c) for m, c in MODELOS if c in predicciones.columns and m not in excluidos]
inicio, fin = predicciones['semana_objetivo'].min(), predicciones['semana_objetivo'].max()
periodo = f"{inicio:%d-%m-%Y} a {fin:%d-%m-%Y}"


def guardar(figura, nombre):
    ruta = os.path.join(carpeta_figuras, f'{nombre}{sufijo}.png')
    figura.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(figura)
    print(f'-> {ruta}')


def titulo_panel(eje, titulo, detalle):
    # Nombre del panel en negrita y una línea de detalle debajo, sin montarse
    eje.text(0, 1.13, titulo, transform=eje.transAxes, va='bottom', fontsize=10, fontweight='semibold', color=TINTA)
    eje.text(0, 1.03, detalle, transform=eje.transAxes, va='bottom', fontsize=8, color=TINTA_SECUNDARIA)


def eje_fechas(eje):
    eje.xaxis.set_major_locator(mdates.MonthLocator())
    # Meses en español, sin depender del idioma del sistema
    meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    eje.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: meses[mdates.num2date(x).month - 1]))
    eje.tick_params(length=0)


# ---------------------------------------------------------------------------
# Figura 1: total semanal de los platos modelables, real frente a cada modelo (un panel por modelo)
# ---------------------------------------------------------------------------
semanal = predicciones.groupby('semana_objetivo')[['y_real'] + [c for _, c in MODELOS]].sum()
filas = int(np.ceil(len(MODELOS) / 3))
figura, ejes = plt.subplots(filas, 3, figsize=(11, 3.1 * filas), sharex=True, sharey=True)
techo = semanal.max().max() * 1.12
for eje, (modelo, columna) in zip(ejes.ravel(), MODELOS):
    eje.plot(semanal.index, semanal['y_real'], color=REAL, label='Real')
    eje.plot(semanal.index, semanal[columna], color=AZUL, linewidth=2, label='Pronóstico')
    fila = metricas.loc[modelo]
    titulo_panel(eje, modelo, f"MAE {fila['mae']:.3f} · total {fila['unidades_pronosticadas']:.0f} de "
                              f"{fila['unidades_reales']:.0f} unidades · sesgo {fila['sesgo']:+.0f} %")
    eje.set_ylim(0, techo)
    eje_fechas(eje)
for eje in ejes.ravel()[len(MODELOS):]:
    eje.set_visible(False)
for eje in ejes[:, 0]:
    eje.set_ylabel('Unidades por semana')
manejadores, etiquetas = ejes.ravel()[0].get_legend_handles_labels()
figura.suptitle(TITULO_SIN_CV,
                x=0.01, y=0.995, ha='left', fontsize=12, fontweight='semibold', color=TINTA)
figura.legend(manejadores, etiquetas, loc='upper left', ncol=2, bbox_to_anchor=(0.005, 0.965))
figura.tight_layout(h_pad=3.2, rect=(0, 0, 1, 0.935))
guardar(figura, 'fig6_01_total_semanal_real_vs_modelos')

# ---------------------------------------------------------------------------
# Figura 2: los 4 platos más vendidos en la prueba, real frente a LSTM, gradient boosting y TSB
# ---------------------------------------------------------------------------
comparados = [('LSTM', 'pred_lstm', AZUL), ('GradientBoosting', 'pred_gradient_boosting', NARANJA)]
if 'pred_tsb' in predicciones.columns:
    comparados.append(('TSB', 'pred_tsb', AQUA))
comparados = [(m, c, color) for m, c, color in comparados if m not in excluidos]
principales = predicciones.groupby('plato')['y_real'].sum().sort_values(ascending=False).head(4).index
figura, ejes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
for eje, plato in zip(ejes.ravel(), principales):
    datos = predicciones[predicciones['plato'] == plato].sort_values('semana_objetivo')
    # Lo real son unidades enteras: escalones con puntos, para que se vean las semanas sin venta
    eje.step(datos['semana_objetivo'], datos['y_real'], where='mid', color=REAL, linewidth=1.2, label='Real')
    eje.scatter(datos['semana_objetivo'], datos['y_real'], s=14, color=REAL, zorder=3, edgecolors=SUPERFICIE,
                linewidths=1)
    finales = []
    for modelo, columna, color in comparados:
        eje.plot(datos['semana_objetivo'], datos[columna], color=color, linewidth=2, label=modelo)
        finales.append((datos[columna].iloc[-1], modelo, color))
    # Etiquetas directas al final de cada línea, separadas para que no se monten
    techo = max(datos['y_real'].max(), datos[[c for _, c, _ in comparados]].max().max()) * 1.15 + 0.5
    separacion = techo * 0.07
    posiciones = []
    for valor, modelo, color in sorted(finales):
        y = valor if not posiciones else max(valor, posiciones[-1] + separacion)
        posiciones.append(y)
        eje.annotate(modelo, (datos['semana_objetivo'].iloc[-1], valor), xytext=(8, 0),
                     textcoords='offset points', va='center', fontsize=8, color=TINTA_SECUNDARIA,
                     annotation_clip=False)
        eje.texts[-1].xy = (datos['semana_objetivo'].iloc[-1], y)
    total_real = datos['y_real'].sum()
    resumen = ' · '.join(f'{m} {datos[c].sum():.0f}' for m, c, _ in comparados)
    titulo_panel(eje, plato, f'Total real {total_real:.0f} · {resumen}')
    eje.set_ylim(0, techo)
    eje_fechas(eje)
for eje in ejes[:, 0]:
    eje.set_ylabel('Unidades por semana')
manejadores, etiquetas = ejes.ravel()[0].get_legend_handles_labels()
figura.suptitle(TITULO_SIN_CV,
                x=0.01, y=0.995, ha='left', fontsize=12, fontweight='semibold', color=TINTA)
figura.legend(manejadores, etiquetas, loc='upper left', ncol=4, bbox_to_anchor=(0.005, 0.965))
figura.tight_layout(h_pad=3.2, w_pad=4, rect=(0, 0, 1, 0.935))
guardar(figura, 'fig6_02_platos_real_vs_modelos')

# ---------------------------------------------------------------------------
# Figura 3: total pronosticado frente al real en cada semestre de la validación cruzada
# El total pronosticado sale del sesgo: real x (1 + sesgo / 100)
# ---------------------------------------------------------------------------
ruta_cv = os.path.join(carpeta, '07_validacion_cruzada.csv')
if os.path.exists(ruta_cv):
    cv = pd.read_csv(ruta_cv)
    cv = cv[cv['modelo'] != 'Siempre 0'].copy()
    cv['pronosticado'] = cv['unidades_reales'] * (1 + cv['sesgo'] / 100)
    orden = [m for m, _ in MODELOS if m in set(cv['modelo'])]
    folds = sorted(cv['fold'].unique(), reverse=True)  # del más antiguo al más reciente
    # Hasta 3 paneles por fila; con 5 semestres quedan 2 filas
    columnas = min(3, len(folds))
    filas = int(np.ceil(len(folds) / columnas))
    figura, ejes = plt.subplots(filas, columnas, figsize=(3.9 * columnas, 3.4 * filas + 0.4), squeeze=False)
    ejes = ejes.ravel()
    for eje in ejes[len(folds):]:
        eje.set_visible(False)
    for numero, (eje, fold) in enumerate(zip(ejes, folds)):
        datos = cv[cv['fold'] == fold].set_index('modelo').loc[orden]
        real = datos['unidades_reales'].iloc[0]
        y = np.arange(len(orden))[::-1]
        limite = max(datos['pronosticado'].max(), real) * 1.3
        eje.axvline(real, color=REAL, linewidth=1.6, zorder=2)
        eje.hlines(y, real, datos['pronosticado'], color=AZUL, linewidth=2, zorder=3)
        eje.scatter(datos['pronosticado'], y, s=46, color=AZUL, edgecolors=SUPERFICIE, linewidths=2, zorder=4)
        for posicion, (modelo, fila) in zip(y, datos.iterrows()):
            derecha = fila['pronosticado'] >= real
            # Si el punto está pegado al 0 no cabe la etiqueta a la izquierda: va a la derecha, sobre la barra,
            # con fondo del color de la superficie para que se lea
            pegado = not derecha and fila['pronosticado'] < 0.15 * limite
            eje.annotate(f"{fila['sesgo']:+.0f} %", (fila['pronosticado'], posicion),
                         xytext=(7 if derecha or pegado else -7, 0), textcoords='offset points', va='center',
                         ha='left' if derecha or pegado else 'right', fontsize=8, color=TINTA_SECUNDARIA,
                         bbox=dict(facecolor=SUPERFICIE, edgecolor='none', pad=1) if pegado else None, zorder=5)
        eje.annotate(f'Real: {real:.0f}', (real, len(orden) - 0.45), xytext=(5, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=8, color=TINTA_SECUNDARIA)
        # El semestre se nombra por la semana del medio del fold: un fold que empieza el 29-12-2025 es 2026-H1
        medio = pd.Timestamp(datos['semana_inicio'].iloc[0]) + (pd.Timestamp(datos['semana_fin'].iloc[0]) -
                                                                pd.Timestamp(datos['semana_inicio'].iloc[0])) / 2
        eje.set_title(f"{medio.year}-{'H1' if medio.month <= 6 else 'H2'}", loc='left')
        eje.set_yticks(y)
        # Nombres de los modelos solo en la primera columna
        eje.set_yticklabels(orden if numero % columnas == 0 else [])
        eje.set_xlim(0, limite)
        eje.set_ylim(-0.6, len(orden) - 0.1)
        eje.grid(axis='y', visible=False)
        eje.tick_params(length=0)
        eje.set_xlabel('Unidades en el semestre')
    # Título y subtítulo con márgenes fijos en pulgadas, para que no cambien con el número de filas
    alto = figura.get_figheight()
    figura.suptitle(TITULO_CON_CV,
                    x=0.01, y=1 - 0.05 / alto, ha='left', va='top', fontsize=12, fontweight='semibold', color=TINTA)
    figura.text(0.01, 1 - 0.32 / alto, 'La línea gris es lo que se vendió; el punto, lo que pronosticó '
                'el modelo. El porcentaje es el sesgo.', ha='left', va='top', fontsize=9, color=TINTA_SECUNDARIA)
    figura.tight_layout(w_pad=2, h_pad=2.5, rect=(0, 0, 1, 1 - 0.6 / alto))
    guardar(figura, 'fig6_03_total_por_semestre_cv')
