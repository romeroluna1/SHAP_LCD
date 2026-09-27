import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine
from matplotlib.ticker import FuncFormatter


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

OUTPUT_DIR = "graficas_exploracion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# ARCHIVOS CON LOS NOMBRES DE LAS CLASES
# ============================================================

IRIS_CLASS_FILE = "CLASS_NAMES_IRIS.json"
VEHICLE_CLASS_FILE = "CLASS_NAMES_VEHICLE.json"


# ============================================================
# FUNCIÓN PARA CARGAR NOMBRES DE CLASES
# ============================================================

def cargar_nombres_clases(ruta):

    with open(ruta, "r", encoding="utf-8") as archivo:
        nombres = json.load(archivo)

    # Convertir las claves del JSON de string a entero
    # Ejemplo:
    # {"0": "setosa", "1": "versicolor"}
    #
    # se convierte en:
    # {0: "setosa", 1: "versicolor"}

    nombres = {
        int(clave): valor
        for clave, valor in nombres.items()
    }

    return nombres


# ============================================================
# CARGAR LOS NOMBRES DE LAS CLASES
# ============================================================

iris_class_names = cargar_nombres_clases(
    IRIS_CLASS_FILE
)

vehicle_class_names = cargar_nombres_clases(
    VEHICLE_CLASS_FILE
)


print("\nNombres de clases Iris:")
print(iris_class_names)

print("\nNombres de clases Vehicle:")
print(vehicle_class_names)


# ============================================================
# FUNCIÓN PARA GUARDAR GRÁFICAS
# ============================================================

def guardar_grafica(nombre):

    ruta = os.path.join(
        OUTPUT_DIR,
        nombre
    )

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Gráfica guardada: {ruta}"
    )


# ============================================================
# FUNCIÓN PARA ANALIZAR DATOS FALTANTES
# ============================================================

def analizar_faltantes(
    df,
    nombre_dataset
):

    print("\n" + "=" * 70)
    print(
        f"ANÁLISIS DE DATOS FALTANTES - "
        f"{nombre_dataset.upper()}"
    )
    print("=" * 70)

    nulos = df.isnull().sum()

    total_nulos = nulos.sum()

    print(
        f"Total de valores NaN: {total_nulos}"
    )

    columnas_con_nulos = (
        nulos[nulos > 0]
        .sort_values(
            ascending=False
        )
    )

    if len(columnas_con_nulos) == 0:

        print(
            "No se encontraron valores nulos."
        )

        return

    print(
        "\nColumnas con valores nulos:"
    )

    print(
        columnas_con_nulos
    )

    plt.figure(
        figsize=(12, 6)
    )

    columnas = (
        columnas_con_nulos.index
    )

    valores = (
        columnas_con_nulos.values
    )

    bars = plt.barh(
        columnas,
        valores
    )

    plt.title(
        f"Valores faltantes - "
        f"{nombre_dataset}"
    )

    plt.xlabel(
        "Cantidad de valores faltantes"
    )

    plt.ylabel(
        "Variable"
    )

    for bar, value in zip(
        bars,
        valores
    ):

        plt.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:,}".replace(
                ",", "."
            ),
            va="center"
        )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_datos_faltantes.png"
    )


# ============================================================
# INFORMACIÓN GENERAL
# ============================================================

def informacion_general(
    df,
    nombre_dataset,
    target=None,
    class_names=None
):

    print("\n" + "=" * 70)

    print(
        f"INFORMACIÓN GENERAL - "
        f"{nombre_dataset.upper()}"
    )

    print("=" * 70)

    print(
        f"Registros: {df.shape[0]}"
    )

    print(
        f"Atributos totales: "
        f"{df.shape[1]}"
    )

    if target is not None:

        print(
            f"Variable objetivo: "
            f"{target}"
        )

        print(
            f"Atributos predictivos: "
            f"{df.shape[1] - 1}"
        )

        cantidad_clases = (
            df[target].nunique()
        )

        print(
            f"Cantidad de clases: "
            f"{cantidad_clases}"
        )

        print(
            "\nDistribución de clases:"
        )

        conteo = (
            df[target]
            .value_counts()
            .sort_index()
        )

        for clase, cantidad in (
            conteo.items()
        ):

            nombre = (
                class_names.get(
                    int(clase),
                    str(clase)
                )
                if class_names
                else str(clase)
            )

            print(
                f"  {clase} - {nombre}: "
                f"{cantidad:,}".replace(
                    ",", "."
                )
            )


# ============================================================
# GRÁFICA DE CLASES
# ============================================================

def grafica_clases(
    df,
    target,
    nombre_dataset,
    class_names=None
):

    if target not in df.columns:

        print(
            f"No se encontró la variable "
            f"'{target}' en "
            f"{nombre_dataset}."
        )

        return

    conteo = (
        df[target]
        .value_counts()
        .sort_index()
    )

    # Convertir códigos a nombres
    etiquetas = []

    for clase in conteo.index:

        try:

            clase_num = int(clase)

            nombre = (
                class_names.get(
                    clase_num,
                    str(clase)
                )
                if class_names
                else str(clase)
            )

        except:

            nombre = str(clase)

        etiquetas.append(
            nombre
        )

    plt.figure(
        figsize=(9, 6)
    )

    bars = sns.barplot(
        x=etiquetas,
        y=conteo.values
    )

    plt.title(
        f"Distribución de clases - "
        f"{nombre_dataset}"
    )

    plt.xlabel(
        "Clase"
    )

    plt.ylabel(
        "Cantidad de registros"
    )

    for bar, valor in zip(
        bars.patches,
        conteo.values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{valor:,}".replace(
                ",", "."
            ),
            ha="center",
            va="bottom"
        )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_distribucion_clases.png"
    )


# ============================================================
# DISTRIBUCIÓN DE ATRIBUTOS
# ============================================================

def grafica_atributos_numericos(
    df,
    nombre_dataset,
    columnas=None,
    cantidad=4
):

    columnas_numericas = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if columnas is None:

        columnas = (
            columnas_numericas[:cantidad]
        )

    else:

        columnas = [
            col
            for col in columnas
            if col in df.columns
        ]

    if len(columnas) == 0:

        print(
            f"No se encontraron "
            f"atributos numéricos "
            f"para {nombre_dataset}."
        )

        return

    print(
        f"\nAtributos seleccionados "
        f"para {nombre_dataset}:"
    )

    for col in columnas:

        print(
            f" - {col}"
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    axes = axes.flatten()

    for i, columna in enumerate(
        columnas
    ):

        sns.histplot(
            df[columna].dropna(),
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Distribución de {columna}"
        )

        axes[i].set_xlabel(
            columna
        )

        axes[i].set_ylabel(
            "Frecuencia"
        )

    for i in range(
        len(columnas),
        len(axes)
    ):

        axes[i].axis("off")

    plt.suptitle(
        f"Distribución de atributos - "
        f"{nombre_dataset}",
        fontsize=15
    )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_distribucion_atributos.png"
    )


# ============================================================
# BOXPLOTS
# ============================================================

def grafica_boxplots(
    df,
    nombre_dataset,
    columnas=None,
    cantidad=4
):

    columnas_numericas = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if columnas is None:

        columnas = (
            columnas_numericas[:cantidad]
        )

    else:

        columnas = [
            col
            for col in columnas
            if col in df.columns
        ]

    if len(columnas) == 0:

        return

    fig, axes = plt.subplots(
        1,
        len(columnas),
        figsize=(16, 6)
    )

    if len(columnas) == 1:

        axes = [axes]

    for ax, columna in zip(
        axes,
        columnas
    ):

        sns.boxplot(
            y=df[columna],
            ax=ax
        )

        ax.set_title(
            columna
        )

        ax.set_ylabel(
            "Valor"
        )

    plt.suptitle(
        f"Distribución y posibles "
        f"valores atípicos - "
        f"{nombre_dataset}",
        fontsize=15
    )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_boxplots.png"
    )


# ============================================================
# 1. IRIS
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET IRIS")
print("#" * 80)

iris = load_iris(
    as_frame=True
)

df_iris = iris.frame.copy()

target_iris = "target"

informacion_general(
    df_iris,
    "Iris",
    target_iris,
    iris_class_names
)

analizar_faltantes(
    df_iris,
    "Iris"
)

grafica_clases(
    df_iris,
    target_iris,
    "Iris",
    iris_class_names
)

grafica_atributos_numericos(
    df_iris,
    "Iris",
    columnas=iris.feature_names
)

grafica_boxplots(
    df_iris,
    "Iris",
    columnas=iris.feature_names
)


# ============================================================
# 2. WINE
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET WINE")
print("#" * 80)

wine = load_wine(
    as_frame=True
)

df_wine = wine.frame.copy()

target_wine = "target"

# Scikit-learn proporciona directamente
# los nombres de las clases de Wine.

wine_class_names = {
    i: nombre
    for i, nombre in enumerate(
        wine.target_names
    )
}

print(
    "\nNombres de clases Wine:"
)

print(
    wine_class_names
)

informacion_general(
    df_wine,
    "Wine",
    target_wine,
    wine_class_names
)

analizar_faltantes(
    df_wine,
    "Wine"
)

grafica_clases(
    df_wine,
    target_wine,
    "Wine",
    wine_class_names
)

columnas_wine = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "proline"
]

grafica_atributos_numericos(
    df_wine,
    "Wine",
    columnas=columnas_wine
)

grafica_boxplots(
    df_wine,
    "Wine",
    columnas=columnas_wine
)


# ============================================================
# 3. VEHICLE
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET VEHICLE")
print("#" * 80)

vehicle_path = "vehicle.csv"

df_vehicle = pd.read_csv(
    vehicle_path,
    sep=None,
    engine="python"
)

print(
    "\nPrimeras filas:"
)

print(
    df_vehicle.head()
)

print(
    "\nColumnas:"
)

print(
    df_vehicle.columns.tolist()
)


# ------------------------------------------------------------
# Identificar variable objetivo
# ------------------------------------------------------------

posibles_targets = [
    "class",
    "Class",
    "target",
    "Target",
    "CLASS"
]

target_vehicle = None

for columna in posibles_targets:

    if columna in df_vehicle.columns:

        target_vehicle = columna

        break


if target_vehicle is None:

    target_vehicle = (
        df_vehicle.columns[-1]
    )

    print(
        "\nNo se identificó "
        "automáticamente la clase."
    )

    print(
        f"Se utilizará la última columna: "
        f"{target_vehicle}"
    )


informacion_general(
    df_vehicle,
    "Vehicle",
    target_vehicle,
    vehicle_class_names
)

analizar_faltantes(
    df_vehicle,
    "Vehicle"
)

grafica_clases(
    df_vehicle,
    target_vehicle,
    "Vehicle",
    vehicle_class_names
)


columnas_vehicle = (
    df_vehicle
    .drop(
        columns=[
            target_vehicle
        ]
    )
    .select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)

columnas_vehicle = (
    columnas_vehicle[:4]
)

grafica_atributos_numericos(
    df_vehicle,
    "Vehicle",
    columnas=columnas_vehicle
)

grafica_boxplots(
    df_vehicle,
    "Vehicle",
    columnas=columnas_vehicle
)



# ============================================================
# 4. TRABAJO INFANTIL
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET TRABAJO INFANTIL")
print("#" * 80)

trabajo_path = "df_Menores_dumy.csv"

# Variable objetivo del dataset (actividad principal del menor)
TARGET_TRABAJO = "ocupa_mptiempo"

# Archivo opcional con los nombres de las clases de la variable
# objetivo, con el mismo formato que los de Iris y Vehicle.
# Si no existe, se usan las etiquetas que traiga el dataset.
TRABAJO_CLASS_FILE = "CLASS_NAMES_WORK_INFANT.json"

df_trabajo = pd.read_csv(
    trabajo_path,
    sep=None,           # detecta ; o , automáticamente
    engine="python"
)


def corregir_codificacion(texto):
    """
    Repara nombres guardados con doble codificación
    (ej. 'niÃ±os' -> 'niños'). Si no aplica, devuelve el original.
    """
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


df_trabajo.columns = [
    corregir_codificacion(c).strip() for c in df_trabajo.columns
]

print("\nPrimeras filas:")
print(df_trabajo.head())

print(f"\nRegistros: {df_trabajo.shape[0]}")
print(f"Columnas:  {df_trabajo.shape[1]}")


# ------------------------------------------------------------
# Utilidades para este dataset
# ------------------------------------------------------------

def buscar_columna(df, candidatos):
    """
    Devuelve la primera columna del DataFrame que coincida
    (sin distinguir mayúsculas) con alguno de los candidatos.
    Si ninguna coincide exactamente, busca por contenido.
    """

    columnas = {c.lower().strip(): c for c in df.columns}

    for cand in candidatos:
        if cand.lower() in columnas:
            return columnas[cand.lower()]

    for cand in candidatos:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c

    return None


def columnas_dummy_de(df, variable):
    """
    Devuelve las columnas dummy que pertenecen a una variable,
    asumiendo el formato 'variable = valor'.
    """

    resultado = []

    for c in df.columns:

        if "=" not in c:
            continue

        base = c.split("=", 1)[0].strip().lower()

        if base == variable.lower():
            resultado.append(c)

    return resultado


def reconstruir_categorica(df, variable):
    """
    Reconstruye una variable categórica a partir de sus columnas
    dummy ('variable = valor'). Devuelve una Serie con el valor
    de la categoría en cada fila, o None si no hay dummies.
    Las filas sin ninguna dummy activa quedan como NaN.
    """

    cols = columnas_dummy_de(df, variable)

    if len(cols) == 0:
        return None

    etiquetas = {
        c: c.split("=", 1)[1].strip()
        for c in cols
    }

    # Convertir True/False, "1"/"0", etc. a numérico
    sub = (
        df[cols]
        .replace({"True": 1, "False": 0, True: 1, False: 0})
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    activa = sub.sum(axis=1) > 0

    serie = sub.idxmax(axis=1).map(etiquetas)

    serie[~activa] = pd.NA

    return serie.rename(variable)


def obtener_variable(df, variable, candidatos_extra=None):
    """
    Obtiene una variable ya sea como columna directa o
    reconstruida desde dummies. Devuelve (Serie, origen).
    """

    col = buscar_columna(
        df,
        [variable] + (candidatos_extra or [])
    )

    # Si la columna directa es en realidad una dummy, ignorarla
    if col is not None and "=" not in col:
        return df[col], f"columna '{col}'"

    serie = reconstruir_categorica(df, variable)

    if serie is not None:
        return serie, "reconstruida desde columnas dummy"

    return None, None


def formato_miles(valor):
    return f"{int(valor):,}".replace(",", ".")


# ------------------------------------------------------------
# Datos faltantes
# ------------------------------------------------------------

analizar_faltantes(
    df_trabajo,
    "Trabajo Infantil"
)


# ------------------------------------------------------------
# Variable objetivo: ocupa_mptiempo
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print(f"VARIABLE OBJETIVO - {TARGET_TRABAJO}")
print("=" * 70)

y_trabajo, origen_target = obtener_variable(
    df_trabajo,
    TARGET_TRABAJO
)

trabajo_class_names = None

if os.path.exists(TRABAJO_CLASS_FILE):
    trabajo_class_names = cargar_nombres_clases(TRABAJO_CLASS_FILE)
    print(f"Nombres de clases cargados desde {TRABAJO_CLASS_FILE}")

if y_trabajo is None:

    print(
        f"ADVERTENCIA: no se encontró '{TARGET_TRABAJO}' ni como "
        f"columna ni como dummies. Revise el nombre de la variable."
    )

    cantidad_clases_trabajo = "N/A"

else:

    print(f"Origen: {origen_target}")

    y_trabajo = y_trabajo.dropna()

    # Aplicar nombres de clases si la variable es numérica
    if trabajo_class_names is not None:
        try:
            y_trabajo = y_trabajo.astype(int).map(
                lambda k: trabajo_class_names.get(k, str(k))
            )
        except (ValueError, TypeError):
            pass

    conteo_target = y_trabajo.value_counts()

    cantidad_clases_trabajo = len(conteo_target)

    print(f"Cantidad de clases: {cantidad_clases_trabajo}")
    print("\nDistribución de clases (n y %):")

    for clase, n in conteo_target.items():
        pct = n / len(y_trabajo) * 100
        print(f"  {clase}: {formato_miles(n)} ({pct:.2f}%)")

    ratio = conteo_target.max() / conteo_target.min()

    print(
        f"\nRazón de desbalance (mayoritaria / minoritaria): "
        f"{ratio:.1f} : 1"
    )

    # Gráfica: barras horizontales con n y porcentaje
    plt.figure(figsize=(11, 6))

    orden = conteo_target.sort_values(ascending=True)

    bars = plt.barh(
        [str(c) for c in orden.index],
        orden.values
    )

    plt.title(
        f"Distribución de la variable objetivo "
        f"({TARGET_TRABAJO}) - Trabajo Infantil"
    )

    plt.xlabel("Cantidad de registros")
    plt.ylabel("Clase")

    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, p: formato_miles(x))
    )

    for bar, n in zip(bars, orden.values):
        pct = n / len(y_trabajo) * 100
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {formato_miles(n)} ({pct:.1f}%)",
            va="center"
        )

    plt.xlim(0, orden.max() * 1.25)

    guardar_grafica(
        "trabajo_infantil_distribucion_clases.png"
    )


# ------------------------------------------------------------
# Tipos de variables (numéricas, dummy/binarias)
# ------------------------------------------------------------

print("\n\nTipos de variables:")

cols_target = columnas_dummy_de(df_trabajo, TARGET_TRABAJO)

if TARGET_TRABAJO in df_trabajo.columns:
    cols_target.append(TARGET_TRABAJO)

cols_dummy = [
    c for c in df_trabajo.columns
    if "=" in c and c not in cols_target
]

# Columnas enteras con muy pocos valores distintos (1/2, 1..4) son
# categóricas codificadas como número, no cantidades.
MAX_VALORES_CATEGORICA = 4

cols_restantes = [
    c for c in df_trabajo.select_dtypes(include="number").columns
    if c not in cols_dummy and c not in cols_target
]

cols_cat_codificadas = [
    c for c in cols_restantes
    if df_trabajo[c].nunique() <= MAX_VALORES_CATEGORICA
]

cols_numericas = [
    c for c in cols_restantes if c not in cols_cat_codificadas
]

variables_origen_dummy = sorted({
    c.split("=", 1)[0].strip() for c in cols_dummy
})

print(f"  Columnas de la variable objetivo:      {len(cols_target)}")
print(f"  Columnas dummy (0/1):                  {len(cols_dummy)}")
print(
    f"    provenientes de {len(variables_origen_dummy)} "
    f"variables categóricas: {variables_origen_dummy}"
)
print(
    f"  Categóricas codificadas como entero:   "
    f"{len(cols_cat_codificadas)}"
)
for c in cols_cat_codificadas:
    print(f"    {c}: valores {sorted(df_trabajo[c].unique().tolist())}")
print(f"  Numéricas (cantidades):                {len(cols_numericas)}")
print(f"    {cols_numericas}")

atributos_predictivos_trabajo = (
    len(cols_dummy) + len(cols_cat_codificadas) + len(cols_numericas)
)

print(
    f"\n  Total atributos predictivos: {atributos_predictivos_trabajo} "
    f"(+ {len(cols_target)} de la variable objetivo = "
    f"{df_trabajo.shape[1]} columnas)"
)


# ------------------------------------------------------------
# Verificación de calidad adicional
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print("VERIFICACIÓN DE CALIDAD - TRABAJO INFANTIL")
print("=" * 70)

# Filas duplicadas
n_dup = int(df_trabajo.duplicated().sum())
print(
    f"Filas exactamente duplicadas: {formato_miles(n_dup)} "
    f"({n_dup / len(df_trabajo) * 100:.2f}%) -> "
    f"{formato_miles(len(df_trabajo) - n_dup)} filas únicas"
)

# Columnas casi constantes (un solo valor en >= 99% de las filas)
print("\nColumnas casi constantes (>= 99% un mismo valor):")
for c in df_trabajo.columns:
    top_pct = df_trabajo[c].value_counts(normalize=True).iloc[0] * 100
    if top_pct >= 99:
        print(
            f"  {c}: valor {df_trabajo[c].mode().iloc[0]} "
            f"en {top_pct:.2f}% de las filas"
        )

# Pares de variables numéricas casi idénticas
print("\nVariables numéricas con correlación >= 0.99:")
if len(cols_numericas) > 1:
    corr = df_trabajo[cols_numericas].corr().abs()
    for i, a in enumerate(cols_numericas):
        for b in cols_numericas[i + 1:]:
            if corr.loc[a, b] >= 0.99:
                iguales = (df_trabajo[a] == df_trabajo[b]).mean() * 100
                print(
                    f"  {a} ~ {b}: r = {corr.loc[a, b]:.4f} "
                    f"(valores idénticos en {iguales:.1f}% de las filas)"
                )

# Posible fuga de información: variables categóricas que
# determinan por completo alguna clase de la variable objetivo
if y_trabajo is not None:

    print(
        "\nClases de la variable objetivo concentradas al 100% en una "
        "categoría poco frecuente (posible fuga de información):"
    )

    for var in variables_origen_dummy:

        cat = reconstruir_categorica(df_trabajo, var)

        tabla = pd.crosstab(y_trabajo, cat.loc[y_trabajo.index])

        for clase in tabla.index:

            fila = tabla.loc[clase]

            if fila.sum() == 0 or not (fila == fila.sum()).any():
                continue

            categoria = fila.idxmax()

            total_cat = tabla[categoria].sum()

            # Solo interesa si la categoría es rara en el dataset:
            # una clase 100% dentro de una categoría mayoritaria
            # no aporta información.
            if total_cat / len(y_trabajo) < 0.05:
                print(
                    f"  clase {clase} ({formato_miles(fila.sum())} reg.) "
                    f"está 100% en '{var} = {categoria}', categoría con "
                    f"{formato_miles(total_cat)} registros en total"
                )


# ------------------------------------------------------------
# Valores cero en variables numéricas
# ------------------------------------------------------------

print("\n\nAnálisis de valores cero (variables numéricas):")

for col in cols_numericas:

    cantidad = int((df_trabajo[col] == 0).sum())

    if cantidad > 0:

        porcentaje = cantidad / len(df_trabajo) * 100

        print(
            f"  {col}: {formato_miles(cantidad)} "
            f"({porcentaje:.2f}%)"
        )


# ------------------------------------------------------------
# Edad
# ------------------------------------------------------------

col_edad = buscar_columna(
    df_trabajo,
    ["edad", "age", "p6040"]
)

if col_edad is None:

    print("\nNo se encontró la variable de edad.")

else:

    print(f"\n\nDistribución de edades ({col_edad}):")
    print(df_trabajo[col_edad].describe())

    conteo_edad = df_trabajo[col_edad].value_counts().sort_index()

    print("\nRegistros por edad:")
    for e, n in conteo_edad.items():
        print(f"  {e:>2} años: {formato_miles(n)}")

    fuera = df_trabajo[
        (df_trabajo[col_edad] < 5) | (df_trabajo[col_edad] > 17)
    ]

    if len(fuera) > 0:
        print(
            f"\nNota: {formato_miles(len(fuera))} registros "
            f"({len(fuera) / len(df_trabajo) * 100:.1f}%) tienen edad "
            f"fuera del rango 5-17 años. Revisar si deben incluirse "
            f"en el análisis de trabajo infantil."
        )

    plt.figure(figsize=(10, 6))

    edades = df_trabajo[col_edad].dropna()

    # Una barra por cada edad entera
    sns.histplot(
        edades,
        discrete=True,
        shrink=0.8
    )

    plt.title("Distribución de edades - Trabajo Infantil")
    plt.xlabel("Edad (años)")
    plt.ylabel("Cantidad de registros")

    plt.xticks(range(int(edades.min()), int(edades.max()) + 1))

    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: formato_miles(x))
    )

    guardar_grafica(
        "trabajo_infantil_distribucion_edades.png"
    )


# ------------------------------------------------------------
# Género (sexo)
# ------------------------------------------------------------

sexo_trabajo, origen_sexo = obtener_variable(
    df_trabajo,
    "sexo",
    candidatos_extra=["genero", "género", "p6020"]
)

if sexo_trabajo is None:

    print("\nNo se encontró la variable de sexo/género.")

else:

    print(f"\n\nDistribución de género ({origen_sexo}):")

    genero_counts = sexo_trabajo.dropna().value_counts()

    for g, n in genero_counts.items():
        pct = n / genero_counts.sum() * 100
        print(f"  {g}: {formato_miles(n)} ({pct:.2f}%)")

    plt.figure(figsize=(8, 6))

    bars = sns.barplot(
        x=[str(g).capitalize() for g in genero_counts.index],
        y=genero_counts.values
    )

    plt.title("Distribución de género - Trabajo Infantil")
    plt.xlabel("Género")
    plt.ylabel("Cantidad de registros")

    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: formato_miles(x))
    )

    for bar, n in zip(bars.patches, genero_counts.values):
        pct = n / genero_counts.sum() * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{formato_miles(n)} ({pct:.1f}%)",
            ha="center",
            va="bottom"
        )

    guardar_grafica(
        "trabajo_infantil_distribucion_genero.png"
    )


# ------------------------------------------------------------
# Variable objetivo cruzada con sexo y edad
# ------------------------------------------------------------

if y_trabajo is not None and sexo_trabajo is not None:

    cruce = pd.crosstab(
        y_trabajo.astype(str),
        sexo_trabajo.loc[y_trabajo.index],
        normalize="index"
    ) * 100

    print("\n\nDistribución de género dentro de cada clase (%):")
    print(cruce.round(1))

    cruce.plot(
        kind="barh",
        stacked=True,
        figsize=(11, 6)
    )

    plt.title(
        f"Composición por género de cada clase "
        f"({TARGET_TRABAJO}) - Trabajo Infantil"
    )

    plt.xlabel("Porcentaje de registros de la clase (%)")
    plt.ylabel("Clase")
    plt.xlim(0, 100)
    plt.legend(title="Género", loc="lower right")

    guardar_grafica(
        "trabajo_infantil_clase_por_genero.png"
    )


if y_trabajo is not None and col_edad is not None:

    df_ce = pd.DataFrame({
        "clase": y_trabajo.astype(str),   # como texto: eje categórico
        "edad": df_trabajo.loc[y_trabajo.index, col_edad]
    }).dropna()

    plt.figure(figsize=(11, 6))

    orden_clases = (
        df_ce["clase"].value_counts().index.tolist()
    )

    sns.boxplot(
        data=df_ce,
        x="edad",
        y="clase",
        order=orden_clases
    )

    plt.title(
        f"Edad según clase ({TARGET_TRABAJO}) - Trabajo Infantil"
    )

    plt.xlabel("Edad (años)")
    plt.ylabel("Clase")

    guardar_grafica(
        "trabajo_infantil_edad_por_clase.png"
    )


# ------------------------------------------------------------
# Formato COP
# ------------------------------------------------------------

def formato_cop(x, pos):
    return f"${x:,.0f}".replace(",", ".")


fmt_cop = FuncFormatter(formato_cop)


# ------------------------------------------------------------
# Variables monetarias: ingresos y gastos
# ------------------------------------------------------------

def grafica_monetaria(df, columna, etiqueta, nombre_archivo):
    """
    Histograma (escala logarítmica) y boxplot de una variable
    monetaria. Los valores cero se reportan aparte porque no
    pueden representarse en escala logarítmica y suelen ser
    registros sin información.
    """

    serie = pd.to_numeric(df[columna], errors="coerce").dropna()

    n_ceros = int((serie == 0).sum())
    n_neg = int((serie < 0).sum())
    positivos = serie[serie > 0]

    print(f"\n\nDistribución de {etiqueta.lower()} ({columna}):")
    print(serie.describe().apply(lambda v: f"{v:,.0f}"))
    print(
        f"  Valores cero: {formato_miles(n_ceros)} "
        f"({n_ceros / len(serie) * 100:.2f}%)"
    )

    if n_neg > 0:
        print(f"  Valores negativos: {formato_miles(n_neg)}")

    if len(positivos) == 0:
        print("  No hay valores positivos para graficar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(
        positivos,
        log_scale=True,
        bins=50,
        ax=axes[0]
    )

    axes[0].set_title(
        f"Distribución de {etiqueta.lower()} (escala log)"
    )
    axes[0].set_xlabel(f"{etiqueta} mensual (COP)")
    axes[0].set_ylabel("Cantidad de registros")
    axes[0].xaxis.set_major_formatter(fmt_cop)
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(
        y=positivos,
        ax=axes[1]
    )

    axes[1].set_yscale("log")
    axes[1].set_title(
        f"Boxplot de {etiqueta.lower()} (escala log)"
    )
    axes[1].set_ylabel(f"{etiqueta} mensual (COP)")
    axes[1].yaxis.set_major_formatter(fmt_cop)

    plt.suptitle(
        f"{etiqueta} - Trabajo Infantil "
        f"(se excluyen {formato_miles(n_ceros)} registros con valor 0)",
        fontsize=13
    )

    guardar_grafica(nombre_archivo)


col_ingreso = buscar_columna(
    df_trabajo,
    ["i_hogar", "ingreso_hogar", "ingr_hogar", "ingtotug", "ingreso"]
)

col_gasto = buscar_columna(
    df_trabajo,
    ["i_ugasto", "ingr_mens_unit_gast", "gasto_hogar", "gasto"]
)

if col_ingreso is None:
    print("\nNo se encontró la variable de ingresos del hogar.")
else:
    grafica_monetaria(
        df_trabajo,
        col_ingreso,
        "Ingresos del hogar",
        "trabajo_infantil_distribucion_ingresos.png"
    )

if col_gasto is None:
    print("\nNo se encontró la variable de gastos del hogar.")
else:
    grafica_monetaria(
        df_trabajo,
        col_gasto,
        "Gastos del hogar",
        "trabajo_infantil_distribucion_gastos.png"
    )


# ============================================================
# RANGOS DE INGRESOS Y GASTOS
# ============================================================

rangos = [
    0,
    1,                  # separa los registros con valor 0
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    float("inf")
]

etiquetas_rangos = [
    "$0",
    "$1-500K",
    "$500K-1M",
    "$1M-2M",
    "$2M-5M",
    "$5M-10M",
    "$10M-50M",
    "$50M+"
]


def conteo_por_rangos(df, columna):

    serie = pd.to_numeric(df[columna], errors="coerce")

    return (
        pd.cut(
            serie,
            bins=rangos,
            labels=etiquetas_rangos,
            right=False
        )
        .value_counts()
        .reindex(etiquetas_rangos, fill_value=0)
    )


def grafica_rangos(ax, conteo, titulo, xlabel):

    sns.barplot(
        x=conteo.index.astype(str),
        y=conteo.values,
        ax=ax
    )

    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cantidad de registros")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: formato_miles(x))
    )

    for i, n in enumerate(conteo.values):
        ax.text(
            i,
            n,
            formato_miles(n),
            ha="center",
            va="bottom",
            fontsize=9
        )


paneles = []

if col_ingreso is not None:
    conteo_ing = conteo_por_rangos(df_trabajo, col_ingreso)
    print("\nDistribución por rangos de ingresos:")
    print(conteo_ing)
    paneles.append(
        (conteo_ing, "Ingresos del hogar por rangos",
         "Rango de ingresos (COP)")
    )

if col_gasto is not None:
    conteo_gas = conteo_por_rangos(df_trabajo, col_gasto)
    print("\nDistribución por rangos de gastos:")
    print(conteo_gas)
    paneles.append(
        (conteo_gas, "Gastos del hogar por rangos",
         "Rango de gastos (COP)")
    )

if paneles:

    fig, axes = plt.subplots(
        1,
        len(paneles),
        figsize=(8 * len(paneles), 7)
    )

    if len(paneles) == 1:
        axes = [axes]

    for ax, (conteo, titulo, xlabel) in zip(axes, paneles):
        grafica_rangos(ax, conteo, titulo, xlabel)

    guardar_grafica(
        "trabajo_infantil_distribucion_por_rangos.png"
    )


# ============================================================
# CONSOLIDADO
# ============================================================

print("\n\n")
print("=" * 80)
print("CONSOLIDADO DE DATASETS")
print("=" * 80)

resumen = pd.DataFrame({

    "Dataset": [
        "Iris",
        "Wine",
        "Vehicle",
        "Trabajo Infantil"
    ],

    "Registros": [
        df_iris.shape[0],
        df_wine.shape[0],
        df_vehicle.shape[0],
        df_trabajo.shape[0]
    ],

    "Atributos": [
        df_iris.shape[1] - 1,
        df_wine.shape[1] - 1,
        df_vehicle.shape[1] - 1,
        atributos_predictivos_trabajo
    ],

    "Clases": [
        df_iris[
            target_iris
        ].nunique(),

        df_wine[
            target_wine
        ].nunique(),

        df_vehicle[
            target_vehicle
        ].nunique(),

        cantidad_clases_trabajo
    ]
})

print(
    resumen.to_string(
        index=False
    )
)

resumen.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consolidado_datasets.csv"
    ),
    index=False
)

print(
    "\nAnálisis finalizado."
)

print(
    f"Las gráficas se encuentran "
    f"en: {OUTPUT_DIR}"
)
