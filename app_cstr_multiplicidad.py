"""
Ingeniería de Reactores II
CSTR no isotérmico: múltiples estados estacionarios, arranque e ignición-extinción

Reacción: A -> B, irreversible, de primer orden, exotérmica, fase líquida.
Ejecutar con:  streamlit run app_cstr_multiplicidad.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R_GAS = 8.314  # J/(mol·K)
T_REF = 300.0  # K, temperatura de referencia para k

# Colores (paleta categórica en orden fijo)
AZUL, NARANJA, AQUA, AMARILLO = "#257de0", "#f47230", "#00b894", "#f2a600"
COLORES_SERIES = [AZUL, NARANJA, AQUA, AMARILLO]

st.set_page_config(page_title="CSTR: multiplicidad de estados", layout="wide")


# ---------------------------------------------------------------------------
# 1. Modelo
# ---------------------------------------------------------------------------
def k_arrhenius(T, k_ref, E):
    """Constante de velocidad k(T) en 1/min."""
    return k_ref * np.exp(E / R_GAS * (1.0 / T_REF - 1.0 / T))


def X_balance_materia(T, p):
    """Conversión que exige el balance de materia: X = τk / (1 + τk)."""
    tk = p["tau"] * k_arrhenius(T, p["k_ref"], p["E"])
    return tk / (1.0 + tk)


def G(T, p):
    """Calor generado por mol de A alimentado, J/mol."""
    return -p["dH"] * X_balance_materia(T, p)


def R_rem(T, p):
    """Calor removido por mol de A alimentado, J/mol."""
    return p["Cp"] * (1.0 + p["kappa"]) * (T - p["Tc"])


def jacobiano(T, X, p):
    """Jacobiano del sistema dinámico (X, T) evaluado en un punto."""
    k = k_arrhenius(T, p["k_ref"], p["E"])
    dk = k * p["E"] / (R_GAS * T**2)
    tau, kap, dTad = p["tau"], p["kappa"], p["dTad"]
    return np.array([
        [-1.0 / tau - k, dk * (1.0 - X)],
        [-dTad * k, -(1.0 + kap) / tau + dTad * dk * (1.0 - X)],
    ])


def clasificar(T, X, p):
    """Clasifica el estado estacionario con los eigenvalores del jacobiano."""
    ev = np.linalg.eigvals(jacobiano(T, X, p))
    if np.all(ev.real < 0):
        return "Estable"
    if np.any(np.abs(ev.imag) > 1e-12):
        return "Inestable (oscilatorio)"
    return "Inestable (punto silla)"


def estados_estacionarios(p):
    """Busca todas las intersecciones G(T) = R(T)."""
    T_min = min(p["Tc"], p["T0"]) - 30.0
    T_max = p["Tc"] + p["dTad"] + 60.0
    T = np.linspace(max(T_min, 150.0), T_max, 4000)
    f = G(T, p) - R_rem(T, p)
    resultados = []
    for i in np.where(np.diff(np.sign(f)) != 0)[0]:
        Ts = brentq(lambda t: G(t, p) - R_rem(t, p), T[i], T[i + 1])
        Xs = float(X_balance_materia(Ts, p))
        resultados.append({"T (K)": Ts, "X": Xs, "Tipo": clasificar(Ts, Xs, p)})
    return resultados


def simular(T_ini, X_ini, p, t_final):
    """Integra los balances dinámicos de materia y energía."""
    def rhs(t, y):
        X, T = y
        k = k_arrhenius(T, p["k_ref"], p["E"])
        dX = -X / p["tau"] + k * (1.0 - X)
        dT = (1.0 + p["kappa"]) * (p["Tc"] - T) / p["tau"] + p["dTad"] * k * (1.0 - X)
        return [dX, dT]

    t_eval = np.linspace(0, t_final, 600)
    sol = solve_ivp(rhs, (0, t_final), [X_ini, T_ini], t_eval=t_eval,
                    method="LSODA", rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


@st.cache_data
def curva_ignicion(params_tuple, T0_min, T0_max, n=160):
    """Estados estacionarios en función de la temperatura de alimentación T0."""
    base = dict(params_tuple)
    filas = []
    for T0 in np.linspace(T0_min, T0_max, n):
        p = parametros(base["tau"], T0, base["Ta"], base["kappa"], base["k_ref"],
                       base["E"], base["dH"], base["Cp"])
        for e in estados_estacionarios(p):
            filas.append({"T0 (K)": T0, **e})
    return pd.DataFrame(filas)


def parametros(tau, T0, Ta, kappa, k_ref, E, dH, Cp):
    """Agrupa parámetros y calcula Tc y ΔT adiabático."""
    return {
        "tau": tau, "T0": T0, "Ta": Ta, "kappa": kappa, "k_ref": k_ref,
        "E": E, "dH": dH, "Cp": Cp,
        "Tc": (T0 + kappa * Ta) / (1.0 + kappa),
        "dTad": -dH / Cp,
    }


def estilo(fig, titulo_x, titulo_y):
    fig.update_layout(
        xaxis_title=titulo_x, yaxis_title=titulo_y,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10), height=430,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Controles
# ---------------------------------------------------------------------------
st.title("CSTR no isotérmico: múltiples estados estacionarios")
st.caption("Reacción A → B, primer orden, irreversible y exotérmica, en fase líquida. "
           "Mueve los parámetros y observa cómo cambian G(T) y R(T).")

with st.sidebar:
    st.header("Parámetros")
    st.subheader("Operación")
    tau = st.slider("Tiempo espacial τ (min)", 1.0, 30.0, 10.0, 0.5)
    T0 = st.slider("Temperatura de alimentación T₀ (K)", 260.0, 380.0, 300.0, 1.0)
    Ta = st.slider("Temperatura del refrigerante Tₐ (K)", 260.0, 380.0, 300.0, 1.0)
    kappa = st.slider("κ = UA / (F_A0 ΣθᵢCpᵢ)", 0.0, 5.0, 1.0, 0.1,
                      help="0 = reactor adiabático. Más grande = más enfriamiento.")
    st.subheader("Cinética y termodinámica")
    k_ref = st.select_slider("k a 300 K (1/min)",
                             options=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
                             value=0.001)
    E = st.slider("Energía de activación E (kJ/mol)", 40.0, 120.0, 80.0, 1.0) * 1000
    dH = st.slider("ΔH_rx (kJ/mol)", -100.0, -10.0, -40.0, 1.0) * 1000
    Cp = st.slider("ΣθᵢCpᵢ (J/mol·K, por mol de A)", 50.0, 400.0, 200.0, 10.0)

p = parametros(tau, T0, Ta, kappa, k_ref, E, dH, Cp)
ee = estados_estacionarios(p)

c1, c2, c3 = st.columns(3)
c1.metric("ΔT adiabático", f"{p['dTad']:.0f} K")
c2.metric("T_c = (T₀ + κTₐ)/(1 + κ)", f"{p['Tc']:.1f} K")
c3.metric("Estados estacionarios", len(ee))

tab1, tab2, tab3 = st.tabs(["G(T) vs R(T)", "Arranque del reactor", "Ignición y extinción"])

# ---------------------------------------------------------------------------
# 3. Pestaña 1: G(T) y R(T)
# ---------------------------------------------------------------------------
with tab1:
    T_max_plot = max(p["Tc"] + p["dTad"] + 40, max([e["T (K)"] for e in ee] + [0]) + 30)
    T = np.linspace(min(p["Tc"], T0) - 20, T_max_plot, 500)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=G(T, p) / 1000, name="G(T): calor generado",
                             line=dict(color=AZUL, width=2)))
    fig.add_trace(go.Scatter(x=T, y=R_rem(T, p) / 1000, name="R(T): calor removido",
                             line=dict(color=NARANJA, width=2)))
    for e in ee:
        estable = e["Tipo"] == "Estable"
        fig.add_trace(go.Scatter(
            x=[e["T (K)"]], y=[G(e["T (K)"], p) / 1000], mode="markers",
            marker=dict(size=12, symbol="circle" if estable else "circle-open",
                        color=AZUL if estable else NARANJA,
                        line=dict(width=2, color="white" if estable else NARANJA)),
            name=e["Tipo"], legendgroup=e["Tipo"], showlegend=False,
            hovertemplate=f"{e['Tipo']}<br>T = %{{x:.1f}} K<br>X = {e['X']:.3f}<extra></extra>",
        ))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(estilo(fig, "Temperatura T (K)", "G, R (kJ/mol de A)"),
                    width="stretch")
    st.caption("Círculo relleno = estado estable. Círculo vacío = estado inestable.")

    if ee:
        tabla = pd.DataFrame(ee)
        tabla["T (K)"] = tabla["T (K)"].round(1)
        tabla["X"] = tabla["X"].round(3)
        st.dataframe(tabla, hide_index=True, width="stretch")

    with st.expander("Ecuaciones del modelo"):
        st.latex(r"G(T) = (-\Delta H_{rx})\,X_{BM}, \qquad X_{BM} = \frac{\tau k}{1+\tau k}")
        st.latex(r"R(T) = \Sigma\theta_i C_{p_i}\,(1+\kappa)\,(T - T_c), \qquad "
                 r"T_c = \frac{T_0 + \kappa T_a}{1+\kappa}")
        st.latex(r"k(T) = k_{300}\,\exp\!\left[\frac{E}{R}\left(\frac{1}{300}-\frac{1}{T}\right)\right]")
        st.markdown("La estabilidad se determina con los eigenvalores del jacobiano "
                    "del sistema dinámico (pestaña *Arranque*). En el punto silla se "
                    "cumple que la pendiente de G es mayor que la de R.")

# ---------------------------------------------------------------------------
# 4. Pestaña 2: arranque (dinámica)
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("Elige hasta 4 temperaturas iniciales del reactor y observa a qué "
                "estado estacionario llega cada una.")
    a, b, c = st.columns([2, 1, 1])
    opciones = list(range(260, 501, 10))
    T_inis = a.multiselect("Temperatura inicial del reactor (K)", opciones,
                           default=[300, 330, 340], max_selections=4)
    X_ini = b.slider("Conversión inicial X₀", 0.0, 1.0, 0.0, 0.05)
    n_tau = c.slider("Duración (múltiplos de τ)", 2, 40, 15)

    with st.expander("Balances dinámicos"):
        st.latex(r"\frac{dX}{dt} = -\frac{X}{\tau} + k(1-X)")
        st.latex(r"\frac{dT}{dt} = \frac{(1+\kappa)(T_c - T)}{\tau} + \Delta T_{ad}\,k(1-X)")

    if T_inis:
        figT, figX = go.Figure(), go.Figure()
        for color, Ti in zip(COLORES_SERIES, T_inis):
            t, X, Tt = simular(float(Ti), X_ini, p, n_tau * tau)
            nombre = f"T inicial = {Ti} K"
            figT.add_trace(go.Scatter(x=t, y=Tt, name=nombre, line=dict(color=color, width=2)))
            figX.add_trace(go.Scatter(x=t, y=X, name=nombre, line=dict(color=color, width=2)))
        for e in ee:
            if e["Tipo"] == "Estable":
                figT.add_hline(y=e["T (K)"], line_dash="dot", line_width=1, line_color="gray")
                figX.add_hline(y=e["X"], line_dash="dot", line_width=1, line_color="gray")
        figT.update_layout(hovermode="x unified")
        figX.update_layout(hovermode="x unified")
        izq, der = st.columns(2)
        izq.plotly_chart(estilo(figT, "Tiempo (min)", "Temperatura del reactor (K)"),
                         width="stretch")
        der.plotly_chart(estilo(figX, "Tiempo (min)", "Conversión X"),
                         width="stretch")
        st.caption("Líneas punteadas = estados estacionarios estables.")

# ---------------------------------------------------------------------------
# 5. Pestaña 3: curva de ignición-extinción
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("Estados estacionarios al variar T₀, manteniendo fijos los demás "
                "parámetros. La zona con tres soluciones produce **histéresis**.")
    rango = st.slider("Rango de T₀ (K)", 200.0, 450.0, (240.0, 400.0), 5.0)
    clave = tuple(sorted({"tau": tau, "Ta": Ta, "kappa": kappa, "k_ref": k_ref,
                          "E": E, "dH": dH, "Cp": Cp}.items()))
    df = curva_ignicion(clave, rango[0], rango[1])

    fig3 = go.Figure()
    for tipo, color, simbolo in [("Estable", AZUL, "circle"),
                                 ("Inestable (punto silla)", NARANJA, "circle-open"),
                                 ("Inestable (oscilatorio)", AQUA, "diamond-open")]:
        d = df[df["Tipo"] == tipo]
        if not d.empty:
            fig3.add_trace(go.Scatter(
                x=d["T0 (K)"], y=d["T (K)"], mode="markers", name=tipo,
                marker=dict(size=8, color=color, symbol=simbolo,
                            line=dict(width=2, color=color)),
                hovertemplate="T₀ = %{x:.1f} K<br>T = %{y:.1f} K<extra></extra>"))
    fig3.add_vline(x=T0, line_dash="dot", line_color="gray",
                   annotation_text="T₀ actual", annotation_position="top")
    st.plotly_chart(estilo(fig3, "Temperatura de alimentación T₀ (K)",
                           "Temperatura en estado estacionario (K)"),
                    width="stretch")

    n_por_T0 = df.groupby("T0 (K)").size()
    multiples = n_por_T0[n_por_T0 >= 3].index
    en_borde = len(multiples) and (multiples.min() <= rango[0] + 1 or multiples.max() >= rango[1] - 1)
    if len(multiples) and en_borde:
        st.warning("La zona de tres estados toca el borde del rango. Amplía el rango de T₀ "
                   "para ver completas la ignición y la extinción.")
    elif len(multiples):
        st.info(f"Hay tres estados estacionarios para T₀ entre "
                f"**{multiples.min():.0f} K** y **{multiples.max():.0f} K**. "
                f"Al calentar la alimentación, el reactor *enciende* cerca de "
                f"{multiples.max():.0f} K; al enfriarla, se *apaga* cerca de "
                f"{multiples.min():.0f} K.")
    else:
        st.info("Con estos parámetros no hay multiplicidad en el rango elegido.")

# ---------------------------------------------------------------------------
# 6. Actividad para la clase
# ---------------------------------------------------------------------------
with st.expander("📝 Actividad para la clase: primero predice, luego comprueba"):
    st.markdown("""
1. Con los valores por defecto hay tres estados estacionarios. **Antes de moverle**,
   predice: si aumentas κ (más enfriamiento), ¿qué pasa con la pendiente de R(T)?
   ¿Siguen existiendo tres soluciones? Compruébalo.
2. En *Arranque*, arranca el reactor a 330 K y luego a 340 K. ¿Por qué dos arranques
   tan parecidos terminan en estados tan distintos? ¿Qué papel juega el punto silla?
3. Pon κ = 0 (adiabático). ¿Cuál es la temperatura máxima posible del reactor?
   Relaciónala con ΔT adiabático.
4. En *Ignición y extinción*, identifica la temperatura de ignición y la de extinción.
   Si operas en el estado alto y la alimentación se enfría poco a poco, ¿en qué T₀
   se "apaga" el reactor?
5. Desde el punto de vista de seguridad, ¿en qué estado conviene operar y por qué?
""")
