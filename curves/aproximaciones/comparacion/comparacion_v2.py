"""
comparacion_v2.py

Compara tres metodos de aproximacion sobre las 500 curvas target.

Diferencia clave respecto a comparacion_3_metodos.py:
  - Spline mejorado: s_factor=0.001 (s=1) en lugar de s_factor=0.1 (s=100).
    El valor anterior sobre-suavizaba porque fue elegido sin analizar el dataset.
    Con s=1 el spline adapta el numero de nudos a la complejidad de cada curva
    (media ~20 nudos vs ~11 anteriores) logrando R² medio ~0.99999 vs ~0.9994.
    No es interpolacion exacta (s>0): el spline sigue siendo un modelo suavizante.

Metodos:
  1. Polinomio      — grado optimo por BIC (grados 1-20), sin cambios.
  2. Gaussianas     — N optimo por AIC (N=1-8), sin cambios.
  3. Spline         — UnivariateSpline, k=5, s_factor=0.001 (mejorado).

Metricas por curva: r2, rmse, mae, max_abs_error, n_params + param propio.
Salida: comparacion_v2.csv en este mismo directorio.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = SCRIPT_DIR.parent.parent / 'datos' / 'target'
OUT_CSV    = SCRIPT_DIR / 'comparacion_v2.csv'

# ── Utilidades ─────────────────────────────────────────────────────────────────
def load_curve(cid: int):
    data = np.loadtxt(TARGET_DIR / f'curve_{cid:04d}.txt', delimiter=',')
    return data[:, 0], data[:, 1]

def compute_metrics(y_true, y_pred):
    r2  = float(r2_score(y_true, y_pred))
    rms = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ma  = float(mean_absolute_error(y_true, y_pred))
    mxe = float(np.max(np.abs(y_true - y_pred)))
    return r2, rms, ma, mxe

# ── Metodo 1: Polinomio con BIC (sin cambios) ──────────────────────────────────
def fit_poly_bic(xs: np.ndarray, ys: np.ndarray, max_deg: int = 20) -> dict:
    n = len(xs)
    best_bic, best_deg, best_pred = np.inf, 1, None
    for deg in range(1, max_deg + 1):
        try:
            coeffs = np.polyfit(xs, ys, deg)
        except np.linalg.LinAlgError:
            continue
        y_pred = np.polyval(coeffs, xs)
        ssr    = max(float(np.sum((ys - y_pred) ** 2)), 1e-12)
        bic    = n * np.log(ssr / n) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_deg, best_pred = bic, deg, y_pred
    r2, rms, ma, mxe = compute_metrics(ys, best_pred)
    return dict(r2=r2, rmse=rms, mae=ma, max_ae=mxe,
                n_params=best_deg + 1, degree=best_deg)

# ── Metodo 2: Suma de gaussianas con AIC (sin cambios) ─────────────────────────
def _gauss_model(x, *p):
    out = np.full_like(x, p[0], dtype=float)
    for i in range((len(p) - 1) // 3):
        A, mu, s = p[1 + 3*i], p[2 + 3*i], p[3 + 3*i]
        out += A * np.exp(-((x - mu) ** 2) / (2.0 * s ** 2))
    return out

def _fit_gauss_n(xs, ys, n):
    rx = xs.max() - xs.min()
    ry = ys.max() - ys.min()
    c0 = float(ys.min())
    mu_locs = np.linspace(xs.min() + 0.05*rx, xs.max() - 0.05*rx, n)
    p0  = [c0];  blo = [c0 - ry - 1.0];  bhi = [float(ys.max())]
    for mu0 in mu_locs:
        idx = int(np.argmin(np.abs(xs - mu0)))
        A0  = max(float(ys[idx] - c0), 0.1)
        s0  = rx / (3.0 * n)
        p0  += [A0,   float(mu0),    s0]
        blo += [0.0,  float(xs.min()), 0.5]
        bhi += [A0*5 + 1.0, float(xs.max()), rx]
    try:
        popt, _ = curve_fit(_gauss_model, xs, ys, p0=p0, bounds=(blo, bhi),
                            maxfev=80_000, method='trf')
        return popt, _gauss_model(xs, *popt)
    except Exception:
        return None, None

def fit_gauss_aic(xs, ys, n_max=8, delta_aic=2.0):
    n_pts = len(xs)
    tabla = []
    for n in range(1, n_max + 1):
        popt, y_pred = _fit_gauss_n(xs, ys, n)
        if y_pred is None:
            continue
        k   = 1 + 3 * n
        ssr = max(float(np.sum((ys - y_pred) ** 2)), 1e-12)
        aic = n_pts * np.log(ssr / n_pts) + 2.0 * k
        tabla.append((n, aic, k, y_pred))
    if not tabla:
        return None
    aic_min    = min(t[1] for t in tabla)
    candidatos = [t for t in tabla if t[1] <= aic_min + delta_aic]
    n_opt, _, k_opt, y_pred = min(candidatos, key=lambda t: t[0])
    r2, rms, ma, mxe = compute_metrics(ys, y_pred)
    return dict(r2=r2, rmse=rms, mae=ma, max_ae=mxe,
                n_params=k_opt, n_campanas=n_opt)

# ── Metodo 3: Spline mejorado ──────────────────────────────────────────────────
# Cambio: s_factor 0.1 → 0.001
# Justificacion: s_factor=0.1 forza s=100 (muy alto), lo que limita el numero de
# nudos a ~11 sin importar la complejidad de la curva.
# Con s_factor=0.001 (s=1) el spline elige ~20 nudos adaptativos y alcanza
# R² medio 0.99999 vs 0.9994 anteriores, sin ser interpolacion exacta.
def fit_spline(xs: np.ndarray, ys: np.ndarray,
               s_factor: float = 0.001, k: int = 5) -> dict | None:
    try:
        spl    = UnivariateSpline(xs, ys, s=s_factor * len(xs), k=k)
        y_pred = spl(xs)
        knots  = spl.get_knots()
        r2, rms, ma, mxe = compute_metrics(ys, y_pred)
        return dict(r2=r2, rmse=rms, mae=ma, max_ae=mxe,
                    n_params=len(knots) + k, n_knots=len(knots),
                    s_factor=s_factor)
    except Exception:
        return None

# ── Loop principal ─────────────────────────────────────────────────────────────
def main():
    print('comparacion_v2.py — Spline mejorado (s_factor=0.001)')
    print(f'  Target dir : {TARGET_DIR}')
    print(f'  Salida CSV : {OUT_CSV}')
    print()

    records = []

    for cid in range(1, 501):
        x, y   = load_curve(cid)
        xs, ys = x[::8], y[::8]

        row = {'curva': cid, 'n_puntos': len(xs)}

        # 1. Polinomio (BIC)
        rp = fit_poly_bic(xs, ys)
        for k, v in rp.items():
            row[f'poly_{k}'] = round(v, 6) if isinstance(v, float) else v

        # 2. Gaussianas (AIC)
        rg = fit_gauss_aic(xs, ys)
        if rg:
            for k, v in rg.items():
                row[f'gauss_{k}'] = round(v, 6) if isinstance(v, float) else v
        else:
            for k in ('r2', 'rmse', 'mae', 'max_ae', 'n_params', 'n_campanas'):
                row[f'gauss_{k}'] = np.nan

        # 3. Spline (s_factor=0.001)
        rs = fit_spline(xs, ys)
        if rs:
            for k, v in rs.items():
                row[f'spline_{k}'] = round(v, 6) if isinstance(v, float) else v
        else:
            for k in ('r2', 'rmse', 'mae', 'max_ae', 'n_params', 'n_knots', 's_factor'):
                row[f'spline_{k}'] = np.nan

        # Mejor metodo segun R²
        r2s = {
            'polinomio':  row['poly_r2'],
            'gaussianas': row.get('gauss_r2', np.nan),
            'spline':     row.get('spline_r2', np.nan),
        }
        validos = {k: v for k, v in r2s.items() if not (isinstance(v, float) and np.isnan(v))}
        if validos:
            mejor = max(validos, key=validos.get)
            row['mejor_metodo'] = mejor
            row['r2_mejor']     = round(validos[mejor], 6)
        else:
            row['mejor_metodo'] = 'none'
            row['r2_mejor']     = np.nan

        records.append(row)

        if cid % 50 == 0:
            print(f'  {cid}/500 completadas')

    df = pd.DataFrame(records)

    col_order = [
        'curva', 'n_puntos', 'mejor_metodo', 'r2_mejor',
        'poly_r2',   'poly_rmse',   'poly_mae',   'poly_max_ae',   'poly_n_params',   'poly_degree',
        'gauss_r2',  'gauss_rmse',  'gauss_mae',  'gauss_max_ae',  'gauss_n_params',  'gauss_n_campanas',
        'spline_r2', 'spline_rmse', 'spline_mae', 'spline_max_ae', 'spline_n_params', 'spline_n_knots', 'spline_s_factor',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)

    # ── Resumen ────────────────────────────────────────────────────────────────
    print(f'\nGuardado: {OUT_CSV}')
    print(f'  {len(df)} filas, {len(df.columns)} columnas\n')
    print('=' * 68)
    print('RESUMEN GLOBAL')
    print('=' * 68)

    for nombre, col_r2, col_extra in [
        ('Polinomio BIC (grado 1-20)',         'poly_r2',   'poly_degree'),
        ('Suma Gaussianas AIC (N 1-8)',         'gauss_r2',  'gauss_n_campanas'),
        ('UnivariateSpline mejorado (s=0.001)', 'spline_r2', 'spline_n_knots'),
    ]:
        vals  = df[col_r2].dropna()
        extra = df[col_extra].dropna()
        print(f'\n{nombre}:')
        print(f'  R² medio:     {vals.mean():.5f}   mediana: {vals.median():.5f}')
        print(f'  R² > 0.99:    {(vals > 0.99).sum()}/500')
        print(f'  R² > 0.999:   {(vals > 0.999).sum()}/500')
        print(f'  {col_extra}: media={extra.mean():.1f}  mediana={extra.median():.0f}  '
              f'rango=[{extra.min():.0f}, {extra.max():.0f}]')

    print('\nMejor metodo por curva:')
    print(df['mejor_metodo'].value_counts().to_string())
    print()


if __name__ == '__main__':
    main()
