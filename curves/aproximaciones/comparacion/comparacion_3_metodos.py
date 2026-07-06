"""
comparacion_3_metodos.py

Compara tres metodos de aproximacion sobre las 500 curvas target:

  1. Polinomio (grado optimo por BIC, grados 1-20)
     Replica la logica central de polinomiales/aproximacion_polinomial_v2.ipynb:
     ajusta directamente sobre los datos target submuetreados y selecciona
     el grado que minimiza el BIC.

  2. Suma de gaussianas (N optimo por AIC, N=1-8, campanas uniformes)
     Replica curves/aproximaciones/gaussian/02_suma_gaussianas_splines.ipynb
     seccion 6: seleccion automatica de N por AIC con delta=2.

  3. UnivariateSpline suavizante (s=0.1*n, k=5)
     Replica la seccion 7 del mismo notebook gaussiano.

Metricas por curva y metodo:
  r2, rmse, mae, max_abs_error, n_params (complejidad del modelo)
  + parametro propio: poly_degree / gauss_n_campanas / spline_n_knots

Salida: comparacion_3_metodos.csv en este mismo directorio.
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
OUT_CSV    = SCRIPT_DIR / 'comparacion_3_metodos.csv'

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

# ── Metodo 1: Polinomio con BIC ────────────────────────────────────────────────
def fit_poly_bic(xs: np.ndarray, ys: np.ndarray, max_deg: int = 20) -> dict:
    """
    Prueba grados 1..max_deg y elige el que minimiza el BIC.
    BIC = n*log(SSR/n) + k*log(n),  k = grado + 1 parametros.
    """
    n = len(xs)
    best_bic  = np.inf
    best_deg  = 1
    best_pred = None

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

# ── Metodo 2: Suma de gaussianas con AIC ───────────────────────────────────────
def _gauss_model(x, *p):
    """y = c + sum_i A_i * exp(-((x-mu_i)^2)/(2*sigma_i^2))"""
    out = np.full_like(x, p[0], dtype=float)
    for i in range((len(p) - 1) // 3):
        A, mu, s = p[1 + 3*i], p[2 + 3*i], p[3 + 3*i]
        out += A * np.exp(-((x - mu) ** 2) / (2.0 * s ** 2))
    return out

def _fit_gauss_n(xs: np.ndarray, ys: np.ndarray, n: int):
    """Ajusta suma de N gaussianas con campanas uniformes. Devuelve (popt, y_pred) o (None, None)."""
    rx   = xs.max() - xs.min()
    ry   = ys.max() - ys.min()
    c0   = float(ys.min())
    margin = 0.05 * rx
    mu_locs = np.linspace(xs.min() + margin, xs.max() - margin, n)

    p0  = [c0]
    blo = [c0 - ry - 1.0]
    bhi = [float(ys.max())]
    for mu0 in mu_locs:
        idx = int(np.argmin(np.abs(xs - mu0)))
        A0  = max(float(ys[idx] - c0), 0.1)
        s0  = rx / (3.0 * n)
        p0  += [A0,   float(mu0),    s0]
        blo += [0.0,  float(xs.min()), 0.5]
        bhi += [A0*5 + 1.0, float(xs.max()), rx]

    try:
        popt, _ = curve_fit(_gauss_model, xs, ys,
                            p0=p0, bounds=(blo, bhi),
                            maxfev=80_000, method='trf')
        return popt, _gauss_model(xs, *popt)
    except Exception:
        return None, None

def fit_gauss_aic(xs: np.ndarray, ys: np.ndarray,
                  n_max: int = 8, delta_aic: float = 2.0) -> dict | None:
    """
    Prueba N=1..n_max campanas gaussianas y elige el N optimo por AIC.
    Entre todos los N con AIC <= AIC_min + delta_aic, elige el menor (mas parsimonioso).
    """
    n_pts  = len(xs)
    tabla  = []  # (n, aic, k, y_pred)

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

    aic_min     = min(t[1] for t in tabla)
    candidatos  = [t for t in tabla if t[1] <= aic_min + delta_aic]
    n_opt, _, k_opt, y_pred = min(candidatos, key=lambda t: t[0])

    r2, rms, ma, mxe = compute_metrics(ys, y_pred)
    return dict(r2=r2, rmse=rms, mae=ma, max_ae=mxe,
                n_params=k_opt, n_campanas=n_opt)

# ── Metodo 3: UnivariateSpline ─────────────────────────────────────────────────
def fit_spline(xs: np.ndarray, ys: np.ndarray,
               s_factor: float = 0.1, k: int = 5) -> dict | None:
    """
    UnivariateSpline suavizante con s = s_factor * n_puntos y grado k.
    Igual a la seccion 7 del notebook gaussiano (s=0.1, k=5).
    """
    try:
        spl    = UnivariateSpline(xs, ys, s=s_factor * len(xs), k=k)
        y_pred = spl(xs)
        knots  = spl.get_knots()
        r2, rms, ma, mxe = compute_metrics(ys, y_pred)
        return dict(r2=r2, rmse=rms, mae=ma, max_ae=mxe,
                    n_params=len(knots) + k, n_knots=len(knots))
    except Exception:
        return None

# ── Loop principal ─────────────────────────────────────────────────────────────
def main():
    print('Comparando 3 metodos en 500 curvas...')
    print(f'  Target dir : {TARGET_DIR}')
    print(f'  Salida CSV : {OUT_CSV}')
    print()

    records = []

    for cid in range(1, 501):
        x, y   = load_curve(cid)
        xs, ys = x[::8], y[::8]   # ~1000 puntos, igual que notebook gaussiano

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

        # 3. Spline
        rs = fit_spline(xs, ys)
        if rs:
            for k, v in rs.items():
                row[f'spline_{k}'] = round(v, 6) if isinstance(v, float) else v
        else:
            for k in ('r2', 'rmse', 'mae', 'max_ae', 'n_params', 'n_knots'):
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

    # Orden de columnas
    col_order = [
        'curva', 'n_puntos', 'mejor_metodo', 'r2_mejor',
        'poly_r2', 'poly_rmse', 'poly_mae', 'poly_max_ae', 'poly_n_params', 'poly_degree',
        'gauss_r2', 'gauss_rmse', 'gauss_mae', 'gauss_max_ae', 'gauss_n_params', 'gauss_n_campanas',
        'spline_r2', 'spline_rmse', 'spline_mae', 'spline_max_ae', 'spline_n_params', 'spline_n_knots',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)

    # ── Resumen impreso ────────────────────────────────────────────────────────
    print(f'\nGuardado: {OUT_CSV}')
    print(f'  {len(df)} filas, {len(df.columns)} columnas\n')
    print('=' * 65)
    print('RESUMEN GLOBAL')
    print('=' * 65)

    for nombre, col_r2, col_extra in [
        ('Polinomio BIC (grado 1-20)',       'poly_r2',  'poly_degree'),
        ('Suma Gaussianas AIC (N 1-8)',       'gauss_r2', 'gauss_n_campanas'),
        ('UnivariateSpline (s=0.1, k=5)',    'spline_r2','spline_n_knots'),
    ]:
        vals = df[col_r2].dropna()
        extra = df[col_extra].dropna()
        print(f'\n{nombre}:')
        print(f'  R² medio:     {vals.mean():.4f}   mediana: {vals.median():.4f}')
        print(f'  R² > 0.95:    {(vals > 0.95).sum()}/500')
        print(f'  R² > 0.99:    {(vals > 0.99).sum()}/500')
        print(f'  {col_extra}: media={extra.mean():.1f}  mediana={extra.median():.0f}  '
              f'rango=[{extra.min():.0f}, {extra.max():.0f}]')

    print('\nMejor metodo por curva:')
    print(df['mejor_metodo'].value_counts().to_string())
    print()

if __name__ == '__main__':
    main()
