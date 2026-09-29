import torch
import cupy as cp
import numpy as np
import pandas as pd
from bs import brent_iv
from smile import Smile
from utils import tensorize_smile
from collections import OrderedDict
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# plt.style.use("seaborn")
SPX = pd.read_csv("./data/csv/spx_bid_ask.csv")
VIX = pd.read_csv("./data/csv/vix_bid_ask.csv")
# YIELD_CURVE = pd.read_csv("./data/csv/yield-curve.csv")
SPX_SPOT = pd.read_csv("./data/csv/spx_spot.csv")


def choose_fwd(asks_fwd, bids_fwd):
    def f(mkt_fwd):
        d = np.abs(asks_fwd - mkt_fwd) + np.abs(mkt_fwd - bids_fwd)
        return d.mean()

    fwd0 = 0.5 * (bids_fwd[0] + asks_fwd[0])
    mkt = minimize(f, x0=fwd0).x[0]

    bid = bids_fwd[bids_fwd < mkt].max()
    ask = asks_fwd[asks_fwd > mkt].min()

    return mkt, bid, ask


def extract_yield_curve(yield_curve):
    return {"days": yield_curve.days.values, "rate": yield_curve.rate.values}


def get_fwd_and_price(df, maturity, T0, r=0):
    calls = df.query("cp_flag == 'C'").sort_values(by='strike_price')
    puts = df.query("cp_flag == 'P'").sort_values(by='strike_price')
    # Make sure calls and puts are in pairs for different strikes
    # assert np.equal(calls.strike_price.values, puts.strike_price.values).all()
    data = pd.merge(calls, puts, on='strike_price', how='inner', suffixes=('_call', '_put'))
    strikes = data.strike_price.values
    call_bids = data.best_bid_call.values
    call_asks = data.best_offer_call.values
    put_bids = data.best_bid_put.values
    put_asks = data.best_offer_put.values

    T = (pd.to_datetime(maturity) - T0).days / 365
    asks_fwd = np.exp(r*T) * (call_asks - put_bids) + strikes
    bids_fwd = np.exp(r*T) * (call_bids - put_asks) + strikes
    fwd, fwd_bid, fwd_ask = choose_fwd(asks_fwd, bids_fwd)

    # call_mids = (call_bids + call_asks) / 2
    # put_mids = (put_bids + put_asks) / 2
    # fwds = np.exp(r * T) * (call_mids - put_mids) + strikes
    # fwd = fwds.mean()

    OTM_put = (strikes <= fwd)
    cp_flags = np.where(OTM_put, -1, 1)  # 1 for call, -1 for put.
    price_bids = np.where(OTM_put, put_bids, call_bids)
    price_asks = np.where(OTM_put, put_asks, call_asks)

    return pd.DataFrame({
        'maturity': maturity,
        'T': T,
        'fwd': fwd,
        'fwd_bid': fwd_bid,
        'fwd_ask': fwd_ask,
        'r': r,
        'cp_flags': cp_flags,
        'strikes': strikes,
        'price_bids': price_bids,
        'price_asks': price_asks
    })


def compute_vol_surface(df, lb=0.001, ub=20, xtol=1e-6, rtol=8.88e-8, maxiter=500):  # float32 can't reach 2e-12, 8.88e-16
    # Load data
    maturity = df['maturity'].to_numpy()
    T, fwd, fwd_bid, fwd_ask, r, cp_flags, strikes, price_bids, price_asks = \
        cp.asarray(df.iloc[:, 1:].to_numpy(), dtype=cp.float32).T
    # price_mids = (price_bids + price_asks) / 2.0

    # Declare variables
    n = T.size
    vol_bids = cp.empty(n, dtype=cp.float32)
    vol_asks = cp.empty(n, dtype=cp.float32)
    # vol_mids = cp.empty(n, dtype=cp.float32)

    # Compute IV by Brent method in CUDA
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    brent_iv((grid_size,), (block_size,),
             (fwd, T, strikes, price_bids, cp_flags, r, cp.float32(lb), cp.float32(ub),
              vol_bids, cp.float32(xtol), cp.float32(rtol), cp.float32(maxiter)))
    brent_iv((grid_size,), (block_size,),
             (fwd, T, strikes, price_asks, cp_flags, r, cp.float32(lb), cp.float32(ub),
              vol_asks, cp.float32(xtol), cp.float32(rtol), cp.float32(maxiter)))
    # brent_iv((grid_size,), (block_size,),
    #          (fwd, T, strikes, price_mids, cp_flags, r, cp.float32(lb), cp.float32(ub),
    #           vol_mids, cp.float32(xtol), cp.float32(rtol), cp.float32(maxiter)))
    vol_mids = (vol_bids + vol_asks) / 2

    # Drop NaN
    mask = cp.logical_or(cp.isnan(vol_asks), cp.isnan(vol_bids))
    # mask = cp.logical_or(cp.isnan(vol_mids), mask)

    result = {}
    for t in cp.unique(T[~mask]):
        Tmask = (T == t)
        result[maturity[Tmask.get()][0]] = {
            "T": t,
            "fwd": fwd[Tmask][0],
            "fwd_bid": fwd_bid[Tmask][0],
            "fwd_ask": fwd_ask[Tmask][0],
            'r': r[Tmask][0],
            "strikes": strikes[Tmask],
            "asks": vol_asks[Tmask],
            "bids": vol_bids[Tmask],
            "mids": vol_mids[Tmask]
        }

    return result


def fwd_curve(fwd, spx_spot, spx_maturities):
    t = np.hstack((0, spx_maturities))
    fwd = np.hstack((spx_spot, fwd))
    fSPX_func = interp1d(t, fwd, kind="cubic")
    fSPX = lambda t: float(fSPX_func(np.maximum(t, 0.0)))
    return fSPX


def tensorize_smile(data, maturity, instrument, nb_points, alpha1=None, alpha2=None, device="cuda:0"):
    data = data[maturity]
    ts = lambda array: torch.tensor(array, device=device, dtype=torch.float32)
    T = ts(data["T"])
    fwd = ts(data["fwd"])
    fwd_bid = ts(data["fwd_bid"])
    fwd_ask = ts(data["fwd_ask"])
    r = ts(data["r"])
    strikes = ts(data["strikes"])
    mids = ts(data["mids"])
    asks = ts(data["asks"])
    bids = ts(data["bids"])

    if alpha1 and alpha2:
        mask = torch.logical_and(alpha1 * fwd <= strikes, strikes <= alpha2 * fwd)
        strikes, mids, asks, bids = strikes[mask], mids[mask], asks[mask], bids[mask]

    m = strikes.shape[0]
    k = max(1, m // nb_points + 1)
    strikes, mids, asks, bids = strikes[::k], mids[::k], asks[::k], bids[::k]
    data = {
        "T": T,
        "fwd": fwd,
        "fwd_bid": fwd_bid,
        "fwd_ask": fwd_ask,
        "r": r,
        "strikes": strikes,
        "mids": mids,
        "bids": bids,
        "asks": asks,
        "calls": None,
        "puts": None,
    }
    return Smile(instrument, maturity, data)


def load_market_data(date, maturities):
    spx = SPX[(SPX['date'] == date) & (SPX['exdate'].isin(maturities['spx']))].sort_values(by='exdate')
    vix = VIX[(VIX['date'] == date) & (VIX['exdate'].isin(maturities['vix']))]
    # yield_curve = YIELD_CURVE.query(f"date == '{date}'")
    # yield_curve = extract_yield_curve(yield_curve)
    spx_spot = SPX_SPOT.loc[SPX_SPOT['date'] == date, 'spot'].values

    # Make sure data exists at this date.
    T0 = spx.date.drop_duplicates().values
    assert len(T0) == 1
    T0 = pd.to_datetime(T0[0])  # T0 = pd.to_datetime(date)

    # Compute forward price by put-call parity and extract out-the-money option prices for IV calculation.
    spx = spx.groupby('exdate', group_keys=False).apply(lambda x: get_fwd_and_price(x, x.name, T0))
    vix = vix.groupby('exdate', group_keys=False).apply(lambda x: get_fwd_and_price(x, x.name, T0))

    spx_mkt_smiles = compute_vol_surface(spx)
    vix_mkt_smiles = compute_vol_surface(vix)

    spx_maturities = [(pd.to_datetime(i) - T0).days / 365 for i in spx_mkt_smiles.keys()]
    vix_maturities = [(pd.to_datetime(i) - T0).days / 365 for i in vix_mkt_smiles.keys()]

    vix_maturities_plus30d = [(pd.to_datetime(i) + pd.Timedelta(days=30) - T0).days / 365 for i in vix_mkt_smiles.keys()]
    spx_vix_maturities = sorted(set(spx_maturities + vix_maturities))
    all_maturities = sorted(set(spx_maturities + vix_maturities + vix_maturities_plus30d))
    maturities = {
        "spx": spx_maturities,
        "vix": vix_maturities,
        "spx_vix": spx_vix_maturities,
        "vix_p30": dict(zip(vix_maturities, vix_maturities_plus30d)),
        "all": all_maturities,
    }

    spx_mkt_smiles = OrderedDict(
        {
            maturity: tensorize_smile(spx_mkt_smiles, maturity, "SPX market", 80, 0.40, 1.20)
            for maturity in spx_mkt_smiles.keys()
        }
    )

    vix_mkt_smiles = OrderedDict(
        {
            maturity: tensorize_smile(vix_mkt_smiles, maturity, "VIX market", 80, 0.90, 1.60)
            for maturity in vix_mkt_smiles.keys()
        }
    )

    mkt_smiles = {"spx": spx_mkt_smiles, "vix": vix_mkt_smiles}
    fSPX = fwd_curve([smile.fwd.cpu() for smile in spx_mkt_smiles.values()], spx_spot, spx_maturities)
    return mkt_smiles, maturities, fSPX
