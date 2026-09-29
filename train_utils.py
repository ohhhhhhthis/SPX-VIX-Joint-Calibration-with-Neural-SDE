import torch
import time
import checkpoint
import json
import os
import numpy as np
from utils import compute_smile_spx
from utils import compute_smile_vix
from utils import compute_loss_smile_spx
from utils import compute_loss_smile_vix


def train(model, epoch, loss_func, checkpoint_func, save_step):
    opt = model["optimizer"]
    while True:
        opt.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast():
        loss, output = loss_func(epoch)
        loss.backward()

        if epoch % save_step == 1:
            checkpoint_func(model, epoch, loss, output)

        opt.step()
        print(f"{epoch} loss={loss.item()}")
        epoch += 1


def wrap_loss_func(date, params, mkt_smiles, maturities, gen, dt, fSPX):
    batch_size = params["batch_size"]
    xy0 = torch.zeros([batch_size, 2], device="cuda:0")
    spx_maturities = maturities["spx"]
    vix_maturities = maturities["vix"]
    all_maturities = maturities["all"]
    vix_p30 = maturities["vix_p30"]
    spx_vix_maturities = maturities["spx_vix"]

    spx_smiles = mkt_smiles["spx"]
    vix_smiles = mkt_smiles["vix"]

    T_last = max(all_maturities)

    spot = fSPX(0.0)
    t = [0.0] + all_maturities

    # w_fVIX, w_CVIX, w_SPX = params["w_fVIX"], params["w_CVIX"], params["w_SPX"]
    n_vix = len(vix_maturities)
    n_spx = len(spx_maturities)

    def loss_func(epoch, trajectories=False):
        XY, R = gen(t)

        loss_dict = {
            "SPX": 0,
            "fVIX": 0,
            "CVIX": 0,
        }

        spx_lst = list(spx_smiles.keys())
        vix_lst = list(vix_smiles.keys())
        model_smiles = {}  # {maturity.string: {} for maturity in spx_vix_maturities}

        for i, maturity in enumerate(all_maturities):
            if maturity in spx_maturities:
                XYi = XY[:, i + 1]
                maturity_date = spx_lst[spx_maturities.index(maturity)]
                smile_spx = compute_smile_spx(maturity_date, maturity, XYi, spot, spx_smiles)
                model_smiles[maturity_date] = {"spx": {"model": smile_spx, "market": spx_smiles[maturity_date]}}
                if maturity in spx_maturities[:2]:
                    loss_spx = compute_loss_smile_spx(smile_spx, spx_smiles[maturity_date], use_weights=True)
                else:
                    loss_spx = compute_loss_smile_spx(smile_spx, spx_smiles[maturity_date], use_weights=False)
                loss_dict["SPX"] += loss_spx
        # for i, maturity in enumerate(all_maturities):
            if maturity in vix_maturities:
                XYi = XY[:, i + 1]
                j = all_maturities.index(vix_p30[maturity])
                Rij = (R[:, j + 1] - R[:, i + 1]).unsqueeze(-1)
                maturity_date = vix_lst[vix_maturities.index(maturity)]
                smile_vix = compute_smile_vix(maturity_date, maturity, XYi, Rij, vix_smiles)
                model_smiles[maturity_date] = {"vix": {"model": smile_vix, "market": vix_smiles[maturity_date]}}
                loss_fVIX, loss_CVIX = compute_loss_smile_vix(smile_vix, vix_smiles[maturity_date], use_weights=False)
                loss_dict["fVIX"] += loss_fVIX
                loss_dict["CVIX"] += loss_CVIX

        subloss_fVIX = (loss_dict["fVIX"] / n_vix)
        subloss_SPX = (loss_dict["SPX"] / n_spx)
        subloss_CVIX = (loss_dict["CVIX"] / n_vix)

        w_fVIX, w_SPX, w_CVIX = compute_dynamic_weight(
            epoch,
            subloss_fVIX,
            subloss_SPX,
            subloss_CVIX,
            sigma=1,
            method=1
        )

        loss = (
            w_fVIX * loss_dict["fVIX"] / n_vix
            + w_SPX * loss_dict["SPX"] / n_spx
            + w_CVIX * loss_dict["CVIX"] / n_vix
        )

        output = {
            "loss": float(loss.detach().cpu().numpy()),
            "time": time.time(),
            "epoch": epoch,
            "smiles": model_smiles,
            "XY": XY if trajectories else None,
            "R": R if trajectories else None,
        }

        return loss, output

    return t, loss_func


def compute_dynamic_weight(epoch, subloss_fVIX, subloss_SPX, subloss_CVIX, sigma=1, method=1):
    # method1: through
    if method == 1:
        """ Compute dynamic loss weights using a Gaussian-based adjustment """
        loss_fVIX = float(subloss_fVIX)
        loss_SPX = float(subloss_SPX)
        loss_CVIX = float(subloss_CVIX)

        total_loss = loss_fVIX + loss_SPX + loss_CVIX
        if total_loss == 0:
            return 20, 2, 3  # Avoid division by zero, use default weights

        # Compute the loss proportions
        p_fVIX = loss_fVIX / total_loss
        p_SPX = loss_SPX / total_loss
        p_CVIX = loss_CVIX / total_loss

        # Apply Gaussian function for dynamic weighting
        w_fVIX = 20 * np.exp(p_fVIX / sigma ** 2)
        w_SPX = 3 * np.exp(p_SPX / sigma ** 2)
        w_CVIX = 2 * np.exp(p_CVIX / sigma ** 2)

    if method == 2:
        # method2: through epoch
        w_fVIX = float(18 + 4 / 40000 * epoch)
        w_SPX = float(5 + 3 / 40000 * epoch)
        w_CVIX = float(2 + 1 / 40000 * epoch)

        w_fVIX = np.clip(w_fVIX, 18, 22)
        w_SPX = np.clip(w_SPX, 5, 8)
        w_CVIX = np.clip(w_CVIX, 2, 3)

    return w_fVIX, w_SPX, w_CVIX


def checkpoint_func(model, epoch, loss, output):
    name = model["name"]
    js = json.dumps(output, indent=4, cls=checkpoint.Encoder)
    path = f"checkpoints/{name}/{epoch}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(js)
    checkpoint.save_checkpoint(epoch, loss, model)
