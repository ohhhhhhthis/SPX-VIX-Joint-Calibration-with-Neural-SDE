import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
from bs import bs_torch

plt.rcParams["figure.figsize"] = (9, 6)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Smile:
    def __init__(self, instrument, maturity, data):
        self.name = f"{instrument}"
        self.maturity = maturity
        self.T = data["T"].clone()
        self.strikes = data["strikes"].clone()
        self.fwd = data["fwd"].clone()
        self.fwd_ask = data["fwd_ask"].clone()
        self.fwd_bid = data["fwd_bid"].clone()
        self.r = data["r"].clone()
        self.mids = data["mids"].clone()
        self.asks = data["asks"].clone()
        self.bids = data["bids"].clone()
        if data["calls"] is None:
            self.calls = bs_torch(self.fwd, self.T, self.strikes, self.r, self.mids, 1)
        else:
            self.calls = data["calls"].clone()

        if data["puts"] is None:
            self.puts = bs_torch(self.fwd, self.T, self.strikes, self.r, self.mids, -1)
        else:
            self.puts = data["puts"].clone()

    def plot(self, color=None, legend=True):
        with torch.no_grad():
            plt.plot(
                self.strikes.cpu(),
                self.mids.cpu(),
                color=color,
                label=f"{self.name} mid implied vol.",
            )
            plt.scatter(
                self.strikes.cpu(), self.mids.cpu(), marker="x", s=20, color=color
            )

            plt.fill_between(
                self.strikes.cpu(),
                self.bids.cpu(),
                self.asks.cpu(),
                color=color,
                alpha=0.5,
                label="bid-ask",
            )

            plt.axvline(
                self.fwd.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.axvline(
                self.fwd_ask.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.axvline(
                self.fwd_bid.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.xlabel("Strike")
            plt.ylabel("Implied volatility")
            if legend:
                plt.legend()
            plt.title(self.name)

    def to_json(self):
        data = {
            "name": self.name,
            "maturity": self.maturity,
            "T": self.T.detach().cpu().tolist(),
            "strikes": self.strikes.detach().cpu().tolist(),
            "fwd": float(self.fwd.detach().cpu().numpy()),
            "fwd_bid": float(self.fwd_bid.detach().cpu().numpy()),
            "fwd_ask": float(self.fwd_ask.detach().cpu().numpy()),
            "r": self.r.detach().cpu().tolist(),
            "mids": self.mids.cpu().detach().tolist(),
            "asks": self.asks.detach().cpu().tolist(),
            "bids": self.bids.detach().cpu().tolist(),
            "calls": self.bids.detach().cpu().tolist(),
            "puts": self.bids.detach().cpu().tolist(),
        }
        return data

    def from_json(js):
        name = js["name"]
        maturity = js["maturity"]
        T = torch.tensor(js["T"], device=device)
        strikes = torch.tensor(js["strikes"], device=device)
        fwd = torch.tensor(js["fwd"], device=device)
        fwd_bid = torch.tensor(js["fwd_bid"], device=device)
        fwd_ask = torch.tensor(js["fwd_ask"], device=device)
        r = torch.tensor(js["r"], device=device)
        mids = torch.tensor(js["mids"], device=device)
        asks = torch.tensor(js["asks"], device=device)
        bids = torch.tensor(js["bids"], device=device)
        calls = torch.tensor(js["calls"], device=device)
        puts = torch.tensor(js["puts"], device=device)
        return Smile(
            name,
            maturity,
            {
                "name": name,
                "T": T,
                "strikes": strikes,
                "fwd": fwd,
                "fwd_bid": fwd_bid,
                "fwd_ask": fwd_ask,
                "r": r,
                "mids": mids,
                "asks": asks,
                "bids": bids,
                "calls": calls,
                "puts": puts,
            },
        )
