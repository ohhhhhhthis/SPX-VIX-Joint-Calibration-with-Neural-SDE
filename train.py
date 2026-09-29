import json
import torch
import checkpoint
import argparse
import my_model
from math import log
from market import load_market_data
from generatorQMC import GeneratorIto
from train_utils import train
from train_utils import wrap_loss_func
from train_utils import checkpoint_func


parser = argparse.ArgumentParser(prog="SPX-VIX joint neural calibration")
parser.add_argument("--date", type=str)
parser.add_argument("--maturities", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--params", type=str)

args, _ = parser.parse_known_args()
args.date = '2021-10-01'
args.maturities = 'maturities.json'
args.model = 'my_model.py'
args.params = 'params.json'
date = args.date

with open(args.maturities) as file:
    maturities = json.load(file)

with open(args.params) as file:
    params = json.load(file)

m = int(log(params["batch_size"], 2))-4
params["batch_size"] = 2**m

mkt_smiles, maturities, fSPX = load_market_data(date, maturities)

dt = torch.tensor(params["dt"] / 365)  # , device="cuda:0"
GEN = GeneratorIto(dt, my_model.V_AND_MUY(my_model.MODEL), fSPX, params["batch_size"])
model, epoch = checkpoint.load_last_checkpoint(my_model.MODEL)


_, loss_func = wrap_loss_func(date, params, mkt_smiles, maturities, GEN, dt, fSPX)


train(my_model.MODEL, epoch, loss_func, checkpoint_func, save_step=100)



