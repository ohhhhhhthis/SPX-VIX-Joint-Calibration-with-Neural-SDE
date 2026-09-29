import cupy as cp
import numpy as np
import torch
from cupyx.scipy.special import erf
from cupyx import jit
from scipy.optimize import brentq
from scipy.special import erf


def bs_torch(S, T, K, r, sigma, cp_flag):
    sigma = torch.abs(sigma)
    sT = torch.sqrt(T)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * T) / sigma / sT
    d2 = d1 - sigma * sT
    N1 = 0.5 * (1 + torch.erf(cp_flag * d1 / np.sqrt(2)))
    N2 = 0.5 * (1 + torch.erf(cp_flag * d2 / np.sqrt(2)))
    price = torch.exp(-r * T) * cp_flag * (N1 * S - N2 * K)
    return price


def bs_np(S, T, K, r, sigma, cp_flag):
    sigma = np.abs(sigma)
    sT = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sigma / sT
    d2 = d1 - sigma * sT
    N1 = 0.5 * (1 + erf(cp_flag * d1 / np.sqrt(2)))
    N2 = 0.5 * (1 + erf(cp_flag * d2 / np.sqrt(2)))
    price = np.exp(-r * T) * cp_flag * (N1 * S - N2 * K)
    return price


def gammabs(S, t, T, K, sigma):
    tT = T - t
    tT = torch.reshape(tT, [t.size()[0], 1, 1])
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    g = torch.exp(-0.5 * d1 * d1) / sqrt(2 * np.pi) / S / sigma / stT
    return g


def deltabs(S, r, T, K, sigma):
    tT = T - t
    tT = torch.reshape(tT, [t.size()[0], 1, 1])
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * tT) / sigma / stT
    N = 0.5 * (1 + torch.erf(d1 / np.sqrt(2.0)))
    return N


def vegabs(S, T, K, sigma):
    sT = torch.sqrt(T)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * T) / sigma / sT
    v = torch.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * S * sT
    return v


kernel_code = r'''
extern "C" {
    __device__  float norm_cdf(float x) {
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    }

    __device__ float bsm_price(float f, float T, float K, float sigma, float cp_flag, float r) {
        float d1 = (logf(f / K) + 0.5f * sigma * sigma * T) / (sigma * sqrtf(T));
        float d2 = d1 - sigma * sqrtf(T);
        float N1 = norm_cdf(cp_flag * d1);
        float N2 = norm_cdf(cp_flag * d2);
        return expf(-r * T) * cp_flag * (f * N1 - K * N2);
    }

    __global__ void brent_iv(const float* fwd, const float* T, const float* K, const float* price,
        const float* cp_flag, const float* r, float xa, float xb, float* roots,
        float xtol, float rtol, int maxiter) {
    
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        //if (tid >= n) return;
    
        // Initialize variables
        float xpre = xa, xcur = xb;
        float fpre = bsm_price(fwd[tid], T[tid], K[tid], xpre, cp_flag[tid], r[tid]) - price[tid];
        float fcur = bsm_price(fwd[tid], T[tid], K[tid], xcur, cp_flag[tid], r[tid]) - price[tid];
    
        float xblk = 0.0f, fblk = 0.0f;
        float spre = 0.0f, scur = 0.0f, sbis = 0.0f;
        float delta = 0.0f;
        float stry = 0.0f, dpre = 0.0f, dblk = 0.0f;

        // Check initial conditions
        if (fpre == 0.0f) {
            roots[tid] = xpre;
            return;
        }
        if (fcur == 0.0f) {
            roots[tid] = xcur;
            return;
        }
        if ((fpre > 0) == (fcur > 0)) {
            roots[tid] = nanf(""); //NAN;
            return;
        }

        // Main iteration loop
        for (int i = 0; i < maxiter; i++) {
            // Update bracketing info
            if (fpre != 0.0f && fcur != 0.0f && ((fpre > 0) != (fcur > 0))) {
                xblk = xpre;
                fblk = fpre;
                spre = scur = xcur - xpre;
            }

            // Ensure |fblk| >= |fcur|
            if (fabsf(fblk) < fabsf(fcur)) {
                xpre = xcur;
                xcur = xblk;
                xblk = xpre;

                fpre = fcur;
                fcur = fblk;
                fblk = fpre;
            }

            // Compute tolerance
            delta = (xtol + rtol * fabsf(xcur)) / 2.0f;
            sbis = (xblk - xcur) / 2.0f;

            // Check convergence
            if (fcur == 0.0f || fabsf(sbis) < delta) {
                roots[tid] = xcur;
                return;
            }

            // Decide interpolation or bisection
            if (fabsf(spre) > delta && fabsf(fcur) < fabsf(fpre)) {
                if (xpre == xblk) {
                    // Linear interpolation (secant)
                    stry = -fcur * (xcur - xpre) / (fcur - fpre);
                }
                else {
                    // Inverse quadratic interpolation
                    dpre = (fpre - fcur) / (xpre - xcur);
                    dblk = (fblk - fcur) / (xblk - xcur);
                    stry = -fcur * (fblk * dblk - fpre * dpre) / 
                        (dblk * dpre * (fblk - fpre));
                }

                if (2.0f * fabsf(stry) < fminf(fabsf(spre), 3.0f * fabsf(sbis) - delta)) { // good short step
                    spre = scur;
                    scur = stry;
                }
                else { // bisect
                    spre = sbis;
                    scur = sbis;
                }
            }
            else { // bisect
                spre = sbis;
                scur = sbis;
            }

            // Update for next iteration
            xpre = xcur;
            fpre = fcur;
            if (fabsf(scur) > delta) {
                xcur += scur;
            }
            else {
                xcur += (sbis > 0 ? delta : -delta);
            }

            fcur = bsm_price(fwd[tid], T[tid], K[tid], xcur, cp_flag[tid], r[tid]) - price[tid];
        }
        roots[tid] = xcur;
    }

} // extern "C"
'''

module = cp.RawModule(code=kernel_code)
brent_iv = module.get_function("brent_iv")


def implied_vol_np(S, T, K, r, C, cp_flag, nan=False):
    f = lambda sigma: bs_np(S, T, K, r, sigma, cp_flag) - C
    try:
        i = brentq(f, a=0.001, b=20)
        if nan and i == 0.001:
            i = np.nan
    except:
        i = np.nan
    return i


class ImpliedVolTorch_C(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, T, K, r, C):
        S_np, C_np = S.cpu().detach().numpy(), C.cpu().detach().numpy()
        K_np = K.cpu().detach().numpy()
        T_np = T.cpu().detach().numpy()
        r_np = r.cpu().detach().numpy()
        sigma = torch.tensor(implied_vol_np(S_np, T_np, K_np, r_np, C_np, cp_flag=1))
        ctx.save_for_backward(sigma, S, T, K, C)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        sigma, S, T, K, C = ctx.saved_tensors
        grad_input = grad_output.clone()
        v = vegabs(S, T, K, sigma)
        g = grad_input / v
        if g.isnan():
            return (
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
            )
        else:
            return 0 * grad_input, 0 * grad_input, 0 * grad_input, 0 * grad_input, g


class ImpliedVolTorch_P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, T, K, r, C):
        S_np, C_np = S.cpu().detach().numpy(), C.cpu().detach().numpy()
        K_np = K.cpu().detach().numpy()
        T_np = T.cpu().detach().numpy()
        r_np = r.cpu().detach().numpy()
        sigma = torch.tensor(implied_vol_np(S_np, T_np, K_np, r_np, C_np, cp_flag=-1))
        ctx.save_for_backward(sigma, S, T, K, C)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        sigma, S, T, K, C = ctx.saved_tensors
        grad_input = grad_output.clone()
        v = vegabs(S, T, K, sigma)
        g = grad_input / v
        if g.isnan():
            return (
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
            )
        else:
            return 0 * grad_input, 0 * grad_input, 0 * grad_input, 0 * grad_input, g


def implied_vol(S, T, K, r, C, P, device=None):
    IC = lambda k, c: ImpliedVolTorch_C.apply(S.detach().cpu(), T, k, r, c)
    IP = lambda k, p: ImpliedVolTorch_P.apply(S.detach().cpu(), T, k, r, p)
    I = [(IC(k, c) if k > S else IP(k, p)) for k, c, p in zip(K, C.cpu(), P.cpu())]
    I = torch.stack(I)
    return I.to(device)
