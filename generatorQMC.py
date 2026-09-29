import torch
from math import log, sqrt

class GeneratorIto(torch.nn.Module):
    sde_type = "ito"
    noise_type = "general"

    def __init__(self, dt, V_and_Mu, fSPX, batch_size):
        super(GeneratorIto, self).__init__()
        self.dt = dt
        self.fSPX = fSPX
        self.batch_size = batch_size
        self.V_and_Mu = V_and_Mu
        self.tau = torch.tensor(30 / 365, device="cuda:0")

    def f_and_g(self, t, XY):
        XY = XY[:, :2]
        tXY = torch.cat([t * torch.ones([self.batch_size, 1], device="cuda:0"), XY], dim=1)
        V, MuY = self.V_and_Mu(tXY)
        Sigma2 = V[:, [0]] ** 2  # + V[:, [1]] ** 2
        zeros = torch.zeros([self.batch_size, 1], device="cuda:0")
        V = torch.concat([V, zeros, zeros], dim=-1).reshape([self.batch_size, 3, 2])
        fwd = log(self.fSPX(t) / self.fSPX(t - self.dt)) / self.dt
        Mu = torch.concat([fwd - 0.5 * Sigma2, MuY, Sigma2 / self.tau], dim=-1)
        return Mu, V

    def gen_sobol(self, batch_size, time_steps):
        sobol = torch.quasirandom.SobolEngine(dimension=len(time_steps) * 2, scramble=True)
        sobol.fast_forward(2 ** 10)
        uniform = sobol.draw(batch_size).to("cuda:0")
        uniform = 0.5 + (1 - torch.finfo(uniform.dtype).eps) * (uniform - 0.5)  # 1.19e-7
        dw = torch.special.erfinv(2 * uniform - 1) * sqrt(2) * (self.dt ** 0.5)
        dw = dw.reshape(batch_size, len(time_steps), 2)
        return dw

    def sdeint(self, maturities, time_steps, dw):
        curr_t = 0
        curr_xyr = torch.zeros([self.batch_size, 3], device="cuda:0")
        xyr = []

        for i in range(dw.shape[1]):
            Mu, V = self.f_and_g(curr_t, curr_xyr[:, :2])
            diffusion = torch.einsum("nij,nj->ni", V, dw[:, i, :])
            curr_xyr += Mu * self.dt + diffusion
            curr_t += self.dt
            xyr.append(curr_xyr.clone())

        xyr = torch.stack(xyr, dim=1)
        maturities = torch.tensor(maturities, dtype=torch.float32, device="cuda:0")
        indices = torch.searchsorted(time_steps, maturities)
        alphas = (maturities - time_steps[indices - 1]) / (time_steps[indices] - time_steps[indices - 1])
        return (1 - alphas[None, :, None]) * xyr[:, indices - 1] + alphas[None, :, None] * xyr[:, indices]

    def forward(self, maturities):
        T_last = maturities[-1]
        time_steps = torch.arange(0, T_last+self.dt, self.dt, dtype=torch.float32, device='cuda:0')
        dw = self.gen_sobol(self.batch_size, time_steps)
        xyr = self.sdeint(maturities, time_steps, dw)
        xy = xyr[:, :, :2]
        r = xyr[:, :, 2]
        return xy, r
