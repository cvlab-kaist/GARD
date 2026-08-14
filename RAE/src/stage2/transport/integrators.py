import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
import torch 

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift

    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t_curr
        dw = w_cur * th.sqrt(t_curr - t_next)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(t_curr - t_next)
        diffusion = self.diffusion(x, th.ones(x.size(0)).to(x) * t_curr)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(
            xhat, th.ones(x.size(0)).to(x) * t_curr, model, **model_kwargs
        )
        xp = xhat - (t_curr - t_next) * K1
        K2 = self.drift(
            xp, th.ones(x.size(0)).to(x) * t_next, model, **model_kwargs
        )
        return xhat - 0.5 * (t_curr - t_next) * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[th.Tensor]:
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type


    def sample(self, x, model, **model_kwargs):
        
        # x: pure_noise (gaussian noise)
        

        # Reset per-scene step counter so attn map folders are iter10/iter20/...
        # for every scene rather than accumulating across scenes.
        if hasattr(self.drift, 'reset_step_counter'):
            self.drift.reset_step_counter()

        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t_batch = torch.ones(
                x[0].size(0) if isinstance(x, tuple) else x.size(0),
                device=device
            ) * t
            return self.drift(x, t_batch, model, **model_kwargs)

        t = self.t.to(device)

        x_curr = x

        for i in range(len(t) - 1):
            t_pair = torch.tensor([t[i], t[i+1]], device=device)
            x_curr = odeint(
                _fn,
                x_curr,
                t_pair,
                method=self.sampler_type,
                atol=self.atol,
                rtol=self.rtol,
            )[-1]
            
            # PHO4 (ON)
            # add safety clamp
            x_curr = torch.nan_to_num(x_curr, nan=0.0, posinf=1e4, neginf=-1e4)
            x_curr = torch.clamp(x_curr, -1e4, 1e4)


        return x_curr.unsqueeze(0)


    # def sample(self, x, model, **model_kwargs) -> tuple[th.Tensor]:
        
    #     # breakpoint()
    #     device = x[0].device if isinstance(x, tuple) else x.device
    #     def _fn(t, x):
    #         t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
    #         model_output = self.drift(x, t, model, **model_kwargs)
    #         return model_output

    #     t = self.t.to(device)
    #     atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
    #     rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
    #     samples = odeint(
    #         _fn,
    #         x,
    #         t,
    #         method=self.sampler_type,
    #         atol=atol,
    #         rtol=rtol
    #     )
    #     return samples


    # def sample(self, x, model, return_trajectory=False, **model_kwargs):
    #     device = x[0].device if isinstance(x, tuple) else x.device
    #     t = self.t.to(device)
        
    #     x_curr = x
    #     if return_trajectory:
    #         traj = [x_curr]
        
    #     with th.no_grad():  # Ensure no gradient tracking
    #         for i in range(len(t) - 1):
    #             dt = (t[i + 1] - t[i]).item()  # Convert to Python scalar once
                
    #             if isinstance(x_curr, tuple):
    #                 t_batch = t[i].expand(x_curr[0].size(0))
    #             else:
    #                 t_batch = t[i].expand(x_curr.size(0))
                
    #             dx = self.drift(x_curr, t_batch, model, **model_kwargs)
                
    #             if isinstance(x_curr, tuple):
    #                 x_curr = tuple(x_i + dt * dx_i for x_i, dx_i in zip(x_curr, dx))
    #             else:
    #                 x_curr = x_curr + dt * dx
                
    #             if return_trajectory:
    #                 traj.append(x_curr)
        
    #     if return_trajectory:
    #         if isinstance(x_curr, tuple):
    #             traj = tuple(th.stack([step[i] for step in traj], dim=0) for i in range(len(x_curr)))
    #         else:
    #             traj = th.stack(traj, dim=0)
    #         return traj
        
    #     if isinstance(x_curr, tuple):
    #         return tuple(x_i.unsqueeze(0) for x_i in x_curr)
    #     else:
    #         return x_curr.unsqueeze(0)
    

