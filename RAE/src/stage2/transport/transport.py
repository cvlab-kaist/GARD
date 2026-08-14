import os 
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

from RAE.src.utils.vis_utils import vis_attn_maps
# from utils.vis_utils import vis_attn_maps


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(
    shape, mu, sigma, low=0.0, high=1.0
):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    Works for scalars or tensors mu/sigma/low/high with broadcasting.

    Args:
        shape: output batch shape (e.g., (N,) or (N,M)). Leave () to broadcast to mu.shape.
        mu, sigma: tensors or floats (sigma > 0).
        low, high: truncation bounds in [0,1]. (low can be 0, high can be 1).
        device, dtype: optional overrides.

    Returns:
        Tensor of samples with shape = broadcast(shape, mu.shape, ...)
    """
    mu   = torch.as_tensor(mu)
    sigma= torch.as_tensor(sigma)
    low  = torch.as_tensor(low)
    high = torch.as_tensor(high)

    # Map truncation bounds to logit space; handles 0/1 → ±inf automatically.
    z_low  = torch.logit(low)   # = -inf if low==0
    z_high = torch.logit(high)  # = +inf if high==1

    # Standardize bounds for the base Normal(0,1)
    base = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma

    # Truncated-normal inverse CDF sampling:
    # U ~ Uniform(Φ(alpha), Φ(beta));  Z = mu + sigma * Φ^{-1}(U);  X = sigmoid(Z)
    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)

    # Draw uniforms on the truncated interval
    out_shape = torch.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = torch.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = torch.sigmoid(Z)

    # Numerical safety when low/high are extremely close; clamp back into [low, high].
    return X.clamp(low, high)


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - torch.sum(x ** 2) / 2.
        return torch.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        x0 = torch.randn_like(x1)  
        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":    
            t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal": 
            assert len(dist_options) == 3, "Logit-normal distribution must specify the mean and variance."
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            assert sigma > 0, "Logit-normal distribution must have positive variance."
            t = truncated_logitnormal_sample(
                (x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1
            )
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")
        t = t.to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)
        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        return terms



    def training_losses_mvrm(
        self, 
        model,  
        x1, 
        xcond,
        model_img_size,
        cfg
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint (latent)
        - model_kwargs: additional arguments for the model
        """
        
        
        terms = {}
        
        mvrm_analysis = cfg.mvrm.get('analysis', None)
        
        assert x1.shape == xcond.shape 
        
        b, v, n, d = x1.shape    # b v n+1 d
        
        
        # x1 = x1.view(b*v, c, h, w)
        # xcond = xcond.view(b*v, c, h, w)
        
        ## PHO understanding flow matching sampling
        # import torch 
        # import torchvision 
        # from PIL import Image 
        # from torchvision.utils import save_image 
        # img=Image.open('tmp.jpg')
        # img = torchvision.transforms.ToTensor()(img).unsqueeze(0)   # 1 3 382 2026
        # t, x0, x1 = self.sample(img)
        # t, xt, ut = self.path_sampler.plan(t, x0, x1)
        # save_image(x0, 'img_x0.jpg')
        # save_image(x1, 'img_x1.jpg')
        # save_image(xt, 'img_xt.jpg')
        # save_image(ut, 'img_ut.jpg')
        
        # x1_hat = xt + ut * (1.0 - t)
        # save_image(x1_hat, 'img_x1_hat.jpg')
        
        # x0: pure noise 
        # x1: pure data 
        # xt: noisy data point at t
        # ut: gt velocity at t 
        
        
        
        # CFG training: randomly zero out lq conditioning so the model learns unconditional denoising
        lq_drop_prob = cfg.training.guidance.get('lq_drop', 0.0)
        if lq_drop_prob > 0.0:
            should_drop = torch.rand(b, device=x1.device) < lq_drop_prob  # (b,) bool
            xcond = xcond.clone()
            xcond[should_drop] = 0.0



        # lq2hq training w/ different noise levels
        noise_lvl = cfg.mvrm.get('noise_lvl', None)     # typically 0.3



        if noise_lvl is not None:
            t, x0, x1 = self.sample(x1)     # b v n+1 d

            # lq_latent conditioning method
            # for CFG-dropped samples (xcond==0), skip noise scaling to keep x0 as pure noise
            if cfg.mvrm.lq_latent_cond == 'addition':
                cond_mask = (xcond.abs().sum(dim=(-1, -2, -3), keepdim=True) > 0).float()  # (b,1,1,1): 1=conditioned, 0=dropped
                x0 = x0 * (1.0 - cond_mask * (1.0 - noise_lvl)) + xcond  # conditioned: x0*noise_lvl + xcond | dropped: x0*1.0 + 0

            t, xt, ut = self.path_sampler.plan(t, x0, x1)


            # for lq2hq cfg
            training_lq_cond = cfg.mvrm.get('training_lq_cond', False)
            
            if training_lq_cond:
                print('LQ2HQ: Using double conditioning for training !!')
                xt = xt + xcond
            else:
                print('LQ2HQ: Using single conditioning for training !!')
        
        
        else:
            t, x0, x1 = self.sample(x1)     # b v n+1 d
            t, xt, ut = self.path_sampler.plan(t, x0, x1)

            # lq_latent conditioning method
            if cfg.mvrm.lq_latent_cond == 'addition':
                xt = xt + xcond                          # b v n+1 d            
        
        
        # mvrm forward pass 
        model_output = model(xt, t, model_img_size, analysis=mvrm_analysis) # b v n+1 d
        assert model_output.shape == ut.shape 
        

        # collect mvrm attention map
        if mvrm_analysis.vis_attn_map and len(mvrm_analysis.mvrm_attn_map.extract_idx) != 0:
        
            mvrm_maps = {}
            for layer_idx in mvrm_analysis.mvrm_attn_map.extract_idx:
                print('MVRM ATTENTION MAP EXTRACTION - LAYER ', layer_idx)
                attn_idx, attn_type, attn_map = model.module.blocks[layer_idx].attn.attn_map
                assert layer_idx == attn_idx
                mvrm_maps[('mvrm', attn_idx, attn_type)] = attn_map.mean(dim=1) 

            terms['mvrm_maps'] = mvrm_maps
        
                
        terms['pred'] = model_output
        terms['target_velocity'] = ut 

        
        if self.model_type == ModelType.VELOCITY:  

            terms['loss'] = mean_flat((model_output - ut) ** 2)
            
        else:   
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        return terms



    def training_losses_mvrm_VAE(
        self, 
        model,  
        x1, 
        xcond,
        cfg
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint (latent)
        - model_kwargs: additional arguments for the model
        """
        
        
        assert x1.shape == xcond.shape 
        
        b, v, d, h, w = x1.shape    

        t, x0, x1 = self.sample(x1)     # b v d h w               
        t, xt, ut = self.path_sampler.plan(t, x0, x1)


        # lq_latent conditioning method 
        if cfg.mvrm.lq_latent_cond == 'addition':
            xt = xt + xcond
        elif cfg.mvrm.lq_latent_cond == 'concat':
            xt = torch.concat([xt, xcond], dim=2)   # channel concat


        # mvrm forward pass 
        model_output = model(xt, t)                 # b v 3072 27 36
        assert model_output.shape == xt.shape 


        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:   # t
            terms['loss'] = mean_flat(((model_output - ut) ** 2))   # b
        else:   # f
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        return terms
    


    def get_drift(
        self
    ):
        
        
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        _step_counter = [0]


        def velocity_ode(x, t, model, **model_kwargs):
            
            _step_counter[0] += 1
                                                          
            mvrm_cfg = model_kwargs.pop('mvrm_cfg', None)
            model_img_size = model_kwargs.pop('model_img_size', None)        
            lq_latent = model_kwargs.pop('lq_latent', None)
            analysis = mvrm_cfg.val.get('analysis', None)  
            guidance = mvrm_cfg.val.get('guidance', None)
            mvrm_vis = model_kwargs.pop('mvrm_vis', None) 
            if mvrm_vis is not None:
                mvrm_vis_attn_root = mvrm_vis['mvrm_vis_attn_root']
                scene = mvrm_vis['scene']
                num_view = mvrm_vis['num_view']
                model_img_size = mvrm_vis['model_img_size']
                lq_imgs = mvrm_vis['lq_imgs'] 
                ref_b_idx = mvrm_vis['ref_b_idx']

            lq_cond_sampling = mvrm_cfg.val.get('lq_cond_sampling', True)
            if lq_cond_sampling:          
                # lq condition
                if mvrm_cfg.lq_latent_cond == 'addition':
                    x_in = x + lq_latent
                    
            else:
                x_in = x

            if guidance.use_cfg and guidance.cfg_scale > 1.0:
                cfg_scale = guidance.cfg_scale
                x_uncond = x        # unconditional: no lq_latent (matches CFG dropout during training)
                x_cond   = x_in     # conditional: lq_latent fused
                v_uncond = model(x_uncond, t, model_img_size, analysis)
                v_cond   = model(x_cond,   t, model_img_size, analysis)
                model_output = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                model_output = model(x_in, t, model_img_size, analysis)
            
            if analysis is not None:
                if analysis.vis_attn_map and len(analysis.mvrm_attn_map.extract_idx) != 0:
                    # MVRM attn_map extraction — only at every 15th step
                    # if _step_counter[0] % 15 == 0:
                    if _step_counter[0] == 30:
                        mvrm_maps = {}
                        print(f"MVRM ATTN_MAP EXTRACTION (step {_step_counter[0]})")
                        for layer_idx in analysis.mvrm_attn_map.extract_idx:
                            attn_idx, attn_type, attn_map = model.__self__.blocks[layer_idx].attn.attn_map
                            assert layer_idx == attn_idx
                            mvrm_maps[('mvrm', attn_idx, attn_type)] = attn_map.mean(dim=1)  # 1 head num_view*(n+1) num_view*(n+1)  -> 1 num_view*(n+1) num_view*(n+1)

                        vis_attn_maps(
                            vis_save_root=mvrm_vis_attn_root,
                            scene=scene,
                            num_view=num_view,
                            model_img_size=model_img_size,
                            imgs_list=[('lq_imgs', lq_imgs)],
                            attn_maps_list=[('mvrm_maps', mvrm_maps)],
                            ref_b_idx=ref_b_idx,
                            mvrm_step=_step_counter[0]
                        )
            
            return model_output

        if self.model_type == ModelType.NOISE:  # f
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:    # f
            drift_fn = score_ode
        else:   # t
            drift_fn = velocity_ode #
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            # assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        def reset_step_counter():
            _step_counter[0] = 0

        body_fn.reset_step_counter = reset_step_counter

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def sde_diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs) - sde_diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
            return drift_mean
    

        return sde_drift_fn, sde_diffusion_fn
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """ 
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, torch.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )
        
        return _ode.sample