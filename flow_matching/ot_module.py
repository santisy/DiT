import torch
import torch.distributed as dist

import numpy as np
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from flow_matching.ot_utils import sample_plan
from flow_matching.ot_utils import sample_map
from flow_matching.ot_utils import get_map


class OT:
    def __init__(self,
                 noise_conditioning_level=0.05):

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0) 
        self._nc_level = noise_conditioning_level
    
    def _noise_conditioning(self, xc):
        """
        Apply noise conditioning to xc using a new OT plan between x0_noise and xc.

        Parameters
        ----------
        xc : Tensor
            The condition tensor associated with x1 (already reordered).

        Returns
        -------
        xc_noisy : Tensor
            Noise-conditioned xc.
        """
        # Step 1: Generate random noise x0_noise
        x0_noise = torch.randn_like(xc)
        
        # Step 2: Perform OT between x0_noise and xc (without reordering xc)
        x0_noise_reordered = self._reorder_noise_to_match_xc(x0_noise, xc)
        
        # Step 3: Sample t and epsilon
        batch_size = xc.size(0)
        # NOTE: The noise level is constrained
        t = (torch.rand(batch_size, device=xc.device) * self._nc_level +
             (1 - self._nc_level))
        epsilon = torch.randn_like(xc)
        
        # Step 4: Use sample_xt to generate xc_noisy
        xc_noisy = self.FM.sample_xt(x0_noise_reordered, xc, t, epsilon)
        
        return xc_noisy

    def _reorder_noise_to_match_xc(self, x0_noise, xc):
        """
        Reorder x0_noise to align with xc using an OT plan.

        Parameters
        ----------
        x0_noise : Tensor
            Random noise tensor.
        xc : Tensor
            Condition tensor (already reordered).

        Returns
        -------
        x0_noise_reordered : Tensor
            Reordered noise tensor aligned with xc.
        """
        local_bs = x0_noise.shape[0]

        # Compute OT plan between x0_noise and xc
        pi_all, x0_noise_all, xc_all = get_map(x0_noise, xc)
        
        # Sample indices from the OT plan
        i_s, i_t = sample_map(pi_all, batch_size=xc_all.shape[0], replace=True)
        
        # Reorder x0_noise according to i_t to align with xc
        # Ensure that x0_noise_reordered[i_t[k]] = x0_noise_all[i_s[k]]
        x0_noise_reordered = torch.zeros_like(xc_all)
        x0_noise_reordered[i_t] = x0_noise_all[i_s]
        
        # Handle any positions not assigned due to possible duplicates or missing indices
        unassigned_indices = [i for i in range(xc_all.shape[0]) if i not in i_t]
        for idx in unassigned_indices:
            x0_noise_reordered[idx] = x0_noise_all[np.random.choice(x0_noise_all.shape[0])]
        
        # Return the local portion of x0_noise_reordered
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            start_idx = rank * local_bs
            end_idx = start_idx + local_bs
            x0_noise_reordered_local = x0_noise_reordered[start_idx:end_idx]
        else:
            x0_noise_reordered_local = x0_noise_reordered
        
        return x0_noise_reordered_local
    

    def __call__(self, model, x1, model_kwargs):

        x0 = torch.randn_like(x1)
        x0, x1, model_kwargs = sample_plan(x0, x1, model_kwargs)

        # OT noise conditioning
        model_kwargs["xc"] = self._noise_conditioning(model_kwargs["xc"])

        t, xt, ut = super(ExactOptimalTransportConditionalFlowMatcher, self.FM
                          ).sample_location_and_conditional_flow(x0, x1)
        vt = model(xt, t, **model_kwargs)
        loss = torch.mean((vt - ut) **2)

        return loss

if __name__ == "__main__":
    model_kwargs = {"xc": torch.randn(4, 256, 4).cuda()}    
    x1 = torch.randn(4, 256, 4).cuda()
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, xt, t, **modek_kwargs):
            return xt

    net = Net().cuda()
    ot = OT()
    loss = ot(net, x1, model_kwargs)
    print(loss.shape)
