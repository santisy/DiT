import torch
import torch.distributed as dist
import numpy as np
import warnings
import ot

def get_map(x0, x1, normalize_cost=False):
    """Compute the OT plan and return the plan along with aggregated x0 and x1 in their original shapes.

    Parameters
    ----------
    x0 : Tensor, shape (local_bs, *dim)
        Represents the source minibatch on the local process.
    x1 : Tensor, shape (local_bs, *dim)
        Represents the target minibatch on the local process.

    Returns
    -------
    pi_local : numpy array, shape (local_bs, total_bs)
        Represents the OT plan between local source minibatch and aggregated target minibatch.
    x0_all : Tensor, shape (total_bs, *dim)
        Aggregated source data from all processes.
    x1_all : Tensor, shape (total_bs, *dim)
        Aggregated target data from all processes.
    """

    # Check if running in distributed mode
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        world_size = dist.get_world_size()
    else:
        world_size = 1

    # Record original shapes
    x0_shape = x0.shape  # Shape: (local_bs, *dim)
    x1_shape = x1.shape

    # Flatten tensors if necessary
    if x0.dim() > 2:
        x0_flat = x0.view(x0.size(0), -1)  # Shape: (local_bs, num_features)
    else:
        x0_flat = x0
    if x1.dim() > 2:
        x1_flat = x1.view(x1.size(0), -1)
    else:
        x1_flat = x1

    # Gather flattened tensors from all processes
    if is_distributed:
        # Prepare lists to gather tensors from all processes
        x0_flat_list = [torch.zeros_like(x0_flat) for _ in range(world_size)]
        x1_flat_list = [torch.zeros_like(x1_flat) for _ in range(world_size)]

        # All-gather flattened tensors
        dist.all_gather(x0_flat_list, x0_flat)
        dist.all_gather(x1_flat_list, x1_flat)

        # Concatenate the gathered tensors
        x0_all_flat = torch.cat(x0_flat_list, dim=0)  # Shape: (total_bs, num_features)
        x1_all_flat = torch.cat(x1_flat_list, dim=0)
    else:
        x0_all_flat = x0_flat
        x1_all_flat = x1_flat

    # Uniform weights for OT
    total_bs_x0 = x0_all_flat.size(0)
    total_bs_x1 = x1_all_flat.size(0)
    a = np.ones(total_bs_x0) / total_bs_x0
    b = np.ones(total_bs_x1) / total_bs_x1

    # Compute cost matrix
    M = torch.cdist(x0_all_flat, x1_all_flat) ** 2
    if normalize_cost:
        M = M / M.max()

    # Compute the OT plan
    pi_all = ot.emd(a, b, M.detach().cpu().numpy(), numThreads=1)

    # Check numerical errors
    if not np.all(np.isfinite(pi_all)):
        print("ERROR: pi is not finite")
        print(pi_all)
        print("Cost mean, max", M.mean(), M.max())
        print(x0_all_flat, x1_all_flat)
    if np.abs(pi_all.sum()) < 1e-8:
        warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
        pi_all = np.ones_like(pi_all) / pi_all.size


    # Reconstruct x0_all and x1_all from x0_all_flat and x1_all_flat
    if x0.dim() > 2:
        # Reshape x0_all_flat back to (total_bs_x0, *dim)
        x0_all = x0_all_flat.view(-1, *x0_shape[1:])
    else:
        x0_all = x0_all_flat
    if x1.dim() > 2:
        x1_all = x1_all_flat.view(-1, *x1_shape[1:])
    else:
        x1_all = x1_all_flat

    return pi_all, x0_all, x1_all


def sample_map(pi, batch_size, replace=True):
    """
    Draw source and target sample indices from the OT plan pi.

    Parameters
    ----------
    pi : numpy array, shape (n_source, n_target)
        The OT plan between the source and target samples.
    batch_size : int
        Number of samples to draw.
    replace : bool
        Whether to sample with replacement.

    Returns
    -------
    i_s : numpy array, shape (batch_size,)
        Indices of source samples.
    i_t : numpy array, shape (batch_size,)
        Indices of target samples.
    """
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(
        pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
    )
    i_s, i_t = np.divmod(choices, pi.shape[1])
    return i_s, i_t

def sample_plan(x0, x1, conditions, replace=True):
    """
    Compute the OT plan and sample source and target samples along with reordering conditions.

    Parameters
    ----------
    x0 : Tensor, shape (local_bs, *dim)
        Source minibatch.
    x1 : Tensor, shape (local_bs, *dim)
        Target minibatch.
    conditions : dict
        Dictionary of conditions associated with x1.
    replace : bool
        Whether to sample with replacement.

    Returns
    -------
    x0_sampled : Tensor
        Sampled source minibatch.
    x1_sampled : Tensor
        Sampled target minibatch.
    conditions_sampled : dict
        Conditions reordered according to the sampled x1 indices.
    """

    local_bs = x0.size(0)
    # Check if running in distributed mode
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    pi_all, x0_all, x1_all = get_map(x0, x1)
    total_bs = x0_all.shape[0]

    # Sample indices from the OT plan
    i_s_all, i_t_all = sample_map(pi_all, total_bs, replace=replace)

    # Extract the indices for the current process
    start_idx = rank * local_bs
    end_idx = start_idx + local_bs
    i_s_local = i_s_all[start_idx:end_idx]
    i_t_local = i_t_all[start_idx:end_idx]

    # Extract sampled x0 and x1
    x0_sampled = x0_all[i_s_local]
    x1_sampled = x1_all[i_t_local]

    # Reorder conditions associated with x1
    conditions_sampled = {}
    for key, value in conditions.items():
        if isinstance(conditions, torch.Tensor):
            # Assume value is a Tensor or array with shape matching x1_all
            # We need to gather conditions from all processes if distributed
            if dist.is_available() and dist.is_initialized():
                # Gather conditions from all processes
                value_list = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
                dist.all_gather(value_list, value)
                value_all = torch.cat(value_list, dim=0)
            else:
                value_all = value

            # Reorder the condition values according to i_t
            conditions_sampled[key] = value_all[i_t_local]
        else:
            conditions_sampled[key] = None


    return x0_sampled, x1_sampled, conditions_sampled



if __name__ == "__main__":
    x0 = torch.randn(4, 256, 4).cuda()
    x1 = torch.randn(4, 256, 4).cuda()
    conditions = {"a": torch.randn(4).cuda(),
                  "y": torch.randn(4, 256, 4).cuda(),
                  "labels": torch.randn(4, 128).cuda()} 
    x0_sampled, x1_sampled, conditions_sampled = sample_plan(x0, x1, conditions)
    print("Checking")
    print(x0_sampled.shape) 
    print(x1_sampled.shape)
    print(conditions_sampled["a"].shape)
