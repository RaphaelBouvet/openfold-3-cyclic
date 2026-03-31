# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from openfold3.core.utils.tensor_utils import binned_one_hot


def _compute_cyclic_offset(
    offset: torch.Tensor,
    asym_id: torch.Tensor,
    is_cyclic: torch.Tensor,
) -> torch.Tensor:
    """Replace linear offsets with shortest-path cyclic offsets for cyclic chains.
 
    For a cyclic chain of length N, the relative position between residues i and j
    is taken as the shortest path around the ring:
 
        offset_cyclic(i, j) = ((i - j) + N/2) mod N  -  N/2
 
    This maps offsets into the range [-(N//2), N - N//2 - 1], preserving sign
    while "wrapping" distances that exceed half the ring length.
 
    Only pairs where BOTH tokens belong to the same cyclic chain are modified;
    all other pairs keep their original linear offsets.
 
    Args:
        offset:
            [*, N_token, N_token] Raw linear residue offsets  (i - j).
        asym_id:
            [*, N_token] Chain identity index (1-based integers).
        is_cyclic:
            [*, N_token] Boolean mask: True for tokens belonging to a cyclic chain.
 
    Returns:
        [*, N_token, N_token] Offsets with cyclic correction applied where relevant.
    """
    same_chain = asym_id[..., None] == asym_id[..., None, :]  # [*, N, N]
    # A pair is subject to cyclic encoding only if both tokens are cyclic AND
    # they are in the same chain.
    is_cyclic_pair = same_chain & is_cyclic[..., None] & is_cyclic[..., None, :]  # [*, N, N]
 
    if not is_cyclic_pair.any():
        return offset
 
    # Compute per-chain length for the cyclic correction.
    # We build a [*, N_token] tensor where each token holds the length of its chain.
    # For non-cyclic chains the value is irrelevant (pairs will not be modified).
    chain_lengths = torch.zeros_like(asym_id)  # [*, N_token]
    unique_chain_ids = asym_id.unique()
    for cid in unique_chain_ids:
        mask = asym_id == cid  # [*, N_token]
        length = mask.sum(dim=-1, keepdim=True)  # [*, 1]
        chain_lengths = chain_lengths + mask * length  # broadcast fill
 
    # Expand chain lengths to pair shape: take the length of token i's chain.
    chain_len_pair = chain_lengths[..., None].expand_as(offset)  # [*, N, N]
 
    # Cyclic shortest-path offset: ((offset + N/2) % N) - N/2
    # Using float for modulo stability, then rounding back to integer.
    N = chain_len_pair.float()
    offset_f = offset.float()
    cyclic_offset = torch.remainder(offset_f + N / 2.0, N) - N / 2.0
    cyclic_offset = cyclic_offset.to(offset.dtype)
 
    return torch.where(is_cyclic_pair, cyclic_offset, offset)


def relpos_complex(
    batch: dict, max_relative_idx: int, max_relative_chain: int
) -> torch.Tensor:
    """
    Args:
        batch:
            Input feature dictionary
        max_relative_idx:
            Maximum relative position and token indices clipped
        max_relative_chain:
            Maximum relative chain indices clipped

    Returns:
        [*, N_token, N_token, C_z] Relative position embedding
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    is_cyclic: torch.Tensor | None = batch.get("is_cyclic")

    def relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int
    ) -> torch.Tensor:
        """
        Args:
            pos:
                [*, N_token] Token index
            condition:
                [*, N_token, N_token] Condition for clipping
            rel_clip_idx:
                Max idx for clipping (max_relative_idx or max_relative_chain)
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        )
        rel_pos = binned_one_hot(
            final_offset,
            boundaries,
        )

        return rel_pos

    def relpos_cyclic(
        pos: torch.Tensor,
        condition: torch.BoolTensor,
        rel_clip_idx: int,
        is_cyclic: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> torch.Tensor:
        """Like relpos() but applies cyclic shortest-path offset for cyclic chains.
 
        Args:
            pos:
                [*, N_token] Residue index
            condition:
                [*, N_token, N_token] same_chain mask
            rel_clip_idx:
                Max idx for clipping
            is_cyclic:
                [*, N_token] Boolean mask for cyclic chain tokens
            asym_id:
                [*, N_token] Chain identity index
 
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
 
        # Apply cyclic shortest-path correction before clipping.
        is_cyclic_bool = is_cyclic.bool()
        offset = _compute_cyclic_offset(offset, asym_id, is_cyclic_bool)
 
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        )
        rel_pos = binned_one_hot(final_offset, boundaries)
        return rel_pos
    
    if is_cyclic is not None:
        rel_pos = relpos_cyclic(
            pos=res_idx,
            condition=same_chain,
            rel_clip_idx=max_relative_idx,
            is_cyclic=is_cyclic,
            asym_id=asym_id,
        )
    else:
        rel_pos = relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)

    rel_token = relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

    rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)

    return rel_feat
