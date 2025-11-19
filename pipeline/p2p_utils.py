import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import FluxAttnProcessor2_0

class AttentionControl:
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class EmptyControl(AttentionControl):
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionStore(AttentionControl):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def get_empty_store(self):
        return {}

    def step_callback(self, x_t):
        if self.num_att_layers == -1:
            self.num_att_layers = self.cur_att_layer
        self.cur_att_layer = 0
        self.cur_step += 1
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
        return x_t

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
            self.step_store = self.get_empty_store()

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_{self.cur_att_layer}"
        if key not in self.step_store:
             self.step_store[key] = []
        self.step_store[key].append(attn)
        self.cur_att_layer += 1
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore):
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend = None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.num_steps = num_steps
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.local_blend = local_blend
        self.text_len = 512 # Default Flux T5 max length

    def step_callback(self, x_t):
        return super(AttentionControlEdit, self).step_callback(x_t)

    def replace_cross_attention(self, attn_base, attn_replace):
        # attn_base: (Heads, Seq, Seq)
        # Replace Image->Text attention (Rows T:, Cols :T)
        T = self.text_len
        if attn_base.shape[-1] <= T:
             return attn_base # Safety check
             
        attn_base[:, T:, :T] = attn_replace[:, T:, :T]
        return attn_base

    def replace_self_attention(self, attn_base, attn_replace):
        # attn_base: (Heads, Seq, Seq)
        # Replace Image->Image attention (Rows T:, Cols T:)
        T = self.text_len
        if attn_base.shape[-1] <= T:
             return attn_base
             
        attn_base[:, T:, T:] = attn_replace[:, T:, T:]
        return attn_base

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # attn shape: (B, Heads, Seq, Seq)
        # We assume B = 2 (Source, Target) or B = 1 (if we run sequentially and store/retrieve)
        # For simplicity, let's assume we run with batch_size=2 [Source, Target]
        
        if attn.shape[0] == 2 * self.batch_size: # e.g. CFG? No, Flux usually B=1.
             pass
             
        # If we run with [Source, Target], shape[0] should be 2.
        if attn.shape[0] == 2:
            # Split
            attn_source = attn[0] # (Heads, Seq, Seq)
            attn_target = attn[1] # (Heads, Seq, Seq)
            
            # Cross Attention Replacement
            # Check if current step is within the replacement window
            if self.cur_step < self.cross_replace_steps * self.num_steps:
                 attn_target = self.replace_cross_attention(attn_target, attn_source)
            
            # Self Attention Replacement
            if self.cur_step < self.self_replace_steps * self.num_steps:
                 attn_target = self.replace_self_attention(attn_target, attn_source)
            
            attn[1] = attn_target
            
        self.cur_att_layer += 1
        return attn

class FluxP2PAttnProcessor(FluxAttnProcessor2_0):
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Manual Attention Computation for P2P
        # shape: (B, Heads, Seq, Dim)
        
        # 1. Q @ K.T
        # query: (B, H, S, D)
        # key: (B, H, S, D)
        # attn_weights: (B, H, S, S)
        
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        # 2. Apply Mask
        # attention_mask in Flux is usually None or handled differently? 
        # Flux uses masking in the loop?
        # In `FluxAttnProcessor2_0`, `attention_mask` is passed to `scaled_dot_product_attention`.
        # If it's not None, we need to add it.
        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        # 3. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 4. P2P Injection
        # We call controller to modify attn_weights
        # We assume Unified Attention is "Self Attention" in P2P terms for now, 
        # but we need to distinguish Cross parts.
        
        # The controller should handle the logic based on batch size (Source, Target)
        attn_weights = self.controller(attn_weights, is_cross=False, place_in_unet=self.place_in_unet)

        # 5. Dropout (usually 0)
        # attn_weights = F.dropout(attn_weights, p=0.0, training=self.training)
        
        # 6. Attn @ V
        hidden_states = torch.matmul(attn_weights, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


