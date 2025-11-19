import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import abc
import numpy as np
from typing import Union, Tuple, List, Dict, Optional

# =================================================================================
# Sequence Alignment Logic (adapted from standard Prompt-to-Prompt)
# =================================================================================

def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Finds the indices of tokens corresponding to a specific word in the text.
    Adapted for T5/CLIP tokenizers.
    """
    split = text.split(" ")
    word = split[word_place]
    
    out = []
    # This is a heuristic. A more robust way involves encoding prefixes.
    # For T5/Flux, we assume standard behavior or we might need adjustment.
    # We'll use the differential encoding approach which is safer.
    
    # Encode full text
    tokens_full = tokenizer.encode(text)
    
    # Encode text up to the word (exclusive)
    if word_place > 0:
        prefix = " ".join(split[:word_place])
        tokens_prefix = tokenizer.encode(prefix)
        # T5 might add EOS/BOS, we need to be careful. 
        # T5 usually adds an EOS at the end.
        if tokenizer.name_or_path and 't5' in tokenizer.name_or_path:
            # Remove EOS from prefix if present
            if tokens_prefix[-1] == 1: # 1 is usually EOS for T5
                tokens_prefix = tokens_prefix[:-1]
    else:
        tokens_prefix = []
        
    # Encode text up to the word (inclusive)
    prefix_with_word = " ".join(split[:word_place+1])
    tokens_prefix_word = tokenizer.encode(prefix_with_word)
    if tokenizer.name_or_path and 't5' in tokenizer.name_or_path:
         if tokens_prefix_word[-1] == 1:
            tokens_prefix_word = tokens_prefix_word[:-1]

    # The tokens for the word are the difference
    # This assumes monotonicity and deterministic tokenization
    len_prefix = len(tokens_prefix)
    len_word = len(tokens_prefix_word)
    
    # Sometimes tokenizers are tricky (e.g. space handling).
    # We fallback to matching sub-sequence if differential fails or looks weird.
    if len_word > len_prefix:
        out = list(range(len_prefix, len_word))
    else:
        # Fallback: search for the word tokens in the full sequence
        # This is less precise if the word appears multiple times.
        # For now, let's assume the differential method works for most cases.
        pass

    return out

def get_replacement_mapper(prompts, tokenizer, max_len=512):
    x_seq = prompts[0]
    y_seq = prompts[1]
    words_x = x_seq.split(" ")
    words_y = y_seq.split(" ")
    
    # Naive alignment: works if we just replaced a word with another word or same length.
    # For "squirrel" -> "lion", lengths might be same (1 word -> 1 word).
    # If structure changes, we need Needleman-Wunsch.
    
    # We'll implement a basic version that assumes similar sentence structure 
    # and identified replaced words.
    
    # Find indices of words that differ
    # We assume words_x and words_y are roughly aligned by index, 
    # or we can use a simple diff.
    
    # Note: This simple mapper assumes 1-to-1 word mapping for unchanged words.
    # If the user adds words, this breaks. 
    # PROPER ALIGNMENT IS COMPLEX. 
    # We will implement the one from the paper/code provided in search results if possible.
    
    # Since we don't have the full seq_aligner module, we implement a simplified 
    # heuristic for replacement:
    # 1. Identify diffs.
    # 2. Map unchanged words 1-to-1.
    # 3. Replaced words map to each other.
    
    mapper = np.zeros((max_len, max_len))
    
    # If lengths differ significantly, this simple heuristic might fail, 
    # but it's better than nothing.
    
    min_len = min(len(words_x), len(words_y))
    
    i_x = 0 # token index x
    i_y = 0 # token index y
    
    # Helper to get tokens for a word index
    def get_tokens(txt, ind):
        return get_word_inds(txt, ind, tokenizer)
        
    # Iterate through words
    # This loop is a simplification.
    for i in range(min_len):
        inds_x = get_tokens(x_seq, i)
        inds_y = get_tokens(y_seq, i)
        
        # If words are same, 1-to-1 map (or N-to-N diagonal)
        if words_x[i] == words_y[i]:
             # Map diagonals
             for ix, iy in zip(inds_x, inds_y):
                 if ix < max_len and iy < max_len:
                    mapper[ix, iy] = 1.0
        else:
             # Words differ (replacement)
             # Map all x tokens to all y tokens (distribute attention)
             if len(inds_y) > 0:
                 val = 1.0 / len(inds_y)
                 for ix in inds_x:
                     for iy in inds_y:
                         if ix < max_len and iy < max_len:
                             mapper[ix, iy] = val
                             
    return torch.from_numpy(mapper).float()

def get_refinement_mapper(prompts, tokenizer, max_len=512):
    # Refinement mapper: maps x to y.
    # For refinement, we want to preserve x's attention for y.
    # This is similar to replacement but we also return alphas for blending.
    
    mapper = get_replacement_mapper(prompts, tokenizer, max_len)
    
    # For now, alphas are all 1.0 (fully use mapper)
    # In advanced P2P, alphas control how much we use the old attention.
    alphas = torch.ones(max_len) 
    
    return mapper, alphas

# =================================================================================
# Attention Controllers
# =================================================================================

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class EmptyControl(AttentionControl):
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    def reset(self):
        pass

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
        # Store a clone to avoid modification issues
        self.step_store[key].append(attn.detach().cpu()) 
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
                 local_blend = None,
                 tokenizer = None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.local_blend = local_blend
        self.text_len = 512 # Default Flux T5 max length, should be passed or inferred
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.num_steps = num_steps

    def step_callback(self, x_t):
        return super(AttentionControlEdit, self).step_callback(x_t)

    def replace_cross_attention(self, attn_base, attn_replace):
        # attn_base: (Heads, Seq, Seq) 
        # attn_replace: (Heads, Seq, Seq) - Source
        # We want to inject Source Attention into Base (Target)
        
        # In Single Stream Flux:
        # Grid is (Text + Image) x (Text + Image)
        # Cross Attention part is Image -> Text.
        # Rows: Image tokens (Indices T:). Cols: Text tokens (Indices :T).
        
        T = self.text_len
        if attn_base.shape[-1] <= T:
             return attn_base 
             
        # Replace Image -> Text attention
        attn_base[:, T:, :T] = attn_replace[:, T:, :T]
        return attn_base

    def replace_self_attention(self, attn_base, attn_replace):
        # Replace Image -> Image attention
        # Rows T:, Cols T:
        T = self.text_len
        if attn_base.shape[-1] <= T:
             return attn_base
             
        attn_base[:, T:, T:] = attn_replace[:, T:, T:]
        return attn_base

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # attn shape: (B, Heads, Seq, Seq)
        # Assumes B=2 (Source, Target)
        
        if attn.shape[0] == 2:
            attn_source = attn[0]
            attn_target = attn[1]
            
            # Determine if we should replace
            cross_inject = self.cur_step < self.cross_replace_steps * self.num_steps
            self_inject = self.cur_step < self.self_replace_steps * self.num_steps
            
            if cross_inject:
                 attn_target = self.replace_cross_attention(attn_target, attn_source)
            
            if self_inject:
                 attn_target = self.replace_self_attention(attn_target, attn_source)
            
            attn[1] = attn_target
            
        self.cur_att_layer += 1
        return attn

class AttentionReplace(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, 
                 local_blend=None, tokenizer=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        if tokenizer is not None:
            self.mapper = get_replacement_mapper(prompts, tokenizer, max_len=self.text_len).to(torch.device('cpu')) # Device will be adjusted in forward if needed
        else:
            self.mapper = None

    def replace_cross_attention(self, attn_base, attn_replace):
        # attn_base: Target
        # attn_replace: Source
        
        # Apply mapper to Source Cross Attention (Image -> Text)
        # Source: Image(I) -> Text(Source_T)
        # Target: Image(I) -> Text(Target_T)
        # We want Image(I) -> Text(Target_T) to look like Image(I) -> Text(Source_T) * Mapper
        
        T = self.text_len
        if attn_base.shape[-1] <= T or self.mapper is None:
            return super().replace_cross_attention(attn_base, attn_replace)
            
        # Extract Image->Text block
        # attn_replace (Source) slice: [Heads, Img_Seq, Text_Seq]
        source_cross = attn_replace[:, T:, :T]
        
        # Mapper: [Text_Seq_Source, Text_Seq_Target] ? 
        # Standard mapper is [Seq_Base, Seq_Replace].
        # Here Base=Source, Replace=Target?
        # In code: mapper maps indices of Source to Target.
        
        # einsum: 
        # h = heads
        # p = image pixels (queries)
        # w = source text words (keys)
        # n = target text words
        # source_cross: (h, p, w)
        # mapper: (w, n)
        # result: (h, p, n) -> injected into target
        
        device = attn_base.device
        self.mapper = self.mapper.to(device)
        
        # Inject
        target_cross_new = torch.einsum('hpw,wn->hpn', source_cross, self.mapper)
        
        # Normalize if needed? Usually softmax output sums to 1. 
        # If mapper is not 1-to-1, we might break sum=1.
        # But standard P2P does this.
        
        attn_base[:, T:, :T] = target_cross_new
        return attn_base

class AttentionRefine(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, 
                 local_blend=None, tokenizer=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        if tokenizer is not None:
            self.mapper, self.alphas = get_refinement_mapper(prompts, tokenizer, max_len=self.text_len)
            self.mapper = self.mapper.to(torch.device('cpu'))
            self.alphas = self.alphas.to(torch.device('cpu'))
        else:
            self.mapper = None
            self.alphas = None

    def replace_cross_attention(self, attn_base, attn_replace):
        # Refine: Blend Base (Target) and Mapped Source
        T = self.text_len
        if attn_base.shape[-1] <= T or self.mapper is None:
            return super().replace_cross_attention(attn_base, attn_replace)
            
        source_cross = attn_replace[:, T:, :T] # Source
        target_cross = attn_base[:, T:, :T]    # Target (Original generation for target prompt)
        
        device = attn_base.device
        self.mapper = self.mapper.to(device)
        self.alphas = self.alphas.to(device)
        
        # Map Source to Target structure
        mapped_source = torch.einsum('hpw,wn->hpn', source_cross, self.mapper)
        
        # Blend: mapped_source * alpha + target_cross * (1 - alpha)
        # alphas shape: (n,) -> broadcast to (h, p, n)
        
        target_cross_new = mapped_source * self.alphas + target_cross * (1 - self.alphas)
        
        attn_base[:, T:, :T] = target_cross_new
        return attn_base

class AttentionReweight(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, 
                 equalizer, local_blend=None, controller=None, tokenizer=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        self.equalizer = equalizer # shape (words,)
        self.prev_controller = controller

    def replace_cross_attention(self, attn_base, attn_replace):
        # First apply previous controller if exists (e.g. Refine)
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, attn_replace)
            
        # Then Reweight
        # attn_base is Target (Image -> Text)
        T = self.text_len
        target_cross = attn_base[:, T:, :T]
        
        device = attn_base.device
        self.equalizer = self.equalizer.to(device)
        
        # equalizer: (T,)
        # target_cross: (h, p, T)
        target_cross = target_cross * self.equalizer
        
        attn_base[:, T:, :T] = target_cross
        return attn_base


# =================================================================================
# Flux Attention Processor
# =================================================================================

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
        # BUT FluxSingleTransformerBlock concatenates [txt, img] before calling attn.
        # So hidden_states ALREADY contains both.
        
        if encoder_hidden_states is not None:
            # This path handles cases where they are separate inputs (Double Blocks)
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
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        # 2. Apply Mask
        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        # 3. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 4. P2P Injection
        # The controller will handle Cross/Self distinctions internally
        # by looking at indices (0:512 is Text, 512: is Image for Single Blocks)
        
        attn_weights = self.controller(attn_weights, is_cross=False, place_in_unet=self.place_in_unet)

        # 5. Attn @ V
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
