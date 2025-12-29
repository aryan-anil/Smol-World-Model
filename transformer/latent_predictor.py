import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionEncoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                       dim_feedforward=intermediate_size, batch_first=True,
                                       dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FrameEncoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                       dim_feedforward=intermediate_size, batch_first=True,
                                       dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # Return full sequence for cross-attention

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        cross_attn_output, _ = self.cross_attn(x, context, context, need_weights=False)
        x = self.norm1(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class LatentPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, intermediate_size=1536, num_layers=8,
                 num_heads=8, num_actions=5, grid_size=(10, 10), num_ghosts=1, 
                 action_layers=4, frame_encoder_layers=4, num_reset_tokens=2, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size[0] * grid_size[1]
        self.num_ghosts = num_ghosts
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.mask_token_id = vocab_size
        self.pad_token_id = vocab_size + 1
        
        self.embed_tokens = nn.Embedding(vocab_size + 2, hidden_size)
        self.action_embeddings_user = nn.Embedding(num_actions, hidden_size)
        self.ghost_action_embeddings = nn.ModuleList([
            nn.Embedding(num_actions, hidden_size) for _ in range(num_ghosts)
        ])
        self.reset_embeddings = nn.Embedding(num_reset_tokens, hidden_size)
        
        # Positional embeddings
        self.row_pos_embeddings = nn.Parameter(torch.randn(1, grid_size[0], hidden_size))
        self.col_pos_embeddings = nn.Parameter(torch.randn(1, grid_size[1], hidden_size))
        self.next_row_pos_embeddings = nn.Parameter(torch.randn(1, grid_size[0], hidden_size))
        self.next_col_pos_embeddings = nn.Parameter(torch.randn(1, grid_size[1], hidden_size))
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Encoders and Layers
        self.action_encoder = ActionEncoder(hidden_size, intermediate_size, action_layers, num_heads, dropout)
        self.prev_frame_encoder = FrameEncoder(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_layers=frame_encoder_layers, num_heads=num_heads, dropout=dropout
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SelfAttentionBlock(hidden_size, intermediate_size, num_heads, dropout))
            self.layers.append(CrossAttentionBlock(hidden_size, intermediate_size, num_heads, dropout))
        self.layers.append(SelfAttentionBlock(hidden_size, intermediate_size, num_heads, dropout))
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=True)

    def _get_positional_encoding(self, row_embeds, col_embeds):
        """Helper function to correctly create 2D positional encoding."""
        H, W = self.grid_size
        D = self.hidden_size
        
        row_pos = row_embeds.unsqueeze(2) # Shape: [1, H, 1, D]
        col_pos = col_embeds.unsqueeze(1) # Shape: [1, 1, W, D]
        
        pos_encoding = (row_pos + col_pos).view(1, H * W, D)
        return pos_encoding

    def _get_contexts(self, current_frame_indices, user_actions, ghost_actions, reset_tokens):
        B = current_frame_indices.shape[0]
        
        # 1. Encode the previous frame
        current_frame_tokens = current_frame_indices.view(B, -1)
        current_frame_embeds = self.embed_tokens(current_frame_tokens)
        
        # --- FIX #1: Correctly apply positional encoding to the input frame ---
        pos_encoding = self._get_positional_encoding(self.row_pos_embeddings, self.col_pos_embeddings)
        current_frame_with_pos = self.dropout(current_frame_embeds + pos_encoding)
        
        encoded_prev_frame = self.prev_frame_encoder(current_frame_with_pos)

        # 2. Encode all actions
        user_action_embed = self.action_embeddings_user(user_actions).unsqueeze(1)
        ghost_action_embeds = [
            self.ghost_action_embeddings[i](ghost_actions[:, i]).unsqueeze(1) 
            for i in range(self.num_ghosts)
        ]
        reset_embed = self.reset_embeddings(reset_tokens).unsqueeze(1)
        action_sequence = torch.cat([user_action_embed] + ghost_action_embeds + [reset_embed], dim=1)
        action_sequence = self.dropout(action_sequence)
        encoded_actions = self.action_encoder(action_sequence)
        
        return [encoded_prev_frame, encoded_actions]

    def forward(self, current_frame_indices, user_actions, ghost_actions, reset_tokens, target):
        B = current_frame_indices.shape[0]
        device = current_frame_indices.device
        
        contexts = self._get_contexts(current_frame_indices, user_actions, ghost_actions, reset_tokens)
        target_tokens = target.view(B, -1)
        
        initial_input = torch.full((B, self.seq_len), self.mask_token_id, dtype=torch.long, device=device)
        x = self.embed_tokens(initial_input)

        # --- FIX #2: Cleaned up and corrected output positional encoding ---
        pos_encoding = self._get_positional_encoding(self.next_row_pos_embeddings, self.next_col_pos_embeddings)
        x = self.dropout(x + pos_encoding)
        
        # Pass through alternating layers
        context_idx = 0
        for layer in self.layers:
            if isinstance(layer, SelfAttentionBlock):
                x = layer(x)
            elif isinstance(layer, CrossAttentionBlock):
                x = layer(x, contexts[context_idx % len(contexts)])
                context_idx += 1
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_tokens.view(-1),
            reduction='mean'
        )
        
        return {"pred": torch.argmax(logits, dim=-1), "loss": loss}

    @torch.no_grad()
    def generate(self, current_frame_indices, user_actions, ghost_actions, reset_tokens,
                 num_iterations=12, temperature=1.0, confidence_threshold=0.9):
        B = current_frame_indices.shape[0]
        device = current_frame_indices.device

        contexts = self._get_contexts(current_frame_indices, user_actions, ghost_actions, reset_tokens)
        predicted_tokens = torch.full((B, self.seq_len), self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones(B, self.seq_len, dtype=torch.bool, device=device)

        # --- FIX #3: Get output positional encoding once outside the loop ---
        pos_encoding = self._get_positional_encoding(self.next_row_pos_embeddings, self.next_col_pos_embeddings)

        for iteration in range(num_iterations):
            x = self.embed_tokens(predicted_tokens) + pos_encoding

            # Forward pass through layers
            context_idx = 0
            for layer in self.layers:
                if isinstance(layer, SelfAttentionBlock):
                    x = layer(x)
                elif isinstance(layer, CrossAttentionBlock):
                    x = layer(x, contexts[context_idx % len(contexts)])
                    context_idx += 1

            x = self.norm(x)
            logits = self.lm_head(x)
            
            # Decoding logic... (This part was okay)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

            tokens_to_keep = torch.zeros_like(predictions, dtype=torch.bool)
            if iteration == num_iterations - 1:
                tokens_to_keep = mask
            else:
                target_num_masked = int(self.seq_len * (1 - (iteration + 1) / num_iterations))
                masked_confidences = confidences.clone()
                masked_confidences[~mask] = -1

                for b in range(B):
                    num_currently_masked = mask[b].sum().item()
                    if num_currently_masked > target_num_masked:
                        num_to_reveal = num_currently_masked - target_num_masked
                        topk_conf, topk_idx = torch.topk(masked_confidences[b], k=min(num_to_reveal, num_currently_masked))
                        valid_indices = topk_idx[topk_conf >= confidence_threshold]
                        tokens_to_keep[b, valid_indices] = True
            
            predicted_tokens[tokens_to_keep & mask] = predictions[tokens_to_keep & mask]
            mask = mask & ~tokens_to_keep
            if not mask.any():
                break

        return predicted_tokens.view(B, self.grid_size[0], self.grid_size[1])

    @torch.no_grad()
    def generate_with_schedule(self, current_frame_indices, user_actions, ghost_actions, 
                               reset_tokens, schedule='cosine', num_iterations=2, 
                               temperature=1.0):
        """
        Generate with different masking schedules.
        
        Args:
            schedule: 'linear', 'cosine', or 'square' - determines how quickly tokens are revealed
        """
        B = current_frame_indices.shape[0]
        device = current_frame_indices.device
        
        # Encode contexts once
        contexts = self._get_contexts(current_frame_indices, user_actions, ghost_actions, reset_tokens)
        
        # Start with all tokens masked
        predicted_tokens = torch.full((B, self.seq_len), self.mask_token_id, 
                                      dtype=torch.long, device=device)
        mask = torch.ones(B, self.seq_len, dtype=torch.bool, device=device)
        
        # --- FIX #1: Calculate positional encoding ONCE before the loop ---
        row_pos = self.next_row_pos_embeddings.unsqueeze(2)
        col_pos = self.next_col_pos_embeddings.unsqueeze(1)
        
        # --- FIX #2: Use self.seq_len which is already defined (H * W) ---
        pos_encoding = (row_pos + col_pos).view(1, self.seq_len, self.hidden_size)
        
        # Iterative decoding
        for iteration in range(num_iterations):
            # --- FIX #3: Efficiently add the pre-calculated encoding ---
            x = self.embed_tokens(predicted_tokens) + pos_encoding
            
            # Forward pass through alternating layers
            total_contexts = len(contexts)
            context_idx = 0
            
            for layer in self.layers:
                if isinstance(layer, SelfAttentionBlock):
                    x = layer(x)
                elif isinstance(layer, CrossAttentionBlock):
                    x = layer(x, contexts[context_idx % total_contexts])
                    context_idx += 1
            
            x = self.norm(x)
            logits = self.lm_head(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            # Calculate masking ratio based on schedule
            t = (iteration + 1) / num_iterations
            if schedule == 'cosine':
                mask_ratio = np.cos(t * np.pi / 2)
            elif schedule == 'square':
                mask_ratio = 1 - t ** 2
            else:  # linear
                mask_ratio = 1 - t
            
            target_num_masked = int(self.seq_len * mask_ratio)
            
            # Reveal tokens based on confidence
            tokens_to_keep = torch.zeros_like(mask)
            masked_confidences = confidences.clone()
            masked_confidences[~mask] = -1
            
            for b in range(B):
                num_currently_masked = mask[b].sum().item()
                if num_currently_masked > target_num_masked:
                    num_to_reveal = num_currently_masked - target_num_masked
                    _, topk_indices = torch.topk(
                        masked_confidences[b], k=min(num_to_reveal, num_currently_masked)
                    )
                    tokens_to_keep[b, topk_indices] = True
            
            predicted_tokens[tokens_to_keep & mask] = predictions[tokens_to_keep & mask]
            mask = mask & ~tokens_to_keep
            
            if not mask.any():
                break
        
        return predicted_tokens.view(B, self.grid_size[0], self.grid_size[1])
