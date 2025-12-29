import torch
import torch.nn as nn
import torch.nn.functional as F


ch = 24
em= 16
num_em=64
sem_nc =8
grp = 8

class Swish(nn.Module):
    """A simple Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SPADE(nn.Module):
    """
    SPatially-Adaptive (DE)normalization layer.
    Uses a semantic map to learn spatial affine transformations.
    """
    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()
        # CORRECTED: Changed num_groups from 32 to 8 to be compatible with new channel counts
        self.param_free_norm = nn.GroupNorm(grp, norm_nc, affine=False)
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

class ResnetBlock(nn.Module):
    """Standard ResNet block used in the Encoder."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        
        # CORRECTED: Changed num_groups from 32 to 8
        self.norm1 = nn.GroupNorm(grp, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # CORRECTED: Changed num_groups from 32 to 8
        self.norm2 = nn.GroupNorm(grp, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.norm1(x))
        h = self.conv1(h)
        h = self.swish(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class SPADEResnetBlock(nn.Module):
    """ResNet block with SPADE normalization used in the Decoder."""
    def __init__(self, in_channels, out_channels=None, semantic_nc=8):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        
        self.norm1 = SPADE(in_channels, semantic_nc)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SPADE(out_channels, semantic_nc)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.norm_shortcut = SPADE(in_channels, semantic_nc)
        else:
            self.shortcut = nn.Identity()
            self.norm_shortcut = None
        self.swish = Swish()

    def forward(self, x, seg):
        h = self.swish(self.norm1(x, seg))
        h = self.conv1(h)
        h = self.swish(self.norm2(h, seg))
        h = self.conv2(h)
        
        if self.norm_shortcut is not None:
            x_s = self.shortcut(self.norm_shortcut(x, seg))
        else:
            x_s = self.shortcut(x)
        return h + x_s

# --- VQ-VAE Components ---
class VectorQuantizer(nn.Module):
    """Improved Vector Quantizer with EMA updates."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.decay = decay

    def forward(self, z):
        # z has shape (B, C, H, W)
        b, c, h, w = z.shape
        
        # Flatten input: (B, C, H, W) -> (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                   + torch.sum(self.embedding.weight**2, dim=1)
                   - 2 * torch.matmul(z_flat, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        z_q_flat = torch.matmul(encodings, self.embedding.weight)
        
        # --- CORRECTED RESHAPING LOGIC ---
        # 1. Reshape back to (B, H, W, C)
        # 2. Permute to (B, C, H, W)
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5) * n)
            
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        loss = self.commitment_cost * e_latent_loss

        z_q = z + (z_q - z).detach()
        
        return z_q, loss, encoding_indices.view(b, h, w)

class SemanticFeatureExtractor(nn.Module):
    """Simplified CNN to extract semantic map from quantized latents."""
    def __init__(self, input_dim, semantic_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, semantic_dim * 2, kernel_size=3, padding=1),
            # This GroupNorm is fine as semantic_dim=8, so channels=16
            nn.GroupNorm(grp, semantic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(semantic_dim * 2, semantic_dim, kernel_size=1),
        )
    def forward(self, z_q):
        return self.net(z_q)

# --- Main Model Architecture ---

class Encoder(nn.Module):
    """A compact encoder network."""
    def __init__(self, in_channels=3, ch=ch, ch_mult=(1, 2, 4), num_res_blocks=2, embedding_dim=16):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(len(ch_mult)):
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.down.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if i_level != len(ch_mult) - 1:
                self.down.append(nn.Conv2d(block_in, block_in, kernel_size=4, stride=2, padding=1))

        self.final_block = nn.Sequential(
            # This GroupNorm is fine as block_in=96
            nn.GroupNorm(grp, block_in),
            Swish(),
            nn.Conv2d(block_in, embedding_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.down:
            h = block(h)
        return self.final_block(h)

class SPADEDecoder(nn.Module):
    """A compact decoder network using SPADE."""
    def __init__(self, out_channels=3, ch=ch, ch_mult=(1, 2, 4), num_res_blocks=2, embedding_dim=16, semantic_nc=8):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(embedding_dim, block_in, kernel_size=3, padding=1)

        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.up.append(SPADEResnetBlock(block_in, block_out, semantic_nc))
                block_in = block_out
            if i_level != 0:
                self.up.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(block_in, block_in, kernel_size=3, padding=1)
                ))

        self.final_block = nn.Sequential(
            SPADE(block_in, semantic_nc),
            Swish(),
            nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z, seg):
        h = self.conv_in(z)
        for module in self.up:
            if isinstance(module, SPADEResnetBlock):
                h = module(h, seg)
            else:
                h = module(h)
        return self.final_block[2](self.final_block[1](self.final_block[0](h, seg)))

class VQModel(nn.Module):
    """Compressed VQ-VAE model with self-conditioned SPADE normalization."""
    def __init__(self, num_embeddings=num_em, embedding_dim=em, commitment_cost=0.25, semantic_nc=sem_nc):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = SPADEDecoder(embedding_dim=embedding_dim, semantic_nc=semantic_nc)
        self.quantize = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.semantic_extractor = SemanticFeatureExtractor(
            input_dim=embedding_dim, 
            semantic_dim=semantic_nc
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        seg = self.semantic_extractor(z_q)
        x_recon = self.decoder(z_q, seg)
        return x_recon, vq_loss, indices, z_e, z_q

    def encode(self, x):
        """Encodes an image and returns the discrete latent indices."""
        z = self.encoder(x)
        _, _, indices = self.quantize(z)
        return indices

    def decode(self, indices):
        """Decodes a map of latent indices back into an image."""
        z_q_flat = self.quantize.embedding(indices.view(-1))
        
        batch_size, h, w = indices.shape
        embedding_dim = self.quantize.embedding_dim
        z_q = z_q_flat.view(batch_size, h, w, embedding_dim).permute(0, 3, 1, 2).contiguous()
        
        seg = self.semantic_extractor(z_q)
        x_recon = self.decoder(z_q, seg)
        return x_recon
        # helper functions
    # helper functions
    def encode_to_indices(self, input):
        """
        Encodes an image directly to its discrete codebook indices.
        """
        z_e = self.encoder(input)
        _, _, indices = self.quantize(z_e)
        return indices

    def decode_from_indices(self, indices):
        """
        Decodes an image from a grid of codebook indices.
        """
        # Get embeddings for the indices
        z_q_flat = self.quantize.embedding(indices.view(-1))
        
        # Reshape to (B, H, W, C) then permute to (B, C, H, W)
        batch_size, h, w = indices.shape
        embedding_dim = self.quantize.embedding_dim
        z_q = z_q_flat.view(batch_size, h, w, embedding_dim).permute(0, 3, 1, 2).contiguous()
        
        # Extract semantic features and decode
        seg = self.semantic_extractor(z_q)
        x_recon = self.decoder(z_q, seg)
        return x_recon
    def encode_full(self, x):
        """
        Encode input to continuous latent representation.
        This is the missing method that the training code is calling.
        It's essentially the same as encode() but with a different name for compatibility.
        """
        z = self.encoder(x)
        return z

