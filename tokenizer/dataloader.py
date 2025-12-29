import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import datetime
from tqdm import tqdm
import argparse
from dataset import create_pacman_dataloaders


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 features
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_features = models.vgg16(pretrained=True).features
        self.features = vgg_features

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x, target):
        h_x = self.features(x)
        h_target = self.features(target)
        return F.mse_loss(h_x, h_target)


def train_enhanced_vqvae(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=20,
    learning_rate=1e-5,
    save_dir="checkpoints",
    save_interval=5,
    image_interval=50,
    perceptual_weight=0.1,
    mse_weight=1.0,
    vq_weight=1.0,
    accumulation_steps=1,
    model_path=None,
    use_validation=True
):
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(save_dir, "logs", timestamp)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    mse_loss = nn.MSELoss()
    perceptual_loss = VGGPerceptualLoss().to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = {
            "total": 0.0,
            "mse": 0.0,
            "perceptual": 0.0,
            "vq": 0.0,
            "codebook_usage": 0.0
        }

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, images in enumerate(pbar):
            images = images.to(device)

            noisy_images = images
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(images) * 0.05
                noisy_images = torch.clamp(images + noise, 0, 1)

            recon, vq_loss, indices, z_e, z_q = model(noisy_images)

            m_loss = mse_loss(recon, images) * mse_weight
            p_loss = perceptual_loss(recon, images) * perceptual_weight
            vq_weighted_loss = vq_loss * vq_weight

            total_loss = (m_loss + p_loss + vq_weighted_loss) / accumulation_steps
            total_loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                codebook_usage = (
                    torch.unique(indices).numel()
                    / model.quantize.num_embeddings
                )

            epoch_losses["total"] += total_loss.item() * accumulation_steps
            epoch_losses["mse"] += m_loss.item()
            epoch_losses["perceptual"] += p_loss.item()
            epoch_losses["vq"] += vq_weighted_loss.item()
            epoch_losses["codebook_usage"] += codebook_usage

            pbar.set_postfix({
                "loss": epoch_losses["total"] / (i + 1),
                "mse": epoch_losses["mse"] / (i + 1),
                "percep": epoch_losses["perceptual"] / (i + 1),
                "vq": epoch_losses["vq"] / (i + 1),
                "cbuse": epoch_losses["codebook_usage"] / (i + 1)
            })

            step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Total", total_loss.item() * accumulation_steps, step)
            writer.add_scalar("Train/MSE", m_loss.item(), step)
            writer.add_scalar("Train/Perceptual", p_loss.item(), step)
            writer.add_scalar("Train/VQ", vq_weighted_loss.item(), step)
            writer.add_scalar("Train/CodebookUsage", codebook_usage, step)

            if i % image_interval == 0:
                save_reconstruction_images(
                    images, recon, writer, images_dir, epoch, i
                )

        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)

        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, epoch_losses["codebook_usage"], path)
            print(f"Saved checkpoint: {path}")

    writer.close()


def save_reconstruction_images(images, recon, writer, images_dir, epoch, batch_idx, n_samples=8):
    with torch.no_grad():
        n = min(n_samples, images.size(0))
        comparison = torch.cat([images[:n], recon[:n]], dim=0)
        grid = make_grid(comparison, nrow=n, normalize=True, value_range=(0, 1))
        writer.add_image(f"Recon/epoch_{epoch+1}", grid, batch_idx)


def save_checkpoint(model, optimizer, epoch, codebook_usage, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "codebook_usage": codebook_usage
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="checkpoints6")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--image_interval", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from vq_vae import VQModel
    model = VQModel().to(device)

    train_loader, val_loader = create_pacman_dataloaders(
        args.data_path, batch_size=args.batch_size
    )

    train_enhanced_vqvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        image_interval=args.image_interval,
        perceptual_weight=args.perceptual_weight,
        mse_weight=args.mse_weight,
        vq_weight=args.vq_weight,
        accumulation_steps=args.accumulation_steps,
        model_path=args.model_path,
        use_validation=(val_loader is not None)
    )


if __name__ == "__main__":
    main()
