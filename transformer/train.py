import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn.functional as F
from VQ_VAE.vq_spade import VQModel as VQVAE 
from dataset import H5Dataset
from latent_predictor import LatentPredictor


def train_model_predictor(
    vqvae_checkpoint_path,
    dataset_path,
    save_dir='checkpoints', 
    log_dir='logs',
    batch_size=64,
    num_epochs=100,
    learning_rate=3e-4,
    device='cuda',
    num_actions=5,
    latent_channels=3,
    num_codebook_vectors=512,
    grid_size=(10, 10),
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=8,
    save_interval=10,
    eval_interval=5,
    num_ghosts=4,
    load_path=None,
    accumulation_steps=1,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    vqvae = VQVAE().to(device)

    checkpoint = torch.load(vqvae_checkpoint_path, map_location=device)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae.eval()
    print("VQ-VAE model loaded successfully.")

  
    model = LatentPredictor(
        vocab_size=num_codebook_vectors,
        num_ghosts=num_ghosts,
        grid_size=grid_size
    ).to(device)

  
    dataset = H5Dataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

   
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2
    )


    global_step = 0
    start_epoch = 0
    print(f"🚀 Starting training on {device} for {num_epochs} epochs.")

    if load_path and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', start_epoch * len(dataloader))
        print(f"🔁 Resuming from epoch {start_epoch}")


  
    for epoch in range(start_epoch, num_epochs):
        model.train()

        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="batch"
        )

        accumulation_count = 0
        accumulated_total_loss = 0.0
        accumulated_ce_loss = 0.0

        for batch_idx, batch_data in enumerate(progress_bar):
            frames, user_actions, ghost_actions, next_frames, reset_tokens = batch_data

            frames = frames.to(device)
            user_actions = user_actions.to(device)
            ghost_actions = ghost_actions.to(device)
            next_frames = next_frames.to(device)
            reset_tokens = reset_tokens.to(device)

            with torch.no_grad():
                z_e_current = vqvae.encode_full(frames)
                _, _, src_indices = vqvae.quantize(z_e_current)
                src_indices = src_indices.reshape(-1, grid_size[0], grid_size[1])

                z_e_target = vqvae.encode_full(next_frames)
                _, _, tgt_indices = vqvae.quantize(z_e_target)
                tgt_indices = tgt_indices.reshape(-1, grid_size[0], grid_size[1])

            result = model(
                current_frame_indices=src_indices,
                user_actions=user_actions,
                ghost_actions=ghost_actions,
                target=tgt_indices,
                reset_tokens=reset_tokens,
            )

            loss = result['loss'] / accumulation_steps
            loss.backward()

            accumulated_total_loss += loss.item()
            accumulated_ce_loss += loss.item()
            accumulation_count += 1

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                avg_total = accumulated_total_loss / accumulation_count
                avg_ce = accumulated_ce_loss / accumulation_count

                epoch_total_loss += avg_total
                epoch_ce_loss += avg_ce

                progress_bar.set_postfix(
                    total_loss=f"{avg_total:.4f}",
                    ce_loss=f"{avg_ce:.4f}",
                    acc_steps=accumulation_count
                )

                writer.add_scalar('Loss/train_batch_total', avg_total, global_step)
                writer.add_scalar('Loss/train_batch_ce', avg_ce, global_step)

                global_step += 1
                accumulation_count = 0
                accumulated_total_loss = 0.0
                accumulated_ce_loss = 0.0

        num_optimizer_steps = (
            len(dataloader) // accumulation_steps +
            (1 if len(dataloader) % accumulation_steps != 0 else 0)
        )

        avg_total_loss = epoch_total_loss / num_optimizer_steps
        avg_ce_loss = epoch_ce_loss / num_optimizer_steps

        writer.add_scalar('Loss/train_epoch_total', avg_total_loss, epoch)
        writer.add_scalar('Loss/train_epoch_ce', avg_ce_loss, epoch)
        writer.add_scalar('Meta/learning_rate', learning_rate, epoch)
        writer.add_scalar('Meta/accumulation_steps', accumulation_steps, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Total: {avg_total_loss:.6f} | "
            f"CE: {avg_ce_loss:.6f} | "
            f"LR: {learning_rate:.2e} | "
            f"Acc: {accumulation_steps}"
        )

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
                'ce_loss': avg_ce_loss,
                'global_step': global_step,
                'accumulation_steps': accumulation_steps
            }, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")

    writer.close()
    return model


if __name__ == "__main__":
    vqvae_checkpoint_path = r""
    data_path = r""

    trained_model = train_model_predictor(
        vqvae_checkpoint_path=vqvae_checkpoint_path,
        dataset_path=data_path,
        save_dir='checkpoints_model_pacman4',
        log_dir='logs_model_pacman4',
        batch_size=8,
        num_epochs=16,
        learning_rate=1e-4,
        num_actions=5,
        latent_channels=16,
        num_codebook_vectors=64,
        grid_size=(10, 10),
        hidden_size=512,
        save_interval=1,
        num_ghosts=1,
        accumulation_steps=8,
        load_path=None
    )
