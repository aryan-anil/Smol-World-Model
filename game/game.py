import torch
import torch.nn.functional as F
import pygame
import numpy as np
import cv2
import h5py
import os
import time
from pygame.locals import *
from transformer.latent_predictor import LatentPredictor
from tokenizer.vq_vae. import VQModel as VQVAE


class GameFrameGenerator:
    def __init__(
        self,
        vqvae_path,
        transformer_path,
        initial_frame=None,
        frame_size=(80, 80),
        display_scale=3,
        fps=30,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ghosts=1,
        num_actions=5
    ):
        self.device = device
        self.frame_size = frame_size
        self.display_scale = display_scale
        self.fps = fps
        self.num_ghosts = num_ghosts
        self.num_actions = num_actions

        pygame.init()
        self.screen_size = (frame_size[0] * display_scale, frame_size[1] * display_scale)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("VQ-VAE Frame Generator with Ghost Actions")
        self.clock = pygame.time.Clock()

        self.action_map = {
            K_UP: 1,
            K_DOWN: 2,
            K_LEFT: 3,
            K_RIGHT: 4,
            None: 0
        }

        self.action_names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
        self.reset_token = 0

        self.load_models(vqvae_path, transformer_path)

        if initial_frame is not None:
            self.current_frame = self.load_initial_frame(initial_frame)
        else:
            print("Error: No initial frame provided!")
            exit(0)

        self.frame_history = []
        self.user_action_history = []
        self.ghost_action_history = []
        self.max_history = 10

        self.ghost_action_probability = 0.7
        self.ghost_direction_persistence = 0.3
        self.previous_ghost_actions = np.zeros(self.num_ghosts, dtype=int)

    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def human_readable_size(self, num_params, dtype_bytes=4):
        size_bytes = num_params * dtype_bytes
        if size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.2f} MB"
        else:
            return f"{size_bytes/1024**3:.2f} GB"

    def load_models(self, vqvae_path, transformer_path):
        print("Loading models...")

        vqvae_ckpt = torch.load(vqvae_path, map_location=self.device)
        self.vqvae = VQVAE().to(self.device)
        self.vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
        self.vqvae.eval()

        transformer_ckpt = torch.load(transformer_path, map_location=self.device)
        self.transformer = LatentPredictor(
            vocab_size=64,
            grid_size=(10, 10),
            num_ghosts=self.num_ghosts
        ).to(self.device)
        self.transformer.load_state_dict(transformer_ckpt['model_state_dict'])
        self.transformer.eval()

        vqvae_params = self.count_params(self.vqvae)
        transformer_params = self.count_params(self.transformer)
        total_params = vqvae_params + transformer_params

        print(f"VQ-VAE params: {vqvae_params:,} ({self.human_readable_size(vqvae_params)})")
        print(f"Transformer params: {transformer_params:,} ({self.human_readable_size(transformer_params)})")
        print(f"Total params: {total_params:,} ({self.human_readable_size(total_params)})")
        print("Models loaded successfully!")

    def load_initial_frame(self, initial_frame_path):
        with h5py.File(initial_frame_path, 'r') as h5_file:
            frames = h5_file['frames']
            if len(frames) > 0:
                frame = torch.from_numpy(frames[0]).float() / 255.0
                frame = frame.permute(2, 0, 1).unsqueeze(0).to(self.device)
                return frame
        raise RuntimeError("No frames found in HDF5 file")

    def generate_ghost_actions(self):
        ghost_actions = np.zeros(self.num_ghosts, dtype=int)
        for i in range(self.num_ghosts):
            if np.random.random() < self.ghost_action_probability:
                if self.previous_ghost_actions[i] != 0 and np.random.random() < self.ghost_direction_persistence:
                    ghost_actions[i] = self.previous_ghost_actions[i]
                else:
                    ghost_actions[i] = np.random.randint(1, self.num_actions)
            else:
                ghost_actions[i] = 0
        self.previous_ghost_actions = ghost_actions.copy()
        return ghost_actions

    def tensor_to_surface(self, tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.resize(img, self.screen_size)
        return pygame.surfarray.make_surface(img.transpose(1, 0, 2))

    def get_action_from_keys(self):
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            return 1
        elif keys[K_RIGHT]:
            return 4
        elif keys[K_DOWN]:
            return 2
        elif keys[K_LEFT]:
            return 3
        return 0

    def predict_next_frame(self, current_frame, user_action, ghost_actions, reset_token):
        with torch.no_grad():
            z_e = self.vqvae.encode_full(current_frame)
            _, _, indices = self.vqvae.quantize(z_e)
            indices = indices.view(-1, 10, 10)

            user_action_tensor = torch.tensor([user_action], device=self.device)
            ghost_action_tensor = torch.tensor([ghost_actions], device=self.device)
            reset_token_tensor = torch.tensor([reset_token], device=self.device)

            predicted_indices = self.transformer.generate_with_schedule(
                user_actions=user_action_tensor,
                ghost_actions=ghost_action_tensor,
                current_frame_indices=indices,
                reset_tokens=reset_token_tensor
            )

            if predicted_indices is None:
                return current_frame

            frame = self.vqvae.decode_from_indices(predicted_indices)
            return torch.clamp(frame, 0, 1)

    def update_display(self, frame, user_action, ghost_actions, reset_token):
        surface = self.tensor_to_surface(frame)
        self.screen.blit(surface, (0, 0))

        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        self.screen.blit(
            font.render(f"User Action: {self.action_names[user_action]}", True, (255, 255, 255)),
            (10, 10)
        )

        if reset_token == 1:
            self.screen.blit(font.render("RESET", True, (255, 0, 0)), (10, 50))
            y_offset = 90
        else:
            y_offset = 50

        self.screen.blit(small_font.render("Ghost Actions:", True, (255, 255, 255)), (10, y_offset))

        for i, ga in enumerate(ghost_actions):
            self.screen.blit(
                small_font.render(f"Ghost {i+1}: {self.action_names[ga]}", True, (200, 200, 200)),
                (10, y_offset + 25 + i * 25)
            )

        pygame.display.flip()

    def run(self):
        running, paused = True, False
        self.frame_history.append(self.current_frame.clone())
        self.user_action_history.append(0)
        self.ghost_action_history.append(np.zeros(self.num_ghosts, dtype=int))

        while running:
            self.reset_token = 0
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        paused = not paused
                    elif event.key == K_r:
                        self.reset_token = 1
                        self.previous_ghost_actions[:] = 0

            if not paused:
                ua = self.get_action_from_keys()
                ga = self.generate_ghost_actions()
                self.current_frame = self.predict_next_frame(self.current_frame, ua, ga, self.reset_token)
                self.frame_history.append(self.current_frame.clone())
                self.user_action_history.append(ua)
                self.ghost_action_history.append(ga.copy())

                if len(self.frame_history) > self.max_history:
                    self.frame_history.pop(0)
                    self.user_action_history.pop(0)
                    self.ghost_action_history.pop(0)

            self.update_display(
                self.current_frame,
                self.user_action_history[-1],
                self.ghost_action_history[-1],
                self.reset_token
            )
            time.sleep(2)

        pygame.quit()


if __name__ == "__main__":
    generator = GameFrameGenerator(
        vqvae_path=r"",
        transformer_path=r"",
        initial_frame=r"",
        fps=1,
        num_ghosts=1,
        num_actions=5
    )
    generator.run()
