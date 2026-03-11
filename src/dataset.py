import csv
import torch
from torch.utils.data import Dataset
import os


class PixelArtDataset(Dataset):
    def __init__(self, csv_path, precomputed_dir):
        self.precomputed_dir = precomputed_dir
        self.data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.data.append((row[0], row[1]))

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        base_name = os.path.splitext(filename)[0]
        pt_path = os.path.join(self.precomputed_dir, f"{base_name}.pt")

        data_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        
        # Le latenze FLUX necessitano di un packing specifico
        # Di base la shape è [C, H, W], Flux usa sequence length packing.
        latent = data_dict["latent"]
        # Convertiamo la latitudine 2D [C, H, W] nel formato richiesto: [H*W//4, C*4]
        # FLUX usa latenti patchati 2x2. C=16
        c, h, w = latent.shape
        latent = latent.view(c, h//2, 2, w//2, 2)
        latent = latent.permute(1, 3, 0, 2, 4)
        latent = latent.reshape(h//2 * w//2, c * 4)

        return {
            "latent": latent,
            "prompt_embeds": data_dict["prompt_embeds"],
            "pooled_prompt_embeds": data_dict["pooled_prompt_embeds"],
        }
