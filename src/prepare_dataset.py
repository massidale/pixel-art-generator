import os
import csv
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import argparse
from pathlib import Path

# Fix per resolve dei path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute FLUX latents and embeddings")
    parser.add_argument("--img_dir", type=str, default="dataset/pixel-art-32")
    parser.add_argument("--csv_path", type=str, default="dataset/pixel-art-32.csv")
    parser.add_argument("--output_dir", type=str, default="dataset/pixel-art-32-flux")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    return parser.parse_args()

def main():
    args = parse_args()
    
    img_dir_path = PROJECT_ROOT / args.img_dir
    csv_path = PROJECT_ROOT / args.csv_path
    output_dir_path = PROJECT_ROOT / args.output_dir
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" or device == "cuda" else torch.float32
    
    print(f"Loading models from {args.model_id} on {device} (dtype: {dtype})...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    
    # Load Tokenizers and Text Encoders
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.model_id, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    
    print("Reading dataset...")
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                data.append((row[0], row[1]))
                
    print(f"Found {len(data)} items to process.")
    
    # We will process image by image to avoid OOM
    for idx, (filename, caption) in enumerate(tqdm(data, desc="Processing datasets")):
        img_path = img_dir_path / filename
        if not img_path.exists():
            print(f"Skipping {filename}, not found.")
            continue
            
        # 1. Process Image -> Latent
        image = Image.open(img_path).convert("RGB")
        # Pixel-art specific scaling
        image = image.resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
        
        # Normalize [-1, 1]
        img_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.reshape(args.resolution, args.resolution, 3)
        img_tensor = img_tensor.permute(2, 0, 1) / 127.5 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)
        
        with torch.no_grad():
            # encode using VAE
            latent = vae.encode(img_tensor).latent_dist.sample()
            # FLUX relies on specific scaling/shifting, typically handled in pipeline, 
            # here we just grab the raw latent. We will scale/pack it in the dataset.
            # Scale it to standard vae shift/scale during packing or here?
            # Easiest is to save raw latents and process in dataset.
            latent = latent.cpu().squeeze(0)
            
        # 2. Process Text -> Embeddings
        # CLIP Tokenizer
        text_inputs_1 = tokenizer_1(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        # T5 Tokenizer
        text_inputs_2 = tokenizer_2(
            caption,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # CLIP → pooled embedding [hidden_size]  (= pooled_prompt_embeds in FLUX)
            pooled_prompt_embeds = text_encoder_1(
                text_inputs_1.input_ids.to(device),
                output_hidden_states=False
            ).pooler_output
            
            # T5 → sequence embedding [seq_len, hidden_size]  (= prompt_embeds in FLUX)
            prompt_embeds = text_encoder_2(
                text_inputs_2.input_ids.to(device),
                output_hidden_states=True
            ).last_hidden_state
            
        # Save precomputed tensors (FLUX convention: prompt_embeds=T5, pooled_prompt_embeds=CLIP)
        out_dict = {
            "latent": latent,
            "prompt_embeds": prompt_embeds.cpu().squeeze(0),         # [seq_len, hidden]
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu().squeeze(0)  # [hidden]
        }
        
        base_name = os.path.splitext(filename)[0]
        out_file = output_dir_path / f"{base_name}.pt"
        torch.save(out_dict, out_file)
        
    print(f"Done! Processed latents and embeddings saved in {output_dir_path}")

if __name__ == "__main__":
    main()
