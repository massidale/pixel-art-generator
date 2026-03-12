#!/usr/bin/env python3
import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

# Root del progetto (src/../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Genera immagini con FLUX + LoRA pixel art")
    parser.add_argument(
        "prompts",
        nargs="+",
        help='Uno o più prompt, es: "a pixel art tree, 32x32 style"',
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="output/flux-pixel-art-lora/final_lora",
        help="Percorso relativo (dalla root) della cartella con i pesi LoRA",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/generated",
        help="Cartella dove salvare le immagini generate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Numero di step di inferenza (FLUX-schnell: 4 è ottimale)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale (FLUX-schnell usa 0.0 per distillation)",
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Larghezza immagine generata"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Altezza immagine generata"
    )
    parser.add_argument(
        "--save-32",
        action="store_true",
        default=True,
        help="Se True, salva anche una versione 32x32 (pixel art look)",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default="models/flux1-schnell-Q4_K_S.gguf",
        help="Percorso relativo al file GGUF del transformer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # ── 1. Carica il Transformer quantizzato GGUF ───────────────────────────
    transformer_path = str(PROJECT_ROOT / args.transformer_path)
    print(f"Loading GGUF transformer: {transformer_path}")

    gguf_config = GGUFQuantizationConfig(compute_dtype=dtype)
    transformer = FluxTransformer2DModel.from_single_file(
        transformer_path,
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        torch_dtype=dtype,
        quantization_config=gguf_config,
    )

    # ── 2. Applica i pesi LoRA PEFT al transformer ──────────────────────────
    # I pesi LoRA sono in formato PEFT (saved via save_pretrained).
    # Non usare pipe.load_lora_weights() che missidentifica il formato.
    from peft import PeftModel
    lora_path = str(PROJECT_ROOT / args.lora_dir)
    print(f"Loading LoRA weights (PEFT) from: {lora_path}")
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    # Non fare merge_and_unload(): i pesi GGUF quantizzati non sono compatibili con il merge floating-point.
    # La LoRA rimane attiva come adapter sopra il modello base, il che è corretto per l'inferenza GGUF.
    transformer.eval()

    # ── 3. Carica la Pipeline FLUX completa (VAE + Text Encoders + Scheduler) ─
    print("Loading FLUX pipeline (VAE, text encoders, scheduler)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()

    # ── 4. Generazione ──────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(args.prompts):
        print(f"\n[{i+1}/{len(args.prompts)}] Generating: '{prompt}'")
        result = pipe(
            prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
        )
        image = result.images[0]

        # Salva originale 512x512
        out_path = output_dir / f"generated_{i:03d}.png"
        image.save(out_path)
        print(f"  Saved: {out_path}")

        # Salva versione 32x32 pixel art look
        if args.save_32:
            image_32 = image.resize((32, 32), Image.Resampling.NEAREST)
            out_path_32 = output_dir / f"generated_{i:03d}_32x32.png"
            image_32.save(out_path_32)
            print(f"  Saved: {out_path_32}")

    print(f"\n✅ Done! {len(args.prompts)} image(s) saved in: {output_dir}")


if __name__ == "__main__":
    main()
