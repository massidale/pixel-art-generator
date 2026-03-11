import os
import argparse
import yaml
import torch
from PIL import Image
from diffusers import DiffusionPipeline


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate(config_path, prompts, output_dir, num_inference_steps=50):
    cfg = load_config(config_path)
    model_name = cfg["model"]["name"]
    lora_dir = os.path.join(cfg["output"]["dir"], "final")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(model_name)
    pipe.load_lora_weights(lora_dir)
    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"Generating: {prompt}")
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        # Scala a 32x32 per vedere il risultato pixel art
        image_32 = image.resize((32, 32), Image.Resampling.NEAREST)
        out_path = os.path.join(output_dir, f"generated_{i}.png")
        image_32.save(out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--output-dir", default="outputs/generated")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "prompts",
        nargs="+",
        help='Uno o più prompt, es: "Spada Magica, 32x32, pixel art style"',
    )
    args = parser.parse_args()
    generate(args.config, args.prompts, args.output_dir, args.steps)
