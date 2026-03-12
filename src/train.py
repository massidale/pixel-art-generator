import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from accelerate import Accelerator
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import PixelArtDataset

# Calcola dinamicamente la root del progetto rispetto alla posizione di questo script
# Poiché script è in `src/train.py`, `parent.parent` punta alla root principale.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_rel_path):
    """Carica il file yaml risolvendo il path rispetto alla root del progetto."""
    config_path = PROJECT_ROOT / config_rel_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configurazione non trovata: {config_path}")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(rel_path):
    """Risolve un path stringa del config rispetto alla root e restituisce una stringa."""
    return str(PROJECT_ROOT / rel_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Script di training LoRA per Flux")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training.yaml", 
        help="Percorso relativo (dalla root) del file di configurazione yaml"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    output_dir = resolve_path(config['training']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Inizializzazione Accelerator (Best Practice per checkpoint e device)
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        project_dir=output_dir
    )

    # 1. Caricamento Modello
    # bfloat16 è supportato sia su MPS (Mac) che su GPU NVIDIA (CUDA/A100)
    device = accelerator.device
    dtype = torch.bfloat16 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else torch.float32
    
    gguf_config = GGUFQuantizationConfig(compute_dtype=dtype)
    transformer_path = resolve_path(config['model']['transformer_path'])
    
    accelerator.print(f"Loading transformer da: {transformer_path}")
    
    transformer = FluxTransformer2DModel.from_single_file(
        transformer_path,
        config=config['model']['config_name'],
        subfolder="transformer", # Specifico per FLUX1.schnell GGUF
        torch_dtype=dtype,
        quantization_config=gguf_config
    )

    # 2. Configurazione LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
    )
    
    transformer = get_peft_model(transformer, lora_config)
    
    # Best Practice: Gradient Checkpointing per risparmiare VRAM
    transformer.enable_gradient_checkpointing()

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(), 
        lr=float(config['training']['learning_rate'])
    )

    # 4. Dataset e DataLoader
    dataset = PixelArtDataset(
        csv_path=resolve_path(config['data']['csv_path']),
        precomputed_dir=resolve_path(config['data']['precomputed_dir'])
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['train_batch_size'], 
        shuffle=True
    )
    
    # 5. Scheduler e posizioni ID specifici di FLUX
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config['model']['config_name'], subfolder="scheduler"
    )

    # Prepara tutto con Accelerator
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    # Helper per img_ids e txt_ids
    # FLUX richiede "img_ids" che mappano le coordinate delle patch
    def prepare_ids(bsz, seq_len_img, seq_len_txt, device, dtype):
        # seq_len_img = (512//16) * (512//16) = 32 * 32 = 1024
        # Siccome latenti sono ridotti di un fattore complessivo 8 dalla VAE e 2x2 dal transformer
        h = int(seq_len_img ** 0.5) # 32
        latent_image_ids = torch.zeros(h, h, 3, dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(h)[None, :]
        latent_image_ids = latent_image_ids.reshape(h * h, 3).unsqueeze(0).repeat(bsz, 1, 1).to(device)
        
        txt_ids = torch.zeros(bsz, seq_len_txt, 3, dtype=dtype, device=device)
        return latent_image_ids, txt_ids

    # 6. Training Loop
    global_step = 0
    max_train_steps = config['training']['max_train_steps']
    
    progress_bar = tqdm(
        range(max_train_steps), 
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )
    
    accelerator.print(f"Inizio addestramento! Checkpoint salvati in {output_dir}")
    transformer.train()
    
    train_iter = iter(dataloader)
    
    for step in range(max_train_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)

        with accelerator.accumulate(transformer):
            latent = batch["latent"].to(device, dtype=dtype)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=dtype)
            
            bsz, seq_len_img, _ = latent.shape
            _, seq_len_txt, _ = prompt_embeds.shape
            
            img_ids, txt_ids = prepare_ids(bsz, seq_len_img, seq_len_txt, device, dtype)
            
            # Flow Matching:
            # 1. Sample noise
            noise = torch.randn_like(latent)
            
            # 2. Sample random timesteps (0 to 1) 
            # In diffusers per FLUX, t è campionato uniformemente
            u = torch.rand((bsz,), device=device, dtype=dtype)
            
            # 3. Create noisy inputs `(1-t) * data + t * noise` 
            # (Flux convention actually expects t*noise + (1-t)*latent)
            u_expanded = u.unsqueeze(1).unsqueeze(2)
            noisy_latent = (1.0 - u_expanded) * latent + u_expanded * noise
            
            # 4. Model predict. Flux uses timestep = t * 1000
            timesteps = u * 1000.0

            # Forward pass
            model_pred = transformer(
                hidden_states=noisy_latent,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                timestep=timesteps,
                return_dict=False
            )[0]
            
            # 5. La loss in Flow Matching è tipicamente MSE tra target vettoriale e la predizione.
            # target = noise - latent
            target = noise - latent
            
            loss = F.mse_loss(model_pred, target)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
            
            # Salvataggio Checkpoint
            if global_step % config['training']['checkpointing_steps'] == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                
                unwrapped_model = accelerator.unwrap_model(transformer)
                unwrapped_model.save_pretrained(os.path.join(save_path, "lora_weights"))

    accelerator.wait_for_everyone()
    final_save_path = os.path.join(output_dir, "final_lora")
    accelerator.print(f"Salvataggio del modello LoRA finale in {final_save_path}")
    
    unwrapped_model = accelerator.unwrap_model(transformer)
    unwrapped_model.save_pretrained(final_save_path)

if __name__ == "__main__":
    main()