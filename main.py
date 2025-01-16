# pip install torch transformers diffusers accelerate datasets tqdm peft torchvision

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# !nvidia-smi

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gc

class MemoryEfficientInteriorTrainer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_hf_dataset(self, dataset_name):
        self.clear_memory()
        dataset = load_dataset(dataset_name)
        
        def preprocess_image(examples):
            images = []
            for img in examples['image']:
                if isinstance(img, Image.Image):
                    img_rgb = img.convert("RGB")
                elif isinstance(img, (str, bytes)):
                    img_rgb = Image.open(img).convert("RGB")
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                images.append(self.transform(img_rgb))
            
            images_tensor = torch.stack(images)
            
            tokenized_text = self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            
            return {
                'pixel_values': images_tensor,
                'input_ids': tokenized_text.input_ids,
                'attention_mask': tokenized_text.attention_mask
            }
        
        processed_dataset = dataset['train'].map(
            preprocess_image,
            batched=True,
            batch_size=4,
            remove_columns=dataset['train'].column_names
        )
        
        processed_dataset.set_format(
            type='torch',
            columns=['pixel_values', 'input_ids', 'attention_mask']
        )
        
        return {'train': processed_dataset}

    def train(self, dataset_name, num_epochs=10, batch_size=1, learning_rate=1e-5, gradient_accumulation_steps=4):
        self.clear_memory()
        
        dataset = self.load_hf_dataset(dataset_name)
        
        # Configure accelerator without mixed precision
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=None,
        )
        
        # Load models in float32
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, 
            subfolder="text_encoder",
        )
        vae = AutoencoderKL.from_pretrained(
            self.model_id, 
            subfolder="vae",
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id, 
            subfolder="unet",
        )
        
        # Freeze models
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Move to device and convert to half precision after freezing
        vae = vae.to(self.device).half()
        text_encoder = text_encoder.to(self.device).half()
        unet = unet.to(self.device)  # Keep UNet in float32
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        lr_scaled = learning_rate * (effective_batch_size / 8)
        optimizer = torch.optim.AdamW(
            unet.parameters(), 
            lr=lr_scaled,
            weight_decay=1e-2
        )
        
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, 
            subfolder="scheduler"
        )
        
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        
        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )
        
        # Create scaler with updated syntax
        scaler = torch.amp.GradScaler('cuda')
        
        global_step = 0
        for epoch in range(num_epochs):
            unet.train()
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    self.clear_memory()
                    
                    pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
                    input_ids = batch["input_ids"].to(self.device)
                    
                    # Use updated autocast syntax
                    with torch.amp.autocast('cuda'):
                        with torch.no_grad():
                            latents = vae.encode(pixel_values).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor
                    
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, 
                            (bsz,), 
                            device=latents.device
                        )
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        
                        with torch.no_grad():
                            encoder_hidden_states = text_encoder(input_ids)[0]
                        
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                    # Scale loss before backward pass
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Unscale gradients and step optimizer
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                
                if step % 10 == 0:
                    self.clear_memory()
            
            progress_bar.close()
            
            if (epoch + 1) % 5 == 0:
                self.clear_memory()
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(f"interior_epoch_{epoch + 1}")
        
        self.clear_memory()
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained("interior_final")
        
        return unwrapped_unet

    def generate_image(self, prompt, num_images=1):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=UNet2DConditionModel.from_pretrained("interior_final"),
            torch_dtype=torch.float16
        ).to(self.device)
        
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images
        
        return images
    


# Clear CUDA memory
import torch
import gc

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Initialize and train
trainer = MemoryEfficientInteriorTrainer()
trained_unet = trainer.train(
    dataset_name="razor7x/Interior_Design_Dataset",
    batch_size=1,
    num_epochs=10,
    gradient_accumulation_steps=4,
    learning_rate=1e-5
)