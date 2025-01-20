import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from PIL import Image
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

class MemoryEfficientInteriorTrainer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
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
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
                else:
                    print(f"Skipping unsupported image type: {type(img)}")
            
            if not images:
                return {}
            
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
        
        processed_dataset = processed_dataset.filter(lambda x: len(x) > 0 and 'pixel_values' in x and 'input_ids' in x and 'attention_mask' in x)
        
        if len(processed_dataset) == 0:
            raise ValueError("No valid data in the dataset after preprocessing. Please check your dataset and preprocessing steps.")
        
        processed_dataset.set_format(
            type='torch',
            columns=['pixel_values', 'input_ids', 'attention_mask']
        )
        
        return {'train': processed_dataset}

    def train(self, dataset_name, num_epochs=10, batch_size=1, learning_rate=1e-5, gradient_accumulation_steps=4):
        self.clear_memory()
        
        dataset = self.load_hf_dataset(dataset_name)
        
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="no",  # Change this to "no" to avoid FP16 issues
        )
        
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        optimizer = torch.optim.AdamW(
            unet.parameters(), 
            lr=learning_rate,
            weight_decay=1e-2
        )

        noise_scheduler = PNDMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )
        
        global_step = 0
        for epoch in range(num_epochs):
            unet.train()
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    self.clear_memory()
                    
                    pixel_values = batch["pixel_values"].to(self.device)
                    input_ids = batch["input_ids"].to(self.device)
                    
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
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
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=unwrapped_unet,
        )
        pipeline.save_pretrained("interior_pipeline")
        
        return unwrapped_unet

    def generate_image(self, prompt, num_images=1):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "interior_pipeline",
        ).to(self.device)
        
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images
        
        return images

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    trainer = MemoryEfficientInteriorTrainer()
    trained_unet = trainer.train(
        dataset_name="razor7x/Interior_Design_Dataset",
        batch_size=1,
        num_epochs=10,
        gradient_accumulation_steps=4,
        learning_rate=1e-5
    )

    prompt = "A modern minimalist living room with large windows"
    generated_images = trainer.generate_image(prompt)
    generated_images[0].save("/kaggle/working/generated_interior.png")
    print("Image generated and saved as 'generated_interior.png'")

