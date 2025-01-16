import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
import accelerate
from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

class InteriorLoraTrainer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        
    def prepare_dataset(self, dataset_path, prompt_prefix="a photo of interior design, "):
        # Load and prepare dataset
        dataset = load_dataset(dataset_path)
        
        def preprocess(examples):
            # Prepare images
            images = [image.convert("RGB").resize((512, 512)) for image in examples["image"]]
            
            # Prepare prompts (you can customize based on your needs)
            prompts = [prompt_prefix + desc for desc in examples["description"]]
            
            # Tokenize prompts
            tokenized_prompts = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": images,
                "input_ids": tokenized_prompts.input_ids,
                "attention_mask": tokenized_prompts.attention_mask,
            }
        
        processed_dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return processed_dataset
    
    def setup_lora(self):
        # Initialize pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Setup LoRA attention processors
        lora_attn_procs = {}
        for name, _ in pipeline.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            else:
                block_id = int(name[len("down_blocks.")])
                hidden_size = list(pipeline.unet.config.block_out_channels)[block_id]
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,  # Lower rank for faster training
            )
        
        pipeline.unet.set_attn_processor(lora_attn_procs)
        
        return pipeline, lora_attn_procs
    
    def train(self, dataset, num_epochs=10, batch_size=4, learning_rate=1e-4):
        # Setup accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )
        
        # Setup pipeline and LoRA
        pipeline, lora_attn_procs = self.setup_lora()
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            [x for x in lora_attn_procs.parameters() if x.requires_grad],
            lr=learning_rate,
        )
        
        # Prepare dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Prepare for training
        pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
            pipeline.unet, optimizer, train_dataloader
        )
        
        # Training loop
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(pipeline.unet):
                    # Convert images to latent space
                    latents = pipeline.vae.encode(
                        batch["pixel_values"].to(dtype=torch.float16)
                    ).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, pipeline.scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device
                    )
                    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get model prediction
                    encoder_hidden_states = pipeline.text_encoder(batch["input_ids"])[0]
                    noise_pred = pipeline.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states
                    ).sample
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                if step % 10 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        # Save LoRA weights
        pipeline.save_pretrained("interior_design_lora")
        
    def generate_image(self, prompt, num_images=1):
        # Load trained pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "interior_design_lora",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Generate images
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images
        
        return images