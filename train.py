import os
import torch
import torch.nn.functional as F
import argparse
import time

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from dataclasses import dataclass
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, DDPMScheduler, Transformer2DModel, DiffusionPipeline
from accelerate import notebook_launcher
from typing import List, Dict, Optional
from PIL import Image
from loguru import logger
from functools import partial
from torch.utils.flop_counter import FlopCounterMode

@dataclass
class TrainingConfig:
    image_size: int = 128
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = "bf16"
    output_dir: str = "ddpm-butterflies-128"
    push_to_hub: bool = True
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    eval: bool = False
    optim: bool = False
    debug: bool = True

class DDPMPipelineTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Training script for DDPM")
        parser.add_argument("--image_size", type=int, default=128, help="Generated image resolution")
        parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
        parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size")
        parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
        parser.add_argument("--save_image_epochs", type=int, default=10, help="Save image every n epochs")
        parser.add_argument("--save_model_epochs", type=int, default=30, help="Save model every n epochs")
        parser.add_argument("--mixed_precision", type=str, default="fp16", help="Choose from no, fp16, bf16 or fp8")
        parser.add_argument("--output_dir", type=str, default="ddpm-butterflies-128", help="Output directory")
        parser.add_argument("--push_to_hub", default=False, help="Upload the saved model to the HF Hub")
        parser.add_argument("--hub_private_repo", action='store_true', help="Make the HF Hub repo private")
        parser.add_argument("--overwrite_output_dir", action='store_true', help="Overwrite the old model")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--eval", type=bool, default=True)
        parser.add_argument("--optim", type=bool, default=False)
        parser.add_argument("--debug", type=bool, default=False)

        return parser.parse_args()

    def preprocess_dataset(self, dataset) -> None:
        preprocess = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def transform(examples: Dict[str, List[Image.Image]]) -> Dict[str, List[torch.Tensor]]:
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)
        logger.info("Pre-processing done")

    def make_grid(self, images: List[Image.Image], rows: int, cols: int) -> Image.Image:
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, epoch: int, pipeline: DDPMPipeline) -> None:
        images = pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.manual_seed(self.config.seed),
        ).images

        image_grid = self.make_grid(images, rows=4, cols=4)

        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
        logger.debug(f"Evals are saved at {test_dir}/{epoch:04d}.png")

    def get_full_repo_name(self, model_id: str, organization: Optional[str] = None, token: Optional[str] = None) -> str:
        if token is None:
            token = HfFolder.get_token()
        if organization is None:
            username = whoami(token)["name"]
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    def estimate_mfu(self, total_flops, steps, total_time_seconds):
        """ Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        flops_per_iter = total_flops / steps
        flops_achieved = flops_per_iter / total_time_seconds  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def train_loop(self, config: TrainingConfig, model: Transformer2DModel, noise_scheduler: DDPMScheduler, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader, lr_scheduler) -> None:
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )

        if accelerator.is_main_process:
            if config.push_to_hub:
                logger.info("Model will be pushed to HF Hub")
                repo_name = self.get_full_repo_name(Path(config.output_dir).name)
                repo = Repository(config.output_dir, clone_from=repo_name)
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0
        total_training_time = 0.0
        flop_counter = FlopCounterMode(display=True)

        with flop_counter:
            for epoch in range(config.num_epochs):
                progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
                progress_bar.set_description(f"Epoch {epoch}")

                for step, batch in enumerate(train_dataloader):
                    start_time = time.time()
                    clean_images = batch["images"].to(accelerator.device)

                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    bs = clean_images.shape[0]

                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                    ).long()

                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                    with accelerator.accumulate(model):
                        noise_pred = model(noisy_images, timestep=timesteps, return_dict=False)[0]
                        loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(loss)

                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    step_time = time.time() - start_time
                    total_training_time += step_time
                    progress_bar.update(1)

                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "step_time": f"{step_time:.4f} seconds",
                        "flops": f"{flop_counter.get_total_flops():.4f} flops",
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    global_step += 1

                avg_step_time = total_training_time / len(train_dataloader)
                logger.info(f"Epoch {epoch} - Average step time: {avg_step_time:.4f} seconds")

            
            total_flops = flop_counter.get_total_flops()

            # Estimate MFU
            mfu = self.estimate_mfu(total_flops=total_flops, steps=global_step, total_time_seconds=total_training_time)
            logger.info(f"MFU: {mfu:.4f}")
            
            model.save_pretrained(config.output_dir)
            logger.info(f"Model saved to {config.output_dir}")
            
            # flash cuda memory
            torch.cuda.empty_cache()
            logger.info("Cuda cache flushed")
            

    def run(self) -> None:
        args = self.parse_args()
        config = TrainingConfig(**vars(args))
        logger.debug(config)

        config.dataset_name = "huggan/smithsonian_butterflies_subset"
        dataset = load_dataset(config.dataset_name, split="train")
        
        if config.debug:
            # Limit dataset to 100 samples for faster training to debug
            dataset = dataset.select(range(100))
        
        logger.debug(f"Dataset before preprocessing: {dataset}")

        self.preprocess_dataset(dataset)
        
        logger.debug(f"Dataset after preprocessing: {dataset}")

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        model = Transformer2DModel(
            num_attention_heads=16,
            attention_head_dim=88,
            in_channels=3,  # The number of channels in the input and output (e.g., 3 for RGB images)
            num_layers=12,  # Number of layers of Transformer blocks to use
            sample_size=config.image_size,  # The width of the latent images
            patch_size=None,  # Disable patch size for this model to avoid NotImplementedError
            dropout=0.1,
            cross_attention_dim=None,  # No cross attention for unconditional generation
            norm_num_groups=1,  # Set to 1 to avoid the ValueError
            norm_type='layer_norm',  # Keep using layer norm
        )

        if config.optim:
            start_compile_time = time.time()
            model = torch.compile(model)
            compile_time = time.time() - start_compile_time
            logger.info(f"Model torch compiled in {compile_time:.4f} seconds")

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        train_func = partial(self.train_loop, config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
        notebook_launcher(train_func, num_processes=1)

if __name__ == "__main__":
    trainer = DDPMPipelineTrainer(TrainingConfig())
    trainer.run()