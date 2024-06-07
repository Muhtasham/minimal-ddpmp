import math
import os
import torch
import torch.nn.functional as F
import argparse

from dataclasses import dataclass, asdict
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset
from diffusers import UNet2DModel
from PIL import Image, ExifTags
from diffusers import DDPMScheduler
from loguru import logger
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import notebook_launcher
from typing import Tuple, List, Dict, Optional
from functools import partial
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, record_function, ProfilerActivity
from rich import print 

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
    mixed_precision: str = "fp16"
    output_dir: str = "ddpm-butterflies-128"
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0

class DDPMPipelineTrainer:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the DDPMPipelineTrainer with the given configuration.

        Args:
            config (TrainingConfig): The configuration for training.
        """
        self.config = config

    def parse_args(self) -> argparse.Namespace:
        """
        Parses command-line arguments.

        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Training script for DDPM")
        parser.add_argument("--image_size", type=int, default=128, help="Generated image resolution")
        parser.add_argument("--train_batch_size", type=int, default=64, help="Training batch size")
        parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
        parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
        parser.add_argument("--save_image_epochs", type=int, default=10, help="Save image every n epochs")
        parser.add_argument("--save_model_epochs", type=int, default=30, help="Save model every n epochs")
        parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision (no, fp16)")
        parser.add_argument("--output_dir", type=str, default="ddpm-butterflies-128", help="Output directory")
        parser.add_argument("--push_to_hub", action='store_true', help="Upload the saved model to the HF Hub")
        parser.add_argument("--hub_private_repo", action='store_true', help="Make the HF Hub repo private")
        parser.add_argument("--overwrite_output_dir", action='store_true', help="Overwrite the old model")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")

        return parser.parse_args()

    def preprocess_dataset(self, dataset) -> None:
        """
        Preprocesses the dataset by resizing, normalizing, and augmenting images.

        Args:
            dataset: The dataset to preprocess.
        """

        # The images are all different sizes, so we'll need to preprocess them first
        preprocess = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples: Dict[str, List[Image.Image]]) -> Dict[str, List[torch.Tensor]]:
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)
        logger.info("pre-processing done")

    def make_grid(self, images: List[Image.Image], rows: int, cols: int) -> Image.Image:
        """
        Creates a grid of images.

        Args:
            images (List[Image.Image]): List of images to arrange in a grid.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.

        Returns:
            Image.Image: The resulting grid image.
        """
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, epoch: int, pipeline: DDPMPipeline) -> None:
        """
        Evaluates the model by generating and saving sample images.

        Args:
            epoch (int): Current epoch number.
            pipeline (DDPMPipeline): The DDPM pipeline for image generation.
        """
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
        """
        Gets the full repository name on the Hugging Face Hub.

        Args:
            model_id (str): The model ID.
            organization (Optional[str]): The organization name.
            token (Optional[str]): The Hugging Face Hub token.

        Returns:
            str: The full repository name.
        """
        if token is None:
            token = HfFolder.get_token()
        if organization is None:
            username = whoami(token)["name"]
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    def train_loop(self, config: TrainingConfig, model: UNet2DModel, noise_scheduler: DDPMScheduler, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader, lr_scheduler) -> None:
        """
        The main training loop.

        Args:
            config (TrainingConfig): The configuration for training.
            model (UNet2DModel): The model to train.
            noise_scheduler (DDPMScheduler): The noise scheduler.
            optimizer (torch.optim.Optimizer): The optimizer.
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.
            lr_scheduler: The learning rate scheduler.
        """
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.push_to_hub:
                repo_name = self.get_full_repo_name(Path(config.output_dir).name)
                repo = Repository(config.output_dir, clone_from=repo_name)
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        flop_counter = FlopCounterMode(model, depth=1, display=True)

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the
        # objects in the same order we gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0
        
        with torch.profiler.profile(
            activities=[
                #torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/unet2d'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as prof:        
            # Now we train the model
            for epoch in range(config.num_epochs):
                progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
                progress_bar.set_description(f"Epoch {epoch}")
                #prof.start()
                for step, batch in enumerate(train_dataloader):
                    #prof.step()
                    clean_images = batch["images"]
                    # Sample noise to add to the images
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    bs = clean_images.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                    ).long()

                    # Add noise to the clean images according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                    with accelerator.accumulate(model):
                        # Predict the noise residual
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                        loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(loss)

                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    progress_bar.update(1)
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    global_step += 1
                
                # After each epoch we optionally sample some demo images with evaluate() and save the model
                if accelerator.is_main_process:
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                    if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                        #self.evaluate(epoch, pipeline)
                        logger.info("skipping eval")

                    if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                        if config.push_to_hub:
                            repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                        else:
                            pipeline.save_pretrained(config.output_dir)
            #prof.stop()

        #print(flop_counter.get_table())
        #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
        #print(prof.key_averages())
        logger.info("exporting profile")
        prof.export_chrome_trace("trace.json")

    def run(self) -> None:
        """
        Runs the training process.
        """
        args = self.parse_args()
        config = TrainingConfig(**vars(args))
        logger.debug(config)

        config.dataset_name = "huggan/smithsonian_butterflies_subset"
        dataset = load_dataset(config.dataset_name, split="train")
        logger.debug(dataset)

        self.preprocess_dataset(dataset)
        logger.debug(dataset)
        
        #  wrap the dataset in a DataLoader for training 
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

        model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        #model = torch.compile(model)
        #logger.info("model torch compiled")

        sample_image = dataset[0]["images"].unsqueeze(0)
        logger.debug("Input shape:", sample_image.shape)
        logger.debug("Output shape:", model(sample_image, timestep=0).sample.shape)

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        noise = torch.randn(sample_image.shape)
        timesteps = torch.LongTensor([50])
        noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
        
        # The training objective of the model is to predict the noise added to the image. The loss at this step can be calculated by below
        noise_pred = model(noisy_image, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

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
