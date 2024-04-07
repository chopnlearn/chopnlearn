import argparse
import itertools
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataset import PROMPT, STATE_TO_TEXT, ObjectStateDataset
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import concat_h, import_model_class_from_model_name_or_path


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Stable Diffusion Args
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--train_split_json",
        type=str,
        default='train_split.json',
        help="Path to train split json file.",
    )
    parser.add_argument(
        "--val_split_json",
        type=str,
        default='test_split.json',
        help="Path to val split json file.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--resume_if_available",
        action="store_true",
        help="Resume from the last checkpoint.",
    )

    # Training Hyperparams
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_textual_inversion",
        type=float,
        default=0.0,
        help="Initial learning rate (after the potential warmup period) to use for the textual inversion.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--exclude_states",
        nargs="*",
        type=str,
        default=['p'],
        help="States to exclude from the datasets",
    )

    parser.add_argument(
        "--cameras",
        nargs="*",
        type=str,
        default=["cam1", "cam2", "cam3", "cam4"],
        help="Cameras to use for the dataset",
    )

    # Logging Hyperparams
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("Logg directory. Will default to logs"),
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="defaule",
        help=("experiment name"),
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help=("wandb entity"),
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../image_dataset/images",
        help="Dataset, the path to the image folder."
    )

    parser.add_argument(
        "--no_cst",
        action="store_true",
        help=("Whether to not use the camera specific transforms"),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.data_dir = os.path.dirname(args.data)
    args.data_folder = os.path.basename(args.data)

    return args


def collate_fn(examples):
    pixel_values = torch.stack([example["instance_images"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["instance_prompt_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def main(args):
    logging_dir = Path(args.logging_dir) / args.exp_name
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            logging_dir=logging_dir / 'wandb',
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            logging_dir=logging_dir,
        )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.wandb:
        wandb_kargs = {"name": args.exp_name, "settings": {"console": "off"}}
        if args.wandb_entity:
            wandb_kargs["entity"] = args.wandb_entity

        # Handle the logging
        if accelerator.is_main_process:
            os.makedirs(logging_dir, exist_ok=True)
            accelerator.init_trackers(
                "chopnlearn",
                config=vars(args),
                init_kwargs={"wandb": wandb_kargs},
            )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = ObjectStateDataset(
        data_root=args.data_dir,
        tokenizer=tokenizer,
        data_folder=args.data_folder,
        size=args.resolution,
        camera_specific_transform=not args.no_cst,
        split=args.train_split_json,
        exclude_states=args.exclude_states,
        cameras=args.cameras,
    )

    val_dataset = ObjectStateDataset(
        data_root=args.data_dir,
        tokenizer=tokenizer,
        data_folder=args.data_folder,
        size=args.resolution,
        camera_specific_transform=not args.no_cst,
        split=args.val_split_json,
        exclude_states=args.exclude_states,
        cameras=args.cameras,
    )

    model_params = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )

    param_groups = []

    if args.learning_rate > 1e-8:
        assert not args.train_text_encoder
        param_groups.append({"params": model_params, "lr": args.learning_rate})
    else:
        unet.requires_grad_(False)

    if args.learning_rate_textual_inversion > 1e-8:
        # if we are using textual inversion, modify the tokenizer
        # add new tokens for the objects

        obj_placeholders = sorted(
            [f'*_obj_{obj}' for obj in train_dataset.unique_objects]
        )
        state_placeholders = sorted(
            [f'*_state_{state}' for state in train_dataset.unique_states]
        )
        num_added_tokens = tokenizer.add_tokens(obj_placeholders + state_placeholders)
        assert num_added_tokens == len(obj_placeholders + state_placeholders)

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        placeholder_token_ids = []

        for obj_ph in obj_placeholders:
            obj_ph_id = tokenizer.convert_tokens_to_ids(obj_ph)
            placeholder_token_ids.append(obj_ph_id)
            init_token = obj_ph.split('_')[-1]
            # assert init_token in tokenizer.get_vocab(), f'{init_token} not in tokenizer vocab'
            if (
                init_token not in tokenizer.get_vocab()
                and f'{init_token}</w>' not in tokenizer.get_vocab()
            ):
                print(f'{init_token} not in tokenizer vocab, using produce instead')
                init_token = 'produce'
            init_token_id = tokenizer.convert_tokens_to_ids(init_token)
            token_embeds[obj_ph_id] = token_embeds[init_token_id]

        for state_ph in state_placeholders:
            state_ph_id = tokenizer.convert_tokens_to_ids(state_ph)
            placeholder_token_ids.append(state_ph_id)
            init_token = STATE_TO_TEXT[state_ph.split('_')[-1]].split(' ')[0]
            # assert init_token in tokenizer.get_vocab() or f'{init_token}</w>' in tokenizer.get_vocab(), f'{init_token} not in tokenizer vocab'
            if (
                init_token not in tokenizer.get_vocab()
                and f'{init_token}</w>' not in tokenizer.get_vocab()
            ):
                if init_token == 'julienne':
                    init_token = 'strip'
                    print(f'julienne not in tokenizer vocab, using strip instead')
                else:
                    raise ValueError(f'{init_token} not in tokenizer vocab')
            init_token_id = tokenizer.convert_tokens_to_ids(init_token)
            token_embeds[state_ph_id] = token_embeds[init_token_id]

        text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

        assert torch.allclose(
            token_embeds, text_encoder.get_input_embeddings().weight.data
        )

        param_groups.append(
            {
                "params": text_encoder.get_input_embeddings().parameters(),
                "lr": args.learning_rate_textual_inversion,
            }
        )

        new_state_to_text = {
            state: f'*_state_{state}' for state in train_dataset.unique_states
        }
        new_object_to_text = {
            obj: f'*_obj_{obj}' for obj in train_dataset.unique_objects
        }

        train_dataset = ObjectStateDataset(
            data_root=args.data_dir,
            tokenizer=tokenizer,
            data_folder=args.data_folder,
            size=args.resolution,
            camera_specific_transform=not args.no_cst,
            split=args.train_split_json,
            exclude_states=args.exclude_states,
            state_to_text=new_state_to_text,
            object_to_text=new_object_to_text,
            cameras=args.cameras,
        )

        val_dataset = ObjectStateDataset(
            data_root=args.data_dir,
            tokenizer=tokenizer,
            data_folder=args.data_folder,
            size=args.resolution,
            camera_specific_transform=not args.no_cst,
            split=args.val_split_json,
            exclude_states=args.exclude_states,
            state_to_text=new_state_to_text,
            object_to_text=new_object_to_text,
            cameras=args.cameras,
        )

    else:
        new_state_to_text = {
            state: STATE_TO_TEXT[state] for state in train_dataset.unique_states
        }
        new_object_to_text = {obj: obj for obj in train_dataset.unique_objects}

    assert val_dataset.unique_states.issubset(train_dataset.unique_states)
    assert val_dataset.unique_objects.issubset(train_dataset.unique_objects)
    assert not val_dataset.unique_object_state_pairs.intersection(
        train_dataset.unique_object_state_pairs
    )

    # define an optimizer that has a different learning rate for the model_params and the text_encoder.get_input_embeddings().parameters()
    optimizer = optimizer_class(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=4,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.sample_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=4,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    last_checkpoint = None
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    if args.learning_rate_textual_inversion > 1e-8:
        orig_embeds_params = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
            safety_checker=None,
        ).to(accelerator.device)

        train_vis_prompts = np.random.choice(
            [
                PROMPT.format(new_object_to_text[obj], new_state_to_text[st])
                for obj, st in zip(
                    train_dataset.object_labels, train_dataset.state_labels
                )
            ],
            args.sample_batch_size,
        ).tolist()
        val_vis_prompts = np.random.choice(
            [
                PROMPT.format(new_object_to_text[obj], new_state_to_text[st])
                for obj, st in zip(val_dataset.object_labels, val_dataset.state_labels)
            ],
            args.sample_batch_size,
        ).tolist()

        logger.info(f"  train prompts = {','.join(train_vis_prompts)}")
        logger.info(f"  val prompts = {','.join(val_vis_prompts)}")
        save_path = os.path.join(logging_dir, f"{global_step:04d}_val.png")
        images = pipeline(
            prompt=val_vis_prompts,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=50,
            num_images_per_prompt=1,
        ).images

        if args.wandb:
            accelerator.log(
                {"val_gen": [wandb.Image(image) for image in images]}, step=global_step
            )

        concat_h(*images, pad=4).save(save_path)
        save_path = os.path.join(logging_dir, f"{global_step:04d}_train.png")
        images = pipeline(
            prompt=train_vis_prompts,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=50,
            num_images_per_prompt=1,
        ).images
        concat_h(*images, pad=4).save(save_path)

        if args.wandb:
            accelerator.log(
                {"train_gen": [wandb.Image(image) for image in images]}, step=global_step
            )

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder or args.learning_rate_textual_inversion > 1e-8:
            text_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.learning_rate_textual_inversion > 1e-8:
                    index_no_updates = torch.tensor(
                        [
                            i
                            for i in range(len(tokenizer))
                            if i not in placeholder_token_ids
                        ]
                    )
                    index_no_updates = index_no_updates.to(text_encoder.device)

                    # pdb.set_trace()

                    with torch.no_grad():
                        accelerator.unwrap_model(
                            text_encoder
                        ).get_input_embeddings().weight.data[
                            index_no_updates
                        ] = orig_embeds_params[
                            index_no_updates
                        ]

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if args.wandb:
                    accelerator.log({"train_loss": train_loss}, step=global_step)

                train_loss = 0.0

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision,
                            safety_checker=None,
                        ).to(accelerator.device)
                        save_path = os.path.join(
                            logging_dir, f"{global_step:04d}_val.png"
                        )
                        images = pipeline(
                            prompt=val_vis_prompts,
                            height=args.resolution,
                            width=args.resolution,
                            num_inference_steps=50,
                            num_images_per_prompt=1,
                        ).images

                        if args.wandb:
                            accelerator.log(
                                {"val_gen": [wandb.Image(image) for image in images]},
                                step=global_step,
                            )

                        images = concat_h(*images, pad=4)
                        images.save(save_path)

                        save_path = os.path.join(
                            logging_dir, f"{global_step:04d}_train.png"
                        )
                        images = pipeline(
                            prompt=train_vis_prompts,
                            height=args.resolution,
                            width=args.resolution,
                            num_inference_steps=50,
                            num_images_per_prompt=1,
                        ).images

                        if args.wandb:
                            accelerator.log(
                                {"train_gen": [wandb.Image(image) for image in images]},
                                step=global_step,
                            )

                        images = concat_h(*images, pad=4)
                        images.save(save_path)

                        # Checkpoint
                        ckpt_path = os.path.join(
                            logging_dir, f'checkpoint-{global_step}'
                        )
                        pipeline.save_pretrained(ckpt_path)
                        if last_checkpoint is not None and os.path.exists(
                            last_checkpoint
                        ):
                            shutil.rmtree(last_checkpoint)
                        last_checkpoint = ckpt_path

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if args.wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        # Start validation
        logger.info("Starting validation")
        val_progress_bar = tqdm(
            range(len(val_dataloader)), disable=not accelerator.is_local_main_process
        )
        val_progress_bar.set_description("Steps")
        unet.eval()
        text_encoder.eval()
        with torch.no_grad():
            val_loss = []
            for step, batch in enumerate(val_dataloader):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.sample_batch_size))
                val_loss.append(avg_loss.mean().item())
                val_progress_bar.update(1)

                logs = {"val_loss": val_loss[-1]}
                val_progress_bar.set_postfix(**logs)

            val_loss = np.mean(val_loss)

            if args.wandb:
                accelerator.log({"val_loss": val_loss}, step=global_step)
            logger.info(f"Validation loss: {val_loss}")

            accelerator.wait_for_everyone()

    # Log all val prompts
    for i, (obj, st) in enumerate(val_dataset.unique_object_state_pairs):
        logger.info(f'{i}/{len(val_dataset.unique_object_state_pairs)} - {obj} {st}')
        prompt = PROMPT.format(new_object_to_text[obj], new_state_to_text[st])
        save_path = os.path.join(logging_dir, f"{global_step:04d}_val_{obj}_{st}.png")
        logger.info(f"  Saving {save_path}")
        images = pipeline(
            prompt=prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=50,
            num_images_per_prompt=4,
        ).images
        concat_h(*images, pad=4).save(save_path)

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
        )
        pipeline.save_pretrained(os.path.join(logging_dir, 'checkpoint'))

    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
