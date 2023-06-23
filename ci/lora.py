import sys

import anyio
import dagger
import os
import time
import subprocess
import urllib.request
import zipfile
import textwrap

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
IMAGE = "quay.io/lukemarsden/lora:v0.0.2"
ASSETS = [
    "coke",
    "dagger",
    "docker",
    "kubernetes",
    "nike",
    "vision-pro",
]
PROMPTS = {
    "mug": "coffee mug with logo on it, in the style of <s1><s2>",
    "mug2": "coffee mug with brand logo on it, in the style of <s1><s2>",
    "mug3": "coffee mug with brand logo on it, in the style of <s1><s2>, 50mm portrait photography, hard rim lighting photography, merchandise",
    "tshirt": "woman torso wearing tshirt with <s1><s2> logo, 50mm portrait photography, hard rim lighting photography, merchandise",
}

async def main():

    print("Spawning docker socket forwarder...")
    p = subprocess.Popen(["socat", "TCP-LISTEN:12345,reuseaddr,fork,bind=172.17.0.1", "UNIX-CONNECT:/var/run/docker.sock"])
    time.sleep(1)
    print("Done!")

    config = dagger.Config(log_output=sys.stdout)

    # create output directory on the host
    output_dir = os.path.join(os.getcwd(), "output")

    print("=============================")
    print(f"OUTPUT DIRECTORY: {output_dir}")
    print("=============================")
    os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "downloads"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "loras"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inference"), exist_ok=True)

    for brand in ASSETS:
        # http download storage.googleapis.com/dagger-assets/dagger.zip
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/dagger-assets/{brand}.zip",
            os.path.join(output_dir, "downloads", f"{brand}.zip"),
        )
        # unzip with zipfile module
        with zipfile.ZipFile(os.path.join(output_dir, "downloads", f"{brand}.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_dir, "assets"))

    # write hello.txt to tmp directory
    with open(os.path.join(output_dir, "hello.txt"), "w") as f:
        f.write("Hello from Dagger!")

    # train the loras
    for brand in ASSETS:
        # initialize Dagger client - no parallelism here
        async with dagger.Connection(config) as client:
            # fine tune lora
            try:
                python = (
                    client
                        .container()
                        .from_("docker:latest")
                        # break cache
                        # .with_env_variable("BREAK_CACHE", str(time.time()))
                        .with_entrypoint("/usr/local/bin/docker")
                        .with_exec(["-H", "tcp://172.17.0.1:12345",
                            "run", "-i", "--rm", "--gpus", "all",
                            "-v", os.path.join(output_dir, "assets", brand)+":/input",
                            "-v", os.path.join(output_dir, "loras", brand)+":/output",
                            IMAGE,
                            'lora_pti',
                            f'--pretrained_model_name_or_path={MODEL_NAME}',
                            '--instance_data_dir=/input', '--output_dir=/output',
                            '--train_text_encoder', '--resolution=512',
                            '--train_batch_size=1',
                            '--gradient_accumulation_steps=4', '--scale_lr',
                            '--learning_rate_unet=1e-4',
                            '--learning_rate_text=1e-5', '--learning_rate_ti=5e-4',
                            '--color_jitter', '--lr_scheduler="linear"',
                            '--lr_warmup_steps=0',
                            '--placeholder_tokens="<s1>|<s2>"',
                            '--use_template="style"', '--save_steps=100',
                            '--max_train_steps_ti=1000',
                            '--max_train_steps_tuning=1000',
                            '--perform_inversion=True', '--clip_ti_decay',
                            '--weight_decay_ti=0.000', '--weight_decay_lora=0.001',
                            '--continue_inversion', '--continue_inversion_lr=1e-4',
                            '--device="cuda:0"', '--lora_rank=1'
                        ])
                )
                # execute
                err = await python.stderr()
                out = await python.stdout()
                # print stderr
                print(f"Hello from Dagger, fine tune LoRA on {brand}: {out}{err}")
            except Exception as e:
                import pdb; pdb.set_trace()
                print(f"error: {e}")

    async with dagger.Connection(config) as client:
        for brand in ASSETS:
            for key, prompt in PROMPTS.items():
                for seed in range(10):
                    # inference!
                    python = (
                        client
                            .container()
                            .from_("docker:latest")
                            .with_entrypoint("/usr/local/bin/docker")
                            .with_exec(["-H", "tcp://172.17.0.1:12345",
                                "run", "-i", "--rm", "--gpus", "all",
                                "-v", os.path.join(output_dir, "loras", brand)+":/input",
                                "-v", os.path.join(output_dir, "inference", brand)+":/output",
                                IMAGE,
                                'python3',
                                '-c',
                                # dedent
                                textwrap.dedent(f"""
                                    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
                                    import torch
                                    from lora_diffusion import tune_lora_scale, patch_pipe

                                    model_id = "{MODEL_NAME}"

                                    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
                                        "cuda"
                                    )
                                    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

                                    prompt = "{prompt}"
                                    seed = {seed}
                                    torch.manual_seed(seed)

                                    patch_pipe(
                                        pipe,
                                        "/input/final_lora.safetensors",
                                        patch_text=True,
                                        patch_ti=True,
                                        patch_unet=True,
                                    )

                                    coeff = 0.5
                                    tune_lora_scale(pipe.unet, coeff)
                                    tune_lora_scale(pipe.text_encoder, coeff)

                                    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
                                    image.save(f"/output/{key}-{{seed}}.jpg")
                                    image
                                    """)
                            ])
                    )
                    # execute
                    err = await python.stderr()
                    out = await python.stdout()
                    # print stderr
                    print(f"Hello from Dagger, inference {brand}, prompt: {prompt} and {out}{err}")

    p.terminate()

anyio.run(main)