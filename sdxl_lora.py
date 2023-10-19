import sys

import anyio
import dagger
import os
import time
import subprocess
import urllib.request
import zipfile
import textwrap
import yaml

# Load from config.yml
config = yaml.load(open("config_sdxl.yml", "r"), Loader=yaml.FullLoader)

IMAGE = config.get("container_image", "quay.io/lukemarsden/sd-scripts:v0.0.3")
ASSETS = config.get("brands", [
    # "coke",
    "dagger",
    # "docker",
    # "kubernetes",
    # "nike",
    # "vision-pro",
])
PROMPTS = config.get("prompts", {
    "mug": "coffee mug with dagger logo on it",
    "mug2": "coffee mug with astronauts on mars on it holding a map",
    "mug3": "coffee mug with dagger logo on it, 50mm portrait photography, hard rim lighting photography, merchandise",
    "tshirt": "woman torso wearing dagger logo tshirt, 50mm portrait photography, hard rim lighting photography, merchandise",
})
NUM_IMAGES = config.get("num_images", 10)
URL_PREFIX = config.get("url_prefix", "https://storage.googleapis.com/dagger-assets/")
COEFF = config.get("finetune_weighting", 0.8)

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
        # http download storage.googleapis.com/dagger-assets/sdxl_dagger.zip (the sdxl_prefixed ones have .txt file captions in there)
        urllib.request.urlretrieve(
            URL_PREFIX + "sdxl_" + brand + ".zip",
            os.path.join(output_dir, "downloads", f"{brand}.zip"),
        )
        # unzip with zipfile module
        with zipfile.ZipFile(os.path.join(output_dir, "downloads", f"{brand}.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_dir, "assets"))

    open(os.path.join(output_dir, "config.toml"), "w").write("""[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 1024                           # Training resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = '/input'                      # Specify the folder containing the training images
  caption_extension = '.txt'                # Caption file extension; change this if using .txt
  num_repeats = 10                          # Number of repetitions for training images
""")

    # train the loras
    for brand in ASSETS:
        # initialize Dagger client - no parallelism here
        async with dagger.Connection(config) as client:
            # fine tune lora
            try:
                args = ["-H", "tcp://172.17.0.1:12345",
                            "run", "-i",
                            "--rm", "--gpus", "all",
                            "-v", os.path.join(output_dir, "config.toml")+":/config.toml",
                            "-v", os.path.join(output_dir, "assets", brand)+":/input",
                            "-v", os.path.join(output_dir, "loras", brand)+":/output",
                            IMAGE,

                            "accelerate", "launch", "--num_cpu_threads_per_process", "1", "sdxl_train_network.py",
                                "--pretrained_model_name_or_path=./sdxl/sd_xl_base_1.0.safetensors",
                                "--dataset_config=/config.toml",
                                "--output_dir=/output",
                                "--output_name=lora",
                                "--save_model_as=safetensors",
                                "--prior_loss_weight=1.0",
                                "--max_train_steps=400",
                                "--vae=madebyollin/sdxl-vae-fp16-fix",
                                "--learning_rate=1e-4",
                                "--optimizer_type=AdamW8bit",
                                "--xformers",
                                "--mixed_precision=fp16",
                                "--cache_latents",
                                "--gradient_checkpointing",
                                "--save_every_n_epochs=1",
                                "--network_module=networks.lora",

                        ]
                print("RUNNING:", " ".join(args))
                python = (
                    client
                        .container()
                        .from_("docker:latest") # TODO: use '@sha256:...'
                        # break cache
                        .with_env_variable("BREAK_CACHE", brand)
                        # .with_entrypoint("/usr/local/bin/docker")
                        .with_entrypoint("/bin/sh")
                        .with_exec(["-c", "docker " + " ".join(args)])
                        # .with_exec(args)
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
                for seed in range(NUM_IMAGES):
                    # inference!
                    python = (
                        client
                            .container()
                            .from_("docker:latest")
                            # .with_env_variable("BREAK_CACHE", str(time.time()))
                            .with_entrypoint("/usr/local/bin/docker")
                            .with_exec(["-H", "tcp://172.17.0.1:12345",
                                "run",
                                "-i", "--rm", "--gpus", "all",
                                "-v", os.path.join(output_dir, "loras", brand)+":/input",
                                "-v", os.path.join(output_dir, "inference", brand)+":/output",
                                IMAGE,

                                "accelerate", "launch", "--num_cpu_threads_per_process", "1", "sdxl_minimal_inference.py",
                                    "--ckpt_path=sdxl/sd_xl_base_1.0.safetensors",
                                    f'--lora_weights="/input/lora.safetensors;{COEFF}"', 
                                    f'--prompt="{prompt}"',
                                    "--output_dir=/output",
                            ])
                    )
                    # execute
                    err = await python.stderr()
                    out = await python.stdout()
                    # print stderr
                    print(f"Hello from Dagger, inference {brand}, prompt: {prompt} and {out}{err}")

    p.terminate()

anyio.run(main)
