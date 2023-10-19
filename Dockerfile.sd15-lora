FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Install lora and pre-cache stable diffusion 1.5 model to avoid re-downloading
# it for every inference.
#
# NB: diffusers downgrade is because of https://github.com/cloneofsimo/lora/issues/231
RUN apt-get update -y && apt-get install -y python3 python3-pip git unzip && \
    pip install git+https://github.com/cloneofsimo/lora.git@v0.1.7 && \
    pip install diffusers==0.14 && \
    pip install accelerate==0.20.3 && \
    python3 -c "from diffusers import StableDiffusionPipeline; model_id = 'runwayml/stable-diffusion-v1-5'; StableDiffusionPipeline.from_pretrained(model_id)"
