import sys

import anyio
import dagger
import os
import time
import subprocess


async def main():

    config = dagger.Config(log_output=sys.stdout)

    # initialize Dagger client
    async with dagger.Connection(config) as client:
        # get version
        python = (
            client
                .container()
                .from_("docker:latest")
                .with_mounted_file("/var/run/docker.sock", client.host().file("/var/run/docker.sock"))
                .with_env_variable("BREAK_CACHE", str(time.time()))
                .with_exec(["docker",
                    "run", "--rm", "--runtime=nvidia", "--gpus", "all", "nvidia/cuda:11.6.2-base-ubuntu20.04",
                    "nvidia-smi"])
        )

        # execute
        version = await python.stdout()

    # print output
    print(f"Hello from Dagger and {version}")

anyio.run(main)
