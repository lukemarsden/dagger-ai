import sys

import anyio
import dagger
import os
import time
import subprocess


async def main():

    print("Spawning docker socket forwarder...")
    p = subprocess.Popen(["socat", "TCP-LISTEN:12345,reuseaddr,fork,bind=172.17.0.1", "UNIX-CONNECT:/var/run/docker.sock"])
    time.sleep(1)
    print("Done!")

    config = dagger.Config(log_output=sys.stdout)

    # initialize Dagger client
    async with dagger.Connection(config) as client:
        # get version
        python = (
            client
                .container()
                .from_("docker:latest")
                .with_env_variable("BREAK_CACHE", str(time.time()))
                .with_exec(["docker", "-H", "tcp://172.17.0.1:12345",
                    "run", "--rm", "--gpus", "all", "nvidia/cuda:11.6.2-base-ubuntu20.04",
                    "nvidia-smi"])
        )

        # execute
        version = await python.stdout()

    # print output
    print(f"Hello from Dagger and {version}")

    p.terminate()

anyio.run(main)
