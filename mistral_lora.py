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

IMAGE = "quay.io/lukemarsden/axolotl:v0.0.1"
PROMPT = "If I put up a hammock hung between opposite sides of a round lake, go to sleep in the hammock and fall out, where will I land?"

async def main():

    print("Spawning docker socket forwarder...")
    p = subprocess.Popen(["socat", "TCP-LISTEN:12345,reuseaddr,fork,bind=172.17.0.1", "UNIX-CONNECT:/var/run/docker.sock"])
    time.sleep(1)
    print("Done!")

    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        try:
            python = (
                client
                    .container()
                    .from_("docker:latest") # TODO: use '@sha256:...'
                    # break cache
                    # .with_env_variable("BREAK_CACHE", str(time.time()))
                    .with_entrypoint("/usr/local/bin/docker")
                    .with_exec(["-H", "tcp://172.17.0.1:12345",
                        "run", "-i", "--rm", "--gpus", "all",
                        IMAGE,
                        "bash", "-c", "echo "{PROMPT}" |python -u -m axolotl.cli.inference examples/mistral/qlora-instruct.yml",
                    ])
            )
            # execute
            err = await python.stderr()
            out = await python.stdout()
            # print stderr
            print(f"Question: {PROMPT}\n\nAnswer: {out}")
        except Exception as e:
            import pdb; pdb.set_trace()
            print(f"error: {e}")

    p.terminate()

anyio.run(main)
