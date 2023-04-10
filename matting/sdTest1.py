import modules.scripts as scripts
import gradio as gr
import random
import os
from PIL import Image

from modules import images
from modules.processing import process_images
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import subprocess
import imageio
import sys
import importlib.util
import shlex
import copy
import platform


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "denoising_strength": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        tag = arg[2:]

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        assert pos + 1 < len(args), f'missing argument for command line option {arg}'

        val = args[pos + 1]

        res[tag] = func(val)

        pos += 2

    return res

def run_pip(args, desc=None):
    index_url = os.environ.get('INDEX_URL', "")
    python = sys.executable
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} -i https://pypi.douban.com/simple', desc=f"Installing {desc}",
               errdesc=f"Couldn't install {desc}")


def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")



import re
class Script(scripts.Script):

    def title(self):
        return "AI福禄娃1"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        info = gr.HTML("输入prompt:")

        prompt_txt = gr.Textbox(label="(无参则延用界面的参数)", lines=2,
                                value="a full body dog\n--prompt \"a full body dog\" --negative_prompt \"five legs\" --seed 111 --steps 20 --cfg_scale 7")

        # make_a_gif.change(fn=lambda x: gr.update(visible=x), inputs=[make_a_gif], outputs=[duration])
        info3 =  gr.HTML("<br>脚本制作：CastBox--zhouzhuo")
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt],
                          outputs=[prompt_txt])

        return [prompt_txt, info3]

    def run(self, p, prompt_txt,info3):
        # allPrompt = [x.strip() for x in prompt_txt.split(',|，|。')]
        allPrompt = re.split(',|，|。',prompt_txt)
        allPrompt = [x for x in allPrompt if len(x) > 0]
        count = 2

        p.do_not_save_grid = True

        job_count = 0
        jobs = []
        for prompt in allPrompt:
            args = {"prompt": prompt}
            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        print(f"Will process {len(allPrompt)} lines in {job_count} jobs.")
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count
        images = []
        all_prompts = []
        infotexts = []

        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)


        # processed, processed_images = process(p, prompt_txt)
        #
        # p.prompt_for_display = processed.prompt
        # processed_images_flattened = []
        #
        # for row in processed_images:
        #     processed_images_flattened += row
        #
        # if len(processed_images_flattened) == 1:
        #     processed.images = processed_images_flattened
        # else:
        #     processed.images = [images.image_grid(processed_images_flattened, rows=p.batch_size * p.n_iter)] \
        #                        + processed_images_flattened
        # return processed

def process(p, prompt_txt):

    first_processed = None
    processed_images = []
    for i in range(p.batch_size * p.n_iter):
        processed_images.append([])
        p.prompt = prompt_txt
        processed = process_images(p)
        if first_processed is None:
            first_processed = processed

        for i, img in enumerate(processed.images):
            processed_images[i].append(img)

    print("python")
    # lines = [x.strip() for x in prompt_txt.splitlines()]
    # lines = [x for x in lines if len(x) > 0]

    p.do_not_save_grid = True

    # for i in range(p.batch_size * p.n_iter):
    #     processed_images.append([])
    # p.prompt_for_display = prompt_txt
    # p.prompt = prompt_txt
    # processed = process_images(p)

    # job_count = 0
    # jobs = []

    # for line in lines:
    #     if "--" in line:
    #         try:
    #             args = cmdargs(line)
    #         except Exception:
    #             print(f"Error parsing line [line] as commandline:", file=sys.stderr)
    #             args = {"prompt": line}
    #     else:
    # args = {"prompt": prompt_txt}

    # n_iter = args.get("n_iter", 1)
        # if n_iter != 1:
        # job_count += n_iter
        # else:
        # job_count += 1
    # job_count += 1
    # jobs.append(args)

    # state.job_count = job_count * p.n_iter

    # for n, args in enumerate(jobs):
    #     state.job = f"{state.job_no + 1} out of {state.job_count}"
    #
    #     copy_p = copy.copy(p)
    #     for k, v in args.items():
    #         setattr(copy_p, k, v)
    #
    #     processed = process_images(copy_p)
    #
    #     if first_processed is None:
    #         first_processed = processed
    #
    #     for i, img in enumerate(processed.images):
    #             processed_images[i].append(img)
    # for i, img in enumerate(processed.images):
    #     processed_images[i].append(img)
    return first_processed, processed_images
