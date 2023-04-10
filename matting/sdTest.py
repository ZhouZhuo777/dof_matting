import modules.scripts as scripts
import gradio as gr
import random
import os
from PIL import Image

from modules import images
from modules.processing import process_images
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


def process(p, prompt_txt):
    first_processed = None
    processed_images = []

    for i in range(p.batch_size * p.n_iter):
        processed_images.append([])
    # p.cfg_scale = 5
        p.prompt = prompt_txt
        processed = process_images(p)

        if first_processed is None:
            first_processed = processed

        for i, img in enumerate(processed.images):
            processed_images[i].append(img)

    return first_processed, processed_images


class Script(scripts.Script):

    def title(self):
        return "zhouzhuoTest"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        # info = gr.HTML("输入prompt")
        prompt_txt = gr.Textbox(label="输入prompt")
        info2 = gr.HTML("<br>脚本制作：CastBox--zhouzhuo")
        return [prompt_txt,  info2]

    def run(self, p, prompt_txt,info2):
        # if p.seed == -1:
        #     p.seed = int(random.randrange(4294967294))

        p.do_not_save_grid = True
        # prompt1 = p.prompt
        processed, processed_images = process(p,prompt_txt)

        p.prompt_for_display = processed.prompt = prompt1
        processed_images_flattened = []

        for row in processed_images:
            processed_images_flattened += row

        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened, rows=p.batch_size * p.n_iter)] \
                               + processed_images_flattened
        return processed