import modules.scripts as scripts
import gradio as gr
import random
import os
from PIL import Image

from modules import images
from modules.processing import process_images
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


def process(p, sex, trueyear, currentyear,lastyear,n_images):
    first_processed = None
    processed_images = []

    for i in range(p.batch_size * p.n_iter):
        processed_images.append([])
    
    state.job_count = n_images * p.n_iter

    for i in range(n_images):
        state.job = f"interpolation: {i + 1} out of {n_images}"
        #interpolation = 0.5 if n_images == 1 else i / (n_images - 1)
        #p.prompt = f"{prompt1} :{1 - interpolation} AND {prompt2} :{interpolation}"
        p.cfg_scale=5
        
        frameyear = currentyear if n_images == 1 else currentyear+(lastyear-currentyear)*i / (n_images - 1)		
		
        if frameyear<trueyear:
            p.denoising_strength=0.2+0.5*(trueyear-frameyear)/trueyear
        if frameyear>trueyear:
            p.denoising_strength=0.2+0.5*(frameyear-trueyear)/(1+abs(trueyear-lastyear))
        if frameyear==trueyear:
            p.denoising_strength=0.2
        prompt1=""
        if sex=="男":
            if frameyear<8:
                prompt1="a little baby,"+str(frameyear)+" years old,black hair"   
            if frameyear>=8 and frameyear<25:
                prompt1="a boy,"+str(frameyear)+" years old,black hair" 
            if frameyear>=25 and frameyear<50:
                prompt1="a man,"+str(frameyear)+" years old,black hair"    
            if frameyear>=50 and frameyear<=100:
                prompt1="a old man,"+str(frameyear)+" years old,white hair"    				
        else:
            if frameyear<8:
                prompt1="a little baby,"+str(frameyear)+" years old,black hair"    
            if frameyear>=8 and frameyear<25:
                prompt1="a girl,"+str(frameyear)+" years old,black hair" 
            if frameyear>=25 and frameyear<50:
                prompt1="a woman,"+str(frameyear)+" years old,black hair"    
            if frameyear>=50 and frameyear<=100:
                prompt1="a old lady,"+str(frameyear)+" years old,white hair" 
        p.prompt = prompt1
        processed = process_images(p)
        
        if first_processed is None:
            first_processed = processed

        for i, img in enumerate(processed.images):
            processed_images[i].append(img)

    return first_processed, processed_images


class Script(scripts.Script):

    def title(self):
        return "相片时光机"


    def show(self, is_img2img):
        return is_img2img


    def ui(self, is_img2img):
        
        info = gr.HTML("prompt,cfg scale,denoising strength在此脚本中无效<br><br><br>")
	
        sex = gr.Dropdown(["男","女"], label="选择人像真实性别", value="男")	
        trueyear = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="预估人像真实年龄（单图模式下适当调大该值可强化幼龄照）")
        
        
        currentyear = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="选择生成某个年龄的照片（动图模式下的起始年龄）")

        lastyear = gr.Slider(minimum=1, maximum=100, step=1, value=100, label="选择生成图片的终止年龄（单图模式下适当调小该值可强化老龄照）", visible=True)
        n_images = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="设置总帧数（各帧均匀分布于所选年龄段，每年最多一帧多出忽略）", visible=True)
        make_a_gif = gr.Checkbox(label="生成gif动图", value=True)        
        duration = gr.Slider(minimum=1, maximum=1000, step=1, value=300, label="每帧时长(毫秒)，生成gif模式下有效", visible=True)
        #make_a_gif.change(fn=lambda x: gr.update(visible=x), inputs=[make_a_gif], outputs=[duration])
        info2 = gr.HTML("<br>脚本制作：CastBox--zhouzhuo")

        return [trueyear, sex,currentyear, make_a_gif,lastyear,n_images,duration,info2]


    def run(self, p, trueyear, sex,currentyear, make_a_gif,lastyear,n_images,duration,info2):
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))
        
        p.do_not_save_grid = True
        #prompt1 = p.prompt
        prompt1=""
        if sex=="男":
            if currentyear<8:
                prompt1="a little baby,"+str(currentyear)+" years old"   
            if currentyear>=8 and currentyear<25:
                prompt1="a boy,"+str(currentyear)+" years old" 
            if currentyear>=25 and currentyear<50:
                prompt1="a man,"+str(currentyear)+" years old"    
            if currentyear>=50 and currentyear<=100:
                prompt1="a old man,"+str(currentyear)+" years old"    				
        else:
            if currentyear<8:
                prompt1="a little baby,"+str(currentyear)+" years old"    
            if currentyear>=8 and currentyear<25:
                prompt1="a girl,"+str(currentyear)+" years old" 
            if currentyear>=25 and currentyear<50:
                prompt1="a woman,"+str(currentyear)+" years old"    
            if currentyear>=50 and currentyear<=100:
                prompt1="a old lady,"+str(currentyear)+" years old"    	
        
        
		
        processed, processed_images = process(p, sex, trueyear, currentyear,lastyear,n_images)

        p.prompt_for_display = processed.prompt = prompt1
        processed_images_flattened = []
        
        for row in processed_images:
            processed_images_flattened += row
        
        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened, rows=p.batch_size * p.n_iter)] \
                + processed_images_flattened
        
        if make_a_gif or opts.grid_save:
            (fullfn, _) = images.save_image(processed.images[0], p.outpath_grids, "grid",
                prompt=p.prompt_for_display, seed=processed.seed, grid=True, p=p)
        
        if make_a_gif:
            for i, row in enumerate(processed_images):
                fullfn = fullfn[:fullfn.rfind(".")] + "_" + str(i) + ".gif"
                # since there is no option for saving gif images in images.save_image(), I had to
                # do it from scratch, maybe it can be improved in the future
                processed_images[i][0].save(fullfn, save_all=True,
                    append_images=processed_images[i][1:], optimize=False, duration=duration, loop=0)
        
        return processed