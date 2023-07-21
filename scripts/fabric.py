import os
import dataclasses
import functools
from pathlib import Path
from dataclasses import dataclass

import gradio as gr
import torchvision.transforms.functional as functional
from PIL import Image

import modules.scripts
from modules import devices, script_callbacks, images
from modules.ui_components import FormGroup

from scripts.patching import patch_unet_forward_pass, unpatch_unet_forward_pass
from scripts.helpers import WebUiComponents


__version__ = "0.2"

DEBUG = False

"""
# Gradio 3.32 bug fix
Fixes FileNotFoundError when displaying PIL images in Gradio Gallery.
"""
import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)


@dataclass
class FabricParams:
    enabled: bool = True
    start: float = 0.0
    end: float = 0.8
    min_weight: float = 0.0
    max_weight: float = 0.8
    neg_scale: float = 0.5
    pos_latents: list = dataclasses.field(default_factory=list)
    neg_latents: list = dataclasses.field(default_factory=list)


# TODO: replace global state with Gradio state
class FabricState:
    batch_images = []


def encode_to_latent(p, feedback_imgs):
    w, h = p.width, p.height
    zs = []
    for image in feedback_imgs:
        image = images.resize_image(1, image, w, h)
        x = functional.pil_to_tensor(image)
        x = functional.center_crop(x, (h, w))  # just to be safe
        x = x.to(devices.device, dtype=devices.dtype_vae)
        x = ((x / 255.0) * 2.0 - 1.0).unsqueeze(0)

        # TODO: use caching to make this faster
        with devices.autocast():
            vae_output = p.sd_model.encode_first_stage(x)
            z = p.sd_model.get_first_stage_encoding(vae_output)
        zs.append(z.squeeze(0))
    return zs


class FabricScript(modules.scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "FABRIC"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        self.selected_image = gr.State(None)
        liked_images = gr.State([])
        disliked_images = gr.State([])
        selected_like = gr.State(None)
        selected_dislike = gr.State(None)

        with gr.Accordion(f"{self.title()} v{__version__}", open=DEBUG, elem_id="fabric"):
            if DEBUG:
                like_example_btn = gr.Button("👍 Example")

            with gr.Tabs():
                with gr.Tab("Current batch"):
                    self.selected_img_display = gr.Image(value=None, type="pil", label="Selected image").style(height=256)

                    with gr.Row():
                        like_btn_selected = gr.Button("👍 Like")
                        dislike_btn_selected = gr.Button("👎 Dislike")
                
                with gr.Tab("Upload image"):
                    upload_img_input = gr.Image(type="pil", label="Upload image").style(height=256)

                    with gr.Row():
                        like_btn_uploaded = gr.Button("👍 Like")
                        dislike_btn_uploaded = gr.Button("👎 Dislike")
            

            with gr.Tabs(initial_tab="👍 Likes"):
                with gr.Tab("👍 Likes"):
                    with gr.Row():
                        remove_selected_like_btn = gr.Button("Remove selected", interactive=False)
                        clear_liked_btn = gr.Button("Clear")
                    like_gallery = gr.Gallery(label="Liked images", elem_id="fabric_like_gallery").style(columns=5, height=128)
                with gr.Tab("👎 Dislikes"):
                    with gr.Row():
                        remove_selected_dislike_btn = gr.Button("Remove selected", interactive=False)
                        clear_disliked_btn = gr.Button("Clear")
                    dislike_gallery = gr.Gallery(label="Disliked images", elem_id="fabric_dislike_gallery").style(columns=5, height=128)


            with FormGroup():
                feedback_disabled = gr.Checkbox(label="Disable feedback", value=False)

            with FormGroup():
                with gr.Row():
                    # TODO: figure out how to make the step size do what it's supposed to
                    feedback_max_images = gr.Slider(minimum=0, maximum=10, step=1, value=4, label="Max. feedback images")

                with gr.Row():
                    feedback_start = gr.Slider(0.0, 1.0, value=0.0, label="Feedback start")
                    feedback_end = gr.Slider(0.0, 1.0, value=0.8, label="Feedback end")
                with gr.Row():
                    feedback_min_weight = gr.Slider(0.0, 1.0, value=0.0, label="Min. weight")
                    feedback_max_weight = gr.Slider(0.0, 1.0, value=0.8, label="Max. weight")
                    feedback_neg_scale = gr.Slider(0.0, 1.0, value=0.5, label="Neg. scale")


        WebUiComponents.on_txt2img_gallery(self.register_txt2img_gallery_select)

        like_btn_selected.click(self.add_image_to_state, inputs=[self.selected_image, liked_images], outputs=[liked_images, like_gallery])
        dislike_btn_selected.click(self.add_image_to_state, inputs=[self.selected_image, disliked_images], outputs=[disliked_images, dislike_gallery])

        like_btn_uploaded.click(self.add_image_to_state, inputs=[upload_img_input, liked_images], outputs=[liked_images, like_gallery])
        dislike_btn_uploaded.click(self.add_image_to_state, inputs=[upload_img_input, disliked_images], outputs=[disliked_images, dislike_gallery])

        clear_liked_btn.click(lambda _: [[], []], inputs=liked_images, outputs=[liked_images, like_gallery])
        clear_disliked_btn.click(lambda _: [[], []], inputs=disliked_images, outputs=[disliked_images, dislike_gallery])

        like_gallery.select(
            self.select_for_removal,
            _js="(a, b) => [a, fabric_selected_gallery_index('fabric_like_gallery')]",
            inputs=[like_gallery, like_gallery],
            outputs=[selected_like, remove_selected_like_btn],
        )

        dislike_gallery.select(
            self.select_for_removal,
            _js="(a, b) => [a, fabric_selected_gallery_index('fabric_dislike_gallery')]",
            inputs=[dislike_gallery, dislike_gallery],
            outputs=[selected_dislike, remove_selected_dislike_btn],
        )

        remove_selected_like_btn.click(
            self.remove_selected,
            inputs=[liked_images, selected_like],
            outputs=[liked_images, like_gallery, selected_like, remove_selected_like_btn],
        )

        remove_selected_dislike_btn.click(
            self.remove_selected,
            inputs=[disliked_images, selected_dislike],
            outputs=[disliked_images, dislike_gallery, selected_dislike, remove_selected_dislike_btn],
        )

        if DEBUG:
            like_example_btn.click(functools.partial(self.on_like_example, example="example1"), inputs=liked_images, outputs=[liked_images, like_gallery])

        return [
            liked_images,
            disliked_images,
            feedback_disabled,
            feedback_max_images,
            feedback_start,
            feedback_end,
            feedback_min_weight,
            feedback_max_weight,
            feedback_neg_scale,
        ]
    

    def select_for_removal(self, gallery, selected_idx):
        return [
            selected_idx,
            gr.update(interactive=True),
        ]
    
    def remove_selected(self, images, idx):
        if idx >= 0 and idx < len(images):
            images.pop(idx)

        return [
            images,
            images,
            gr.update(value=None),
            gr.update(interactive=False),
        ]
    
    def add_image_to_state(self, img, images):
        if img is not None:
            images.append(img)
        return images, images
    
    def on_like_example(self, liked_images, example="example1"):
        img_path = Path(__file__).parent.parent.absolute() / "images" / f"{example}.png"
        image = Image.open(img_path)
        if image is not None:
            liked_images.append(image)
        return liked_images, liked_images
    
    def register_txt2img_gallery_select(self, gallery):
        gallery.select(
            self.on_txt2img_gallery_select, 
            _js="(a, b) => [a, selected_gallery_index()]",
            inputs=[
                gallery,
                gallery,  # can be any Gradio component (but not None), will be overwritten with selected gallery index
            ],
            outputs=[self.selected_image, self.selected_img_display],
        )
    
    def on_txt2img_gallery_select(self, gallery, selected_idx):
        images = FabricState.batch_images
        idx = selected_idx - (len(gallery) - len(images))

        if idx >= 0 and idx < len(images):
            return images[idx], gr.update(value=images[idx])
        else:
            return None, None
    
    def process(self, p, *args):
        (
            liked_images,
            disliked_images,
            feedback_disabled,
            feedback_max_images,
            feedback_start, 
            feedback_end, 
            feedback_min_weight, 
            feedback_max_weight, 
            feedback_neg_scale,
        ) = args

        print("[FABRIC] Encoding feedback images into latent space...")
        likes = liked_images[:int(feedback_max_images)]
        dislikes = disliked_images[:int(feedback_max_images)]
        pos_latents = encode_to_latent(p, likes)
        neg_latents = encode_to_latent(p, dislikes)

        print("[FABRIC] Patching U-Net forward pass...")
        params = FabricParams(
            enabled=not feedback_disabled,
            start=feedback_start,
            end=feedback_end,
            min_weight=feedback_min_weight,
            max_weight=feedback_max_weight,
            neg_scale=feedback_neg_scale,
            pos_latents=pos_latents,
            neg_latents=neg_latents,
        )
        unet = p.sd_model.model.diffusion_model
        patch_unet_forward_pass(p, unet, params)
    
    def postprocess(self, p, processed, *args):
        print("[FABRIC] Restoring original U-Net forward pass")
        unpatch_unet_forward_pass(p.sd_model.model.diffusion_model)

        FabricState.batch_images = processed.images[processed.index_of_first_image:]


script_callbacks.on_after_component(WebUiComponents.register_component)
