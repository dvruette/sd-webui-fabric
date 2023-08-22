import os
import dataclasses
import functools
import hashlib
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict

import gradio as gr
from PIL import Image

import modules.scripts
from modules import script_callbacks
from modules.ui_common import create_refresh_button
from modules.ui_components import FormGroup, ToolButton
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

from scripts.helpers import WebUiComponents
from scripts.patching import patch_unet_forward_pass, unpatch_unet_forward_pass


__version__ = "0.4.2"

DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1")

OUTPUT_PATH = "log/fabric/images"
PRESET_PATH = "log/fabric/presets"

if DEBUG:
    print(f"WARNING: Loading FABRIC v{__version__} in DEBUG mode")
else:
    print(f"Loading FABRIC v{__version__}")

"""
# Gradio 3.32 bug fix
Fixes FileNotFoundError when displaying PIL images in Gradio Gallery.
"""
import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)


def use_feedback(params):
    if not params.enabled:
        return False
    if params.start >= params.end and params.min_weight <= 0:
        return False
    if params.max_weight <= 0:
        return False
    if params.neg_scale <= 0 and len(params.pos_images) == 0:
        return False
    if len(params.pos_images) == 0 and len(params.neg_images) == 0:
        return False
    return True


def image_hash(img, length=16):
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(img.tobytes())
    img_hash = hash_sha256.hexdigest()
    if length and length > 0:
        img_hash = img_hash[:length]
    return img_hash


def save_feedback_image(img, filename=None, base_path=OUTPUT_PATH):
    if filename is None:
        filename = image_hash(img) + ".png"
    img_path = Path(modules.scripts.basedir(), base_path, filename)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(img_path)
    return filename


@functools.lru_cache(maxsize=128)
def load_feedback_image(filename, base_path=OUTPUT_PATH):
    img_path = Path(modules.scripts.basedir(), base_path, filename)
    return Image.open(img_path)


def full_image_path(filename, base_path=OUTPUT_PATH):
    img_path = Path(modules.scripts.basedir(), base_path, filename)
    return str(img_path)


# helper functions for loading saved params
def _load_feedback_paths(d, key):
    try:
        paths = json.loads(d.get(key, "[]").replace("'", '"'))
    except Exception as e:
        traceback.print_exc()
        print(d)
        print(f"Failed to load feedback images: {d.get(key, '[]')}")
        paths = []

    paths = [path for path in paths if os.path.exists(full_image_path(path))]
    return paths

def _load_gallery(d, key):
    paths = _load_feedback_paths(d, key)
    return [full_image_path(path) for path in paths]


def _save_preset(preset_name, liked_paths, disliked_paths, base_path=PRESET_PATH):
    preset_path = Path(modules.scripts.basedir(), base_path, f"{preset_name}.json")
    preset_path.parent.mkdir(parents=True, exist_ok=True)

    preset = {
        "liked_paths": liked_paths,
        "disliked_paths": disliked_paths,
    }

    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=4)

def _load_presets(base_path=PRESET_PATH):
    presets_path = Path(modules.scripts.basedir(), base_path)
    presets_path.mkdir(parents=True, exist_ok=True)
    presets = [preset.stem for preset in presets_path.iterdir() if preset.is_file() and preset.suffix == ".json"]
    return presets


@dataclass
class FabricParams:
    enabled: bool = True
    start: float = 0.0
    end: float = 0.8
    min_weight: float = 0.0
    max_weight: float = 0.8
    neg_scale: float = 0.5
    pos_images: list = dataclasses.field(default_factory=list)
    neg_images: list = dataclasses.field(default_factory=list)
    pos_latents: list = None
    neg_latents: list = None
    feedback_during_high_res_fix: bool = False


# TODO: replace global state with Gradio state
class FabricState:
    txt2img_images = []
    img2img_images = []


class FabricScript(modules.scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "FABRIC"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        self.txt2img_selected_image = gr.State(None)
        self.img2img_selected_image = gr.State(None)
        selected_like = gr.State(None)
        selected_dislike = gr.State(None)
        # need to use JSON over State to make it compatible with gr.update
        liked_paths = gr.JSON(value=[], visible=False)
        disliked_paths = gr.JSON(value=[], visible=False)

        with gr.Accordion(f"{self.title()} v{__version__}", open=DEBUG, elem_id="fabric"):
            with gr.Row():
                presets_list = gr.Dropdown(label="Presets", choices=_load_presets(), default=None, live=False)
                reload_presets_btn = create_refresh_button(presets_list, lambda: None, lambda: {"choices": _load_presets()}, "fabric_reload_presets_btn")

            with gr.Tabs():
                with gr.Tab("Current batch"):
                    # TODO: figure out why the display is shared between tabs
                    self.img2img_selected_display = gr.Image(value=None, type="pil", label="Selected image", visible=is_img2img).style(height=256)
                    self.txt2img_selected_display = gr.Image(value=None, type="pil", label="Selected image", visible=not is_img2img).style(height=256)

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
                    like_gallery = gr.Gallery(label="Liked images", elem_id="fabric_like_gallery").style(columns=4, height=128)
                with gr.Tab("👎 Dislikes"):
                    with gr.Row():
                        remove_selected_dislike_btn = gr.Button("Remove selected", interactive=False)
                        clear_disliked_btn = gr.Button("Clear")
                    dislike_gallery = gr.Gallery(label="Disliked images", elem_id="fabric_dislike_gallery").style(columns=4, height=128)

            save_preset_btn = gr.Button("Save as preset")
            
            gr.HTML("<hr style='border-color: var(--block-border-color)'>")


            with FormGroup():
                gr.HTML("<h3>FABRIC Settings</h3>")

                with gr.Row():
                    feedback_disabled = gr.Checkbox(label="Disable feedback", value=False)

                with gr.Row():
                    feedback_max_images = gr.Slider(minimum=0, maximum=10, step=1, value=4, label="Max. feedback images")

                with gr.Row():
                    feedback_start = gr.Slider(0.0, 1.0, value=0.0, label="Feedback start")
                    feedback_end = gr.Slider(0.0, 1.0, value=0.8, label="Feedback end")
                with gr.Row():
                    feedback_min_weight = gr.Slider(0.0, 1.0, value=0.0, label="Min. weight")
                    feedback_max_weight = gr.Slider(0.0, 1.0, value=0.8, label="Max. weight")
                    feedback_neg_scale = gr.Slider(0.0, 1.0, value=0.5, label="Neg. scale")
                with gr.Row():
                    feedback_during_high_res_fix = gr.Checkbox(label="Enable feedback during hires. fix", value=False)


        WebUiComponents.on_txt2img_gallery(self.register_txt2img_gallery_select)
        WebUiComponents.on_img2img_gallery(self.register_img2img_gallery_select)

        if is_img2img:
            like_btn_selected.click(self.add_image_to_state, inputs=[self.img2img_selected_image, liked_paths], outputs=[like_gallery, liked_paths])
            dislike_btn_selected.click(self.add_image_to_state, inputs=[self.img2img_selected_image, disliked_paths], outputs=[dislike_gallery, disliked_paths])
        else:
            like_btn_selected.click(self.add_image_to_state, inputs=[self.txt2img_selected_image, liked_paths], outputs=[like_gallery, liked_paths])
            dislike_btn_selected.click(self.add_image_to_state, inputs=[self.txt2img_selected_image, disliked_paths], outputs=[dislike_gallery, disliked_paths])

        like_btn_uploaded.click(self.add_image_to_state, inputs=[upload_img_input, liked_paths], outputs=[like_gallery, liked_paths])
        dislike_btn_uploaded.click(self.add_image_to_state, inputs=[upload_img_input, disliked_paths], outputs=[dislike_gallery, disliked_paths])

        clear_liked_btn.click(lambda _: ([], [], []), inputs=[], outputs=[like_gallery, liked_paths])
        clear_disliked_btn.click(lambda _: ([], [], []), inputs=[], outputs=[dislike_gallery, disliked_paths])

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
            inputs=[liked_paths, selected_like],
            outputs=[like_gallery, liked_paths, selected_like, remove_selected_like_btn],
        )

        remove_selected_dislike_btn.click(
            self.remove_selected,
            inputs=[disliked_paths, selected_dislike],
            outputs=[dislike_gallery, disliked_paths, selected_dislike, remove_selected_dislike_btn],
        )

        save_preset_btn.click(
            self.save_preset,
            _js="(a, b, c, d) => [a, b, c, prompt('Enter a name for your preset:')]",
            inputs=[presets_list, liked_paths, disliked_paths, disliked_paths],  # last input is a dummy
            outputs=[presets_list],
        )

        presets_list.input(
            self.on_preset_selected,
            inputs=[presets_list, liked_paths, disliked_paths],
            outputs=[
                liked_paths,
                disliked_paths,
                like_gallery,
                dislike_gallery,
            ],
        )

        # sets FABRIC params when "send to txt2img/img2img" is clicked
        self.infotext_fields = [
            (feedback_disabled, lambda d: gr.Checkbox.update(value="fabric_start" not in d)),
            (feedback_start, "fabric_start"),
            (feedback_end, "fabric_end"),
            (feedback_min_weight, "fabric_min_weight"),
            (feedback_max_weight, "fabric_max_weight"),
            (feedback_neg_scale, "fabric_neg_scale"),
            (feedback_during_high_res_fix, "fabric_feedback_during_high_res_fix"),
            (liked_paths, lambda d: gr.update(value=_load_feedback_paths(d, "fabric_pos_images")) if "fabric_pos_images" in d else None),
            (disliked_paths, lambda d: gr.update(value=_load_feedback_paths(d, "fabric_neg_images")) if "fabric_neg_images" in d else None),
            (like_gallery, lambda d: gr.Gallery.update(value=_load_gallery(d, "fabric_pos_images")) if "fabric_pos_images" in d else None),
            (dislike_gallery, lambda d: gr.Gallery.update(value=_load_gallery(d, "fabric_neg_images")) if "fabric_neg_images" in d else None),
        ]

        return [
            liked_paths,
            disliked_paths,
            feedback_disabled,
            feedback_max_images,
            feedback_start,
            feedback_end,
            feedback_min_weight,
            feedback_max_weight,
            feedback_neg_scale,
            feedback_during_high_res_fix,
        ]
    

    def select_for_removal(self, gallery, selected_idx):
        return [
            selected_idx,
            gr.update(interactive=True),
        ]
    
    def remove_selected(self, paths, idx):
        if idx >= 0 and idx < len(paths):
            paths.pop(idx)
        gallery = [full_image_path(path) for path in paths]

        return [
            gallery,
            paths,
            gr.update(value=None),
            gr.update(interactive=False),
        ]
    
    def add_image_to_state(self, img, paths):
        if img is not None:
            path = save_feedback_image(img)
            paths.append(path)
        gallery = [full_image_path(path) for path in paths]
        return gallery, paths

    def save_preset(self, presets, liked_paths, disliked_paths, preset_name):
        if preset_name is not None and preset_name != "":
            _save_preset(preset_name, liked_paths, disliked_paths)
        return gr.update(choices=_load_presets())

    def on_preset_selected(self, preset_name, liked_paths, disliked_paths):
        preset_path = Path(modules.scripts.basedir(), PRESET_PATH, f"{preset_name}.json")
        if preset_path.exists():
            try:
                with open(preset_path, "r") as f:
                    preset = json.load(f)
                assert "liked_paths" in preset, "Missing 'liked_paths' in preset"
                assert "disliked_paths" in preset, "Missing 'disliked_paths' in preset"
                liked_paths = preset["liked_paths"]
                disliked_paths = preset["disliked_paths"]
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to load preset: {preset_path}")
        like_gallery = [full_image_path(path) for path in liked_paths]
        dislike_gallery = [full_image_path(path) for path in disliked_paths]
        return liked_paths, disliked_paths, like_gallery, dislike_gallery
    
    def register_txt2img_gallery_select(self, gallery):
        self.register_gallery_select(
            gallery,
            listener=self.on_txt2img_gallery_select,
            selected=self.txt2img_selected_image,
            display=self.txt2img_selected_display,
        )

    def register_img2img_gallery_select(self, gallery):
        self.register_gallery_select(
            gallery,
            listener=self.on_img2img_gallery_select,
            selected=self.img2img_selected_image,
            display=self.img2img_selected_display,
        )
        
    def register_gallery_select(self, gallery, listener=None, selected=None, display=None):
        gallery.select(
            listener, 
            _js="(a, b) => [a, selected_gallery_index()]",
            inputs=[
                gallery,
                gallery,  # can be any Gradio component (but not None), will be overwritten with selected gallery index
            ],
            outputs=[selected, display],
        )
    
    def on_txt2img_gallery_select(self, gallery, selected_idx):
        return self.on_gallery_select(gallery, selected_idx, FabricState.txt2img_images)
        
    def on_img2img_gallery_select(self, gallery, selected_idx):
        return self.on_gallery_select(gallery, selected_idx, FabricState.img2img_images)
        
    def on_gallery_select(self, gallery, selected_idx, images):
        idx = selected_idx - (len(gallery) - len(images))

        if idx >= 0 and idx < len(images):
            return images[idx], gr.update(value=images[idx])
        else:
            return None, None
    
    def process(self, p, *args):
        (
            liked_paths,
            disliked_paths,
            feedback_disabled,
            feedback_max_images,
            feedback_start, 
            feedback_end, 
            feedback_min_weight, 
            feedback_max_weight, 
            feedback_neg_scale,
            feedback_during_high_res_fix,
        ) = args

        # restore original U-Net forward pass in case previous batch errored out
        unpatch_unet_forward_pass(p.sd_model.model.diffusion_model)

        liked_paths = liked_paths[-int(feedback_max_images):]
        disliked_paths = disliked_paths[-int(feedback_max_images):]

        likes = [load_feedback_image(path) for path in liked_paths]
        dislikes = [load_feedback_image(path) for path in disliked_paths]

        params = FabricParams(
            enabled=(not feedback_disabled),
            start=feedback_start,
            end=feedback_end,
            min_weight=feedback_min_weight,
            max_weight=feedback_max_weight,
            neg_scale=feedback_neg_scale,
            pos_images=likes,
            neg_images=dislikes,
            feedback_during_high_res_fix=feedback_during_high_res_fix,
        )


        if use_feedback(params) or (DEBUG and not feedback_disabled):
            print(f"[FABRIC] Patching U-Net forward pass... ({len(likes)} likes, {len(dislikes)} dislikes)")
            
            # log the generation params to be displayed/stored as metadata
            log_params = asdict(params)
            log_params["pos_images"] = json.dumps(liked_paths)
            log_params["neg_images"] = json.dumps(disliked_paths)
            del log_params["enabled"]

            log_params = {f"fabric_{k}": v for k, v in log_params.items()}
            p.extra_generation_params.update(log_params)
            
            unet = p.sd_model.model.diffusion_model
            patch_unet_forward_pass(p, unet, params)
        else:
            print("[FABRIC] Skipping U-Net forward pass patching")
    
    def postprocess(self, p, processed, *args):
        print("[FABRIC] Restoring original U-Net forward pass")
        unpatch_unet_forward_pass(p.sd_model.model.diffusion_model)

        images = processed.images[processed.index_of_first_image:]
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            FabricState.txt2img_images = images
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            FabricState.img2img_images = images
        else:
            raise RuntimeError(f"Unsupported processing type: {type(p)}")


script_callbacks.on_after_component(WebUiComponents.register_component)
