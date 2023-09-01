# FABRIC Plugin for Stable Diffusion WebUI

Official FABRIC implementation for [automatic1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Steer the model towards generating desirable results by simply liking/disliking images. These feedback images can be generated or provided by you and will make the model generate images that look more/less like the feedback. Instead of meticulously iterating on your prompt until you get what you're looking for, with FABRIC you can simply "show" the model what you want and don't want.

📜 Paper: https://arxiv.org/abs/2307.10159

🎨 Project page: https://sd-fabric.github.io

ComfyUI node (by [@ssitu](https://github.com/ssitu)): https://github.com/ssitu/ComfyUI_fabric

![demo](static/fabric_demo.gif)

## Releases

- [29.08.2023] 🏎️ v0.6.0: Up to 2x faster and 4x less VRAM usage thanks to [Token Merging](https://github.com/dbolya/tomesd/tree/main) (tested with 16 feedback images and a batch size of 4), moderate gains for fewer feedback images (10% speedup for 2 images, 30% for 8 images). Enable the Token Merging option to take advantage of this.
- [22.08.2023] 🗃️ v0.5.0: Adds support for presets. Makes generated images using FABRIC more reproducible by loading the correct (previously used) feedback images when using "send to text2img/img2img".

## Installation

1. Open the "Extensions" tab
2. Open the "Install from URL" tab
3. Copy-paste `https://github.com/dvruette/sd-webui-fabric.git` into "URL for extension's git repository" and press "Install"
4. Switch to the "Installed" tab and press "Apply and restart UI"
5. (optional) Since FABRIC is quite VRAM intensive, using `--xformers` is recommended.
   1. If you still run out of VRAM, try enabling the "Token Merging" setting for even better memory efficiency.

### Compatibility Notes
- SDXL is currently not supported (PRs welcome!)
- Compatibility with other plugins is largely untested. If you experience errors with other plugins enabled, please disable all other plugins for the best chance for FABRIC to work. If you can figure out which plugin is incompatible, please open an issue.
- The plugin is INCOMPATIBLE with `reference` mode in the ControlNet plugin. Instead of using a reference image, simply add it as a liked image. If you accidentally enable FABRIC and `reference` mode at the same time, you will have to restart the WebUI to fix it.
- Some attention processors are not supported. In particular, `--opt-sub-quad-attention` and `--opt-split-attention-v1` are not supported at the moment.



## How-to and Examples

Coming soon. Feel free to share examples with us if you have found something that works well and we'll add it here :)


## Citation
```
@misc{vonrutte2023fabric,
      title={FABRIC: Personalizing Diffusion Models with Iterative Feedback}, 
      author={Dimitri von Rütte and Elisabetta Fedele and Jonathan Thomm and Lukas Wolf},
      year={2023},
      eprint={2307.10159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
