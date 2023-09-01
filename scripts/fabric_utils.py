import hashlib

from PIL import Image


def image_hash(img: Image.Image, length: int = 16):
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(img.tobytes())
    img_hash = hash_sha256.hexdigest()
    if length and length > 0:
        img_hash = img_hash[:length]
    return img_hash


class WebUiComponents:
    txt2img_gallery = None
    img2img_gallery = None
    txt2img_callbacks = []
    img2img_callbacks = []

    @staticmethod
    def on_txt2img_gallery(callback):
        if WebUiComponents.txt2img_gallery is not None:
            callback(WebUiComponents.txt2img_gallery)
        else:
            WebUiComponents.txt2img_callbacks.append(callback)

    def on_img2img_gallery(callback):
        if WebUiComponents.img2img_gallery is not None:
            callback(WebUiComponents.img2img_gallery)
        else:
            WebUiComponents.img2img_callbacks.append(callback)

    @staticmethod
    def register_component(component, **kwargs):
        elem_id = getattr(component, "elem_id", None)
        if elem_id == "txt2img_gallery":
            WebUiComponents.txt2img_gallery = component
            for callback in WebUiComponents.txt2img_callbacks:
                callback(component)
            WebUiComponents.txt2img_callbacks = []
        elif elem_id == "img2img_gallery":
            WebUiComponents.img2img_gallery = component
            for callback in WebUiComponents.img2img_callbacks:
                callback(component)
            WebUiComponents.img2img_callbacks = []
