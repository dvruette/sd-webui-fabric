
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
