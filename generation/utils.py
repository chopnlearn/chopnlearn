from transformers import PretrainedConfig
from PIL import Image


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def concat_h(*argv, pad=0):
    width = 0
    height = 0
    count = len(argv)

    for img in argv:
        width += img.width
        height = max(height, img.height)

    dst = Image.new('RGB', (width + (count-1)*pad, height))
    start = 0
    for i, img in enumerate(argv):
        dst.paste(img, (start, 0))
        start += img.width + pad
    return dst

