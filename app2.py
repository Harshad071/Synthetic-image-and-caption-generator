import io
from typing import Tuple, Literal, Optional

import streamlit as st
from PIL import Image
import torch

# Transformers imports (optional branches; only what's needed will be used)
from transformers import AutoTokenizer, AutoProcessor
from transformers import VisionEncoderDecoderModel
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except Exception:
    Blip2Processor = None
    Blip2ForConditionalGeneration = None


ModelKind = Literal["ved", "blip", "blip2"]


@st.cache_resource(show_spinner="Loading model from local folder...")
def get_model(model_dir: str):
    """
    Load a local image captioning model from disk and cache it as a resource.
    Supports:
      - VisionEncoderDecoderModel (+ AutoTokenizer + AutoProcessor)
      - BLIP (BlipForConditionalGeneration + BlipProcessor)
      - BLIP-2 (Blip2ForConditionalGeneration + Blip2Processor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try VisionEncoderDecoder first
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
        # For VED we typically need an image processor & tokenizer
        # AutoProcessor can provide image processing; tokenizer is separate.
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model.to(device)
        return {"kind": "ved", "model": model, "processor": processor, "tokenizer": tokenizer, "device": device}
    except Exception:
        pass

    # Try BLIP
    if BlipForConditionalGeneration is not None and BlipProcessor is not None:
        try:
            model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
            processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
            model.to(device)
            return {"kind": "blip", "model": model, "processor": processor, "tokenizer": None, "device": device}
        except Exception:
            pass

    # Try BLIP-2
    if Blip2ForConditionalGeneration is not None and Blip2Processor is not None:
        try:
            model = Blip2ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
            processor = Blip2Processor.from_pretrained(model_dir, local_files_only=True)
            model.to(device)
            return {"kind": "blip2", "model": model, "processor": processor, "tokenizer": None, "device": device}
        except Exception:
            pass

    raise RuntimeError(
        "Failed to load a supported captioning model. "
        "Ensure your local folder contains a compatible model (VisionEncoderDecoder, BLIP, or BLIP-2) with config files."
    )


def _infer_with_ved(
    image: Image.Image,
    model,
    image_processor,
    tokenizer,
    device: str,
    max_length: int,
    num_beams: int,
) -> str:
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        # Some processors return a different structure; fallback to attribute
        pixel_values = inputs["pixel_values"]
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def _infer_with_blip(
    image: Image.Image,
    model,
    processor,
    device: str,
    max_length: int,
    num_beams: int,
) -> str:
    # BLIP usually does unconditional captioning with empty text prompt
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    # BLIP uses processor.tokenizer for decoding in most releases
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


def _infer_with_blip2(
    image: Image.Image,
    model,
    processor,
    device: str,
    max_length: int,
    num_beams: int,
) -> str:
    # BLIP-2 can also do unconditional captioning (no prompt)
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


@st.cache_data(show_spinner="Generating caption...", ttl=3600)
def generate_caption(
    image_bytes: bytes,
    model_dir: str,
    max_length: int,
    num_beams: int,
) -> str:
    """
    Cached caption generation that only takes hashable inputs.
    This function fetches the (cached) model internally to avoid passing
    unhashable model objects to Streamlit's cache.
    """
    payload = get_model(model_dir)
    kind: ModelKind = payload["kind"]  # type: ignore
    model = payload["model"]
    processor = payload["processor"]
    tokenizer = payload["tokenizer"]
    device: str = payload["device"]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if kind == "ved":
        return _infer_with_ved(
            image=image,
            model=model,
            image_processor=processor,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
        )
    elif kind == "blip":
        return _infer_with_blip(
            image=image,
            model=model,
            processor=processor,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
        )
    else:  # "blip2"
        return _infer_with_blip2(
            image=image,
            model=model,
            processor=processor,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
        )


def main():
    st.set_page_config(page_title="Local Image Captioning", page_icon="🖼️", layout="centered")

    st.title("Local Image Captioning")
    st.caption("Runs fully offline using local Hugging Face model files")

    with st.sidebar:
        st.subheader("Settings")
        model_dir = st.text_input(
            "Local model folder",
            value="model",
            help="Path to the folder containing your model weights and config (config.json, tokenizer.json, etc.)",
        )
        gpu_available = torch.cuda.is_available()
        st.write(f"GPU available: {'Yes' if gpu_available else 'No'}")

        max_length = st.slider("Max caption length", 8, 64, 32, step=1)
        num_beams = st.slider("Beam size", 1, 8, 4, step=1)

        st.divider()
        st.write("Tip: Model is cached across runs. Change the path if you switch models.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
        with col2:
            st.write("Model folder:", f"`{model_dir}`")
            if st.button("Generate Caption", use_container_width=True):
                try:
                    caption = generate_caption(
                        image_bytes=uploaded.getvalue(),
                        model_dir=model_dir,
                        max_length=max_length,
                        num_beams=num_beams,
                    )
                    st.success("Caption generated")
                    st.markdown(f"**Caption**: {caption}")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

    st.divider()
    st.markdown(
        "Notes:\n"
        "- Ensure the model folder is fully downloaded locally (no internet required) and compatible with VisionEncoderDecoder, BLIP, or BLIP-2.\n"
        "- Large models may take time to load on first use; subsequent runs are fast due to caching."
    )


if __name__ == "__main__":
    main()
