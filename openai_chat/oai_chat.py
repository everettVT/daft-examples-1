# /// script
# description = "Daft OpenAI Chat with images and audio"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "openai","numpy", "pydantic", "python-dotenv", "requests"]
# ///
import asyncio
import json
import base64
from typing import Any
import mimetypes
import requests
import pydantic
import numpy as np
import daft
from openai import AsyncOpenAI
from itertools import zip_longest


ALLOWED_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp"]
ALLOWED_AUDIO_MIME_TYPES = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"]
ALLOWED_VIDEO_MIME_TYPES = ["video/mp4", "video/webm", "video/ogg", "video/avi", "video/mov", "video/flv", "video/wmv", "video/mkv"]


def fetch_url_as_data_url(url: str) -> str:
   with requests.get(url) as response:
        response.raise_for_status()
        data = base64.b64encode(response.content).decode("utf-8")
        content_type = response.headers.get("Content-Type") or mimetypes.guess_type(url)[0]
        return f"data:{content_type};base64,{data}"


def handle_image(image: str | np.ndarray | bytes):
    if isinstance(image, str):
        if image.startswith("http"):
            # Try to inline to avoid provider-side fetch failures; fallback to the URL
            return {"type":"image_url", "image_url": {"url": fetch_url_as_data_url(image)}}

        else: 
            # Assume base64
            mime_guess = mimetypes.guess_type(image)[0]
            if mime_guess in ALLOWED_IMAGE_MIME_TYPES:
                return {"type":"image_url", "image_url": {"url": f"data:{mime_guess};base64,{image}"}}
            else:
                raise ValueError(f"Unsupported image mime type: {mime_guess} for image: {str(image)[:100]}")

    elif isinstance(image, np.ndarray):
        return {"type":"image_url", "image_url": {"data": base64.b64encode(image.tobytes()).decode("utf-8"), "format": "png"}}
    elif isinstance(image, bytes):
         return {"type":"image_url", "image_url": {"data": image, "format": "png"}}
    return {"url": f"data:image/png;base64,{image}"}


def handle_audio(audio: str | np.ndarray | bytes):
    if isinstance(audio, str):
        if audio.startswith("http"):
            return {"type": "audio_url", "audio_url": {"url": audio}}
        else: 
            # Assume base64
            mime_guess = mimetypes.guess_type(audio)[0]
            if mime_guess in ALLOWED_AUDIO_MIME_TYPES:
                return {"type": "audio_url", "audio_url": {"url": f"data:{mime_guess};base64,{audio}"}}
            else:
                raise ValueError(f"Unsupported audio mime type: {mime_guess} for audio: {str(audio)[:100]}")
    elif isinstance(audio, np.ndarray):
        return {"data": base64.b64encode(audio.tobytes()).decode("utf-8"), "format": "wav"}
    elif isinstance(audio, bytes):
        return {"data": base64.b64encode(audio).decode("utf-8"), "format": "wav"}
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

def handle_video(video: str | np.ndarray | bytes):
    if isinstance(video, str):
        if video.startswith("http"):
            return {"type":"video_url", "video_url": {"url": fetch_url_as_data_url(video)}}
        else: 
            # Assume base64
            mime_guess = mimetypes.guess_type(video)[0]
            if mime_guess in ALLOWED_VIDEO_MIME_TYPES:
                return {"type":"video_url", "video_url": {"url": f"data:{mime_guess};base64,{video}"}}
            else:
                raise ValueError(f"Unsupported video mime type: {mime_guess} for video: {str(video)[:100]}")
    elif isinstance(video, np.ndarray):
        return {"type":"video_url", "video_url": {"data": base64.b64encode(video.tobytes()).decode("utf-8"), "format": "mp4"}}
    elif isinstance(video, bytes):
        return {"type":"video_url", "video_url": {"data": base64.b64encode(video).decode("utf-8"), "format": "mp4"}}
    else:
        raise ValueError(f"Unsupported video type: {type(video)}")


def build_messages(
    history:  str | None,
    texts:    str | list[str] | None,
    images:   str | list[str] | None,
    audio:    str | list[str] | None,
    ) -> list[dict]:

    messages = []
    content = []

    if history: 
        messages.extend(json.loads(history))
        
    # Normalize inputs to lists
    texts_list = [] if texts is None else (texts if isinstance(texts, list) else [texts])
    images_list = [] if images is None else (images if isinstance(images, list) else [images])
    audio_list = [] if audio is None else (audio if isinstance(audio, list) else [audio])

    if images_list:
        content.extend([handle_image(i) for i in images_list])


    if audio_list:
        content.extend([handle_audio(a) for a in audio_list])

    # Text goes last so the model sees the image or audio first
    if texts_list:
        content.extend([{"type":"text", "text": t} for t in texts_list])
    
    messages.append({
                "role":"user",
                "content": content,
            })

    return messages

def build_kwargs(
    sampling_params: dict[str, Any] | None, 
    pydantic_model: pydantic.BaseModel | None,
    extra_body: dict[str, Any] | None, 
    kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:

    merged: dict[str, Any] = {}
    
    if sampling_params:
        merged.update(sampling_params)
    
    if pydantic_model:
        merged["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_model.__name__,
                "schema": pydantic_model.model_json_schema(),
            },
        }
    
    if extra_body:
        merged["extra_body"] = extra_body
    
    if kwargs:
        merged.update(kwargs)
    
    return merged



@daft.udf(return_dtype=daft.DataType.string())
class openai_chat:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def __call__(self,
        history_col: daft.Series | None = None, # JSON
        text_col:  daft.Series | None = None, # String or list of strings
        image_col: daft.Series | None = None, # Image or list of images
        audio_col: daft.Series | None = None, # Audio or list of files
        model_id: str = "openai/gpt-5-nano", # Model id
        sampling_params: dict[str, Any] | None = None,
        pydantic_model: pydantic.BaseModel | None = None,
        extra_body: dict | None = None,
        **kwargs: dict | None,
        ):
        try: 

            kwargs = build_kwargs(sampling_params, pydantic_model, extra_body, kwargs)

            # Define nested async for client
            async def generate(
                history: str | None,
                text: str | list[str] | None, 
                image: str | list[str] | None,
                audio: str | list[str] | None,
                ) -> str:

                try: 
                    messages = build_messages(history, text, image, audio)
                    result = await self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        **kwargs,
                    )
                except Exception as e:
                    raise e 

                # Fallback to stringifying whatever we got
                return json.dumps(result.choices[0].message.content)

            async def gather_completions(histories, texts, images, audios) -> list[str]:
                tasks = [
                    generate(h, t, i, a)
                    for h, t, i, a in zip_longest(histories, texts, images, audios, fillvalue=None)
                ]
                return await asyncio.gather(*tasks)

            
            texts = text_col.to_pylist() if text_col is not None else []
            images = image_col.to_pylist() if image_col is not None else []
            audios = audio_col.to_pylist() if audio_col is not None else []
            histories = history_col.to_pylist() if history_col is not None else []

            results = self.loop.run_until_complete(gather_completions(histories, texts, images, audios))

            return results
        finally:
            self.loop.close()

@daft.func()
def append_response_to_history(history: str, response: str) -> str:
    history = json.loads(history)
    history.append({"role": "assistant", "content": response})
    return json.dumps(history)

    
if __name__ == "__main__":
    # Desired Usage Pattern
    from dotenv import load_dotenv

    load_dotenv()

    image_url_rhino = "https://d1jyxxz9imt9yb.cloudfront.net/medialib/5483/image/s1300x1300/XKP144_reduced.jpg"
    image_url_tiger = "https://transforms.stlzoo.org/production/animals/amur-tiger-01-01.jpg?w=1200&h=1200&auto=compress%2Cformat&fit=crop&dm=1658935145&s=95d03aceddd44dc8271beed46eae30bc"
    bird_vocalization = "https://www.allaboutbirds.org/guide/assets/sound/550417.mp3"
   
    df = daft.from_pylist([{
        "history": json.dumps([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]),
        "texts": ["I've always loved Paris.", "Anyways, what kinds of animals are in the images?"],
        "images": [
            image_url_rhino,
            image_url_tiger,
        ],
    }])

    # Test with google/gemma-3-4b-it
    df = df.with_column(
        "response_to_images", 
        openai_chat(
            model_id="google/gemini-2.5-flash", 
            history_col=df["history"], 
            text_col=df["texts"],
            image_col=df["images"],
        )
    )
    print(json.dumps(df.select("response_to_images").to_pydict(), indent=2, ensure_ascii=False))

    df = (
        df
        .with_column("history", append_response_to_history(daft.col("history"), daft.col("response_to_images")))
        .with_column(
            "response_to_audio", 
            openai_chat(
                model_id="openai/gpt-4o-audio-preview", 
                history_col=daft.col("history"), 
                text_col = daft.lit("Now given the audio can you tell me what kind of animal it is?"),
                audio_col=daft.lit(bird_vocalization),
            )
        )
    )
    print(json.dumps(df.select("response_to_audio").to_pydict(), indent=2, ensure_ascii=False))