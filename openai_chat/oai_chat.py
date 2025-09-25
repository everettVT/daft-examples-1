import daft
from daft import col

from openai import AsyncOpenAI


import asyncio
from openai import AsyncOpenAI
import daft
import json
import base64
import binascii
from typing import Any, Literal
import pydantic

load_dotenv()

client = AsyncOpenAI()
model_id = "gpt-5-nano"

def url_or_base64(image: str | np.ndarray | bytes):
    if isinstance(image, str):
        if image.startswith("http"):
            return {"url": image}
        elif 
    return {"url": f"data:image/png;base64,{image}"}

def build_messages(
    history: str | None,
    texts:    str | list[str] ,
    images:   str | list[str] | None,
    audio:    str | list[str] | None,
    files:    str | list[str] | None,
    ) -> list[dict]:

    messages = []
    content = []

    if history: 
        messages = json.loads(history)
        
    if texts:
        content.extend([{"type":"text", "text": t} for t in texts])
    
    if images: 
       content.extend([{"type":"image_url", "image_url": url_or_base64(i)} for i in images])

    if audio:
        content.extend
    
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

    kwargs = {}
    
    if sampling_params:
        kwargs.update(sampling_params)
    
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "",
                "schema": pydantic_model.model_json_schema(),
            },
        }
    
    if extra_body:
        kwargs["extra_body"] = extra_body
    
    if kwargs:
        kwargs.update(kwargs)
    
    return kwargs

async def generate(
    client: AsyncOpenAI,
    model_id: str,
    history: str | None,
    text: str | list[str] | None, 
    image: str | list[str] | None,
    audio: str | list[str] | None,
    file: str | list[str] | None,
    json_schema: dict | None,
    extra_body: dict | None,
    sampling_params: dict | None,
    kwargs: dict | None,
    ) -> str:

    messages = build_messages(history, text, image, audio, file)

    kwargs = build_kwargs(sampling_params, json_schema, extra_body, kwargs)
    
    result = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        **kwargs,
    )

    # Fallback to stringifying whatever we got
    return json.dumps(result.choices[0].message.content)

@daft.udf(return_dtype=daft.DataType.string())
class openai_chat:
    def __init__(self, base_url: str, api_key: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def __call__(self,
        model_id: str = "openai/gpt-5-nano",
        history_col: daft.DataType.string() | None = None,
        text_col:  daft.DataType.string() | None = None,
        image_col: daft.DataType.image() | None = None,
        audio_col: daft.DataType.audio() | None = None,
        file_col:  daft.DataType.file() | None = None,
        sampling_params: dict[str, Any] | None = None,
        pydantic_model: pydantic.BaseModel | None = None,
        extra_body: dict | None = None
        ) -> list[str]:

        async def gather_completions(history, texts, images) -> list[str]:
            tasks = [generate(h, t, i, , extra_body, sampling_params) for h, t, i in zip(history, texts, images)]
            return await asyncio.gather(*tasks)

        texts = text_col.to_pylist() if text_col else None
        images = image_col.to_pylist() if image_col else None
        audios = audio_col.to_pylist() if audio_col else None
        history = history_col.to_pylist() if history_col else None

        results = self.loop.run_until_complete(gather_completions(history, texts, images))

        return results

    



if __name__ == "__main__":
    # Desired Usage Pattern
    from dotenv import load_dotenv

    df = daft.from_pydict({
        "history": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        "texts": ["I've always loved Paris.", "The view from the Eiffel Tower is breathtaking."],
        "images": ["https://example.com/image.png", "https://example.com/image.jpg", "https://example.com/image.webp", "https://example.com/image.giff", "https://example.com/image.jpg", "https://example.com/image.webp"],
    })

    result = df.with_column("result", openai_chat(model_id="openai/gpt-5-nano", history_col=df["role"], text_col=df["content"]))
    print(result.to_pydict())