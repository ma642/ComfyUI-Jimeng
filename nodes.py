import os
import io
import re
import time
import base64
import requests
import shutil
import time

import numpy
import PIL

from volcenginesdkarkruntime import Ark

import folder_paths
from comfy_api.util import VideoContainer

GLOBAL_CATEGORY = "JimengAI"

def _fetch_data_from_url(url, stream=True):
    return requests.get(url, stream=stream).content

def _tensor2images(tensor):
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0.0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]
def _encode_image(img, mask=None):
    if mask is not None:
        img = img.copy()
        img.putalpha(mask)
    with io.BytesIO() as bytes_io:
        if mask is not None:
            img.save(bytes_io, format='PNG')
        else:
            img.save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return data_bytes

def _image_to_base64(image):
    if image is None:
        return None
    return base64.b64encode(_encode_image(_tensor2images(image)[0])).decode("utf-8")

class JimengAPIClient:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    RETURN_TYPES = ("JIMENG_API_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_client"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    def create_client(self, api_key):
        client = Ark(api_key=api_key)
        return (client,)

class JimengImage2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "image": ("IMAGE",),
                "model": (["doubao-seedance-1-0-lite-i2v-250428", "doubao-seedance-1-0-pro-250528"], {"default": "doubao-seedance-1-0-lite-i2v-250428"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # video_url, task_id
    RETURN_NAMES = ("url", "task_id")

    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    def generate(self,
                 client,
                 image,
                 model="doubao-seedance-1-0-lite-i2v-250428",
                 prompt="",
                 duration="5",
                 resolution="720p",
                 camerafixed=True):
        
        # to base   
        base64_str = _image_to_base64(image)
        data_url = f"data:image/jpeg;base64,{base64_str}"

        # image to video prompt
        prompt_string = f"{prompt} --resolution {resolution} --dur {duration} --camerafixed {'true' if camerafixed else 'false'}"

        # create task
        try:
            create_result = client.content_generation.tasks.create(
                model=model,
                content=[
                    {"type": "text", "text": prompt_string},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            )
            task_id = create_result.id
        except Exception as e:
            raise RuntimeError(f"create task failed: {e}")

        # query task status
        timeout = 300 
        interval = 5
        for _ in range(timeout // interval):
            time.sleep(interval)
            try:
                get_result = client.content_generation.tasks.get(task_id=task_id)
                if get_result.status == "succeeded":
                    return (get_result.content.video_url, task_id)
                elif get_result.status in ["failed", "cancelled"]:
                    raise RuntimeError(f"task failed, task_id={task_id}")
            except Exception as e:
                print(f"get task failed, retry continue: {e}")
                continue

        return ("", task_id)
    

class JimengFirstLastFrame2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "first_frame_image": ("IMAGE",),  # First frame image input
                "last_frame_image": ("IMAGE",),   # Last frame image input, optional
                "model": (["doubao-seedance-1-0-lite-i2v-250428"], {"default": "doubao-seedance-1-0-lite-i2v-250428"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # video_url, task_id
    RETURN_NAMES = ("url", "task_id")

    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def generate(self,
                 client,
                 first_frame_image,
                 last_frame_image,
                 model="doubao-seedance-1-0-lite-i2v-250428",
                 prompt="",
                 duration="5",
                 resolution="720p",
                 camerafixed=True):

        # Convert images to base64
        first_frame_base64 = _image_to_base64(first_frame_image)
        first_frame_data_url = f"data:image/jpeg;base64,{first_frame_base64}"
        
        last_frame_base64 = _image_to_base64(last_frame_image)
        last_frame_data_url = f"data:image/jpeg;base64,{last_frame_base64}"
        
        content = [
            {"type": "text", "text": f"{prompt} --resolution {resolution} --dur {duration} --camerafixed {'true' if camerafixed else 'false'}"},
            {"type": "image_url", "image_url": {"url": first_frame_data_url}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": last_frame_data_url}, "role": "last_frame"}
        ]

        # Create task
        try:
            create_result = client.content_generation.tasks.create(
                model=model,
                content=content
            )
            task_id = create_result.id
        except Exception as e:
            raise RuntimeError(f"Create task failed: {e}")

        # Query task status
        timeout = 300 
        interval = 5
        for _ in range(timeout // interval):
            time.sleep(interval)
            try:
                get_result = client.content_generation.tasks.get(task_id=task_id)
                if get_result.status == "succeeded":
                    return (get_result.content.video_url, task_id)
                elif get_result.status in ["failed", "cancelled"]:
                    raise RuntimeError(f"Task failed, task_id={task_id}")
            except Exception as e:
                print(f"Get task failed, retry continue: {e}")
                continue

        return ("", task_id)


class PreviewVideoFromUrl:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "tmp_preview"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "format": (VideoContainer.as_input(), {"default": "mp4"}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "image/video"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def run(self, video_url, filename_prefix, save_output, format):
        if not video_url or not save_output:
            return {"ui": {"video_url": [video_url]}, "result": ('', )}

        output_dir = self.output_dir
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1

        file_ext = VideoContainer.get_extension(format)
        final_filename = f"{filename}_{counter:05}_.{file_ext}"
        final_path = os.path.join(full_output_folder, final_filename)

        if isinstance(video_url, list):
            video_url = video_url[0]

        if video_url.startswith("http://") or video_url.startswith("https://"):
            
            try:
                data = _fetch_data_from_url(video_url)
            except Exception as e:
                raise RuntimeError(f"failed to download video from url: '{video_url}': {e}")
            
            with open(final_path, "wb") as f:
                f.write(data)
        else:
            if not os.path.isfile(video_url):
                raise FileNotFoundError(f"local file not found: {video_url}")
            try:
                shutil.copy(video_url, final_path)
            except Exception as e:
                raise RuntimeError(f"failed to copy video file to output path: {final_path}, {e}, ")

        results = [{
            "filename": final_filename,
            "subfolder": subfolder,
            "type": self.type
        }]

        return {
            "ui": {
                "images": results,
                "animated": (True,)
            }
        }


NODE_CLASS_MAPPINGS = {
    "JimengAPIClient": JimengAPIClient,
    "JimengImage2Video": JimengImage2Video,
    "JimengFirstLastFrame2Video": JimengFirstLastFrame2Video,
    "PreviewVideoFromUrl": PreviewVideoFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "JimengAPIClient",
    "JimengImage2Video": "JimengImage2Video",
    "JimengFirstLastFrame2Video": "JimengFirstLastFrame2Video",
    "PreviewVideoFromUrl": "PreviewVideoFromUrl",
}