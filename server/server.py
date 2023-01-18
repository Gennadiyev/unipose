"""Server for UniPose"""

import io
from typing import Union
import uuid
from flask import Flask, Response, request, jsonify, send_file
import argparse
import base64
import os
import time
from datetime import datetime
import re
import orjson
import imageio.v2 as imageio
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from easydict import EasyDict
from unipose.datasets import MPIIDataset, COCODataset, AnimalKingdomDataset, AP10KDataset

from unipose.models import UniPose

@logger.catch
def draw_skel(
    src_image: bytes,
    heatmaps: torch.Tensor,
    font_path: str,
    threshold: float = 0,
    output_size: int = 512,
) -> bytes:
    """Draws the skeleton on the image."""
    base_image = Image.open(io.BytesIO(src_image))
    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")
    base_image = base_image.resize((output_size, output_size), resample=Image.Resampling.BICUBIC)
    # Generate output
    x = []
    y = []
    confidence = []
    base_image_draw = ImageDraw.Draw(base_image, "RGBA")

    # Color preset: e54bd6-59aff9-8fefb7-fc906c-ccbf0e
    draw_config = {
        "font": font_path,
        "colors": {
            "head": "#e54bd6",
            "arm_left": "#59aff9",
            "arm_right": "#8fefb7",
            "leg_left": "#fc906c",
            "leg_right": "#ccbf0e",
            "shadow": "#e54bd6",
            "border": "#ffffff",
        },
        "radius": 7,
        "line_width": 3,
        "caption": True,
    }
    __radius = draw_config["radius"]
    __line_width = draw_config["line_width"]
    __caption = draw_config["caption"]
    __colors = draw_config["colors"]
    __font = ImageFont.truetype(draw_config["font"], 20) if os.path.exists(draw_config["font"]) else None
    for i in range(heatmaps.shape[1]):
        _image_arr = heatmaps[0, i, :, :].cpu().numpy()
        # _image_arr = (_image_arr - _image_arr.min()) / (_image_arr.max() - _image_arr.min())
        idx = np.argmax(_image_arr)
        conf = np.max(_image_arr)
        if conf < threshold:
            x.append(0)
            y.append(0)
            confidence.append(0)
        else:
            x.append(idx % _image_arr.shape[0] * output_size // _image_arr.shape[1])
            y.append(idx // _image_arr.shape[0] * output_size // _image_arr.shape[0])
            confidence.append(conf)
    if sum(x) == 0 and sum(y) == 0:
        logger.warning("No keypoints in heatmaps: All keypoints are (0, 0). Will not render skeleton.")

    # Draw lines between joints
    eps = 1e-6
    if (x[1] + y[1] > eps) and (x[2] + y[2] > eps):
        base_image_draw.line((x[1], y[1], x[2], y[2]), fill=__colors["arm_left"], width=__line_width)
    if (x[2] + y[2] > eps) and (x[3] + y[3] > eps):
        base_image_draw.line((x[2], y[2], x[3], y[3]), fill=__colors["arm_left"], width=__line_width)
    if (x[4] + y[4] > eps) and (x[5] + y[5] > eps):
        base_image_draw.line((x[4], y[4], x[5], y[5]), fill=__colors["arm_right"], width=__line_width)
    if (x[5] + y[5] > eps) and (x[6] + y[6] > eps):
        base_image_draw.line((x[5], y[5], x[6], y[6]), fill=__colors["arm_right"], width=__line_width)
    if (x[7] + y[7] > eps) and (x[8] + y[8] > eps):
        base_image_draw.line((x[7], y[7], x[8], y[8]), fill=__colors["leg_left"], width=__line_width)
    if (x[8] + y[8] > eps) and (x[9] + y[9] > eps):
        base_image_draw.line((x[8], y[8], x[9], y[9]), fill=__colors["leg_left"], width=__line_width)
    if (x[10] + y[10] > eps) and (x[11] + y[11] > eps):
        base_image_draw.line((x[10], y[10], x[11], y[11]), fill=__colors["leg_right"], width=__line_width)
    if (x[11] + y[11] > eps) and (x[12] + y[12] > eps):
        base_image_draw.line((x[11], y[11], x[12], y[12]), fill=__colors["leg_right"], width=__line_width)

    # For each joint, draw a circle
    for i in range(13):
        if x[i] == 0 and y[i] == 0:
            continue
        if i in [0]:
            # Head
            base_image_draw.ellipse(
                (x[i] - __radius, y[i] - __radius, x[i] + __radius, y[i] + __radius),
                fill=__colors["head"],
                outline=__colors["border"],
            )
        elif i in [1, 2, 3]:
            # Left arm
            base_image_draw.ellipse(
                (x[i] - __radius, y[i] - __radius, x[i] + __radius, y[i] + __radius),
                fill=__colors["arm_left"],
                outline=__colors["border"],
            )
        elif i in [4, 5, 6]:
            # Right arm
            base_image_draw.ellipse(
                (x[i] - __radius, y[i] - __radius, x[i] + __radius, y[i] + __radius),
                fill=__colors["arm_right"],
                outline=__colors["border"],
            )
        elif i in [7, 8, 9]:
            # Left leg
            base_image_draw.ellipse(
                (x[i] - __radius, y[i] - __radius, x[i] + __radius, y[i] + __radius),
                fill=__colors["leg_left"],
                outline=__colors["border"],
            )
        elif i in [10, 11, 12]:
            # Right leg
            base_image_draw.ellipse(
                (x[i] - __radius, y[i] - __radius, x[i] + __radius, y[i] + __radius),
                fill=__colors["leg_right"],
                outline=__colors["border"],
            )
        if __caption:
            if x[i] < output_size - 160 and y[i] < output_size - 20:
                base_image_draw.text(
                    (x[i] + __radius, y[i] + __radius),
                    "{:.2f}@{}".format(confidence[i], i),
                    fill="white",
                    font=__font,
                    anchor="la",
                )
            else:
                base_image_draw.text(
                    (x[i] - __radius, y[i] - __radius),
                    "{:.2f}@{}".format(confidence[i], i),
                    fill="white",
                    font=__font,
                    anchor="rs",
                )

    # Render to bytes
    output_image = io.BytesIO()
    base_image.save(output_image, format="png")
    return output_image.getvalue()

class Backend:

    def __init__(self):
        pass

    def start(self, config: dict={}):
        self.cwd = os.getcwd()
        self._status = "NOMODEL"
        self.model_name = ""
        self.config = EasyDict(config)
        self.models = self._refresh_models()
        self.gpu_id = self.config.default_device
        self.model = UniPose(13, resnet_layers=[3, 8, 36, 3]).to(self.device)
        os.makedirs(self.config.paths.cache_path, exist_ok=True)

    @property
    def device(self):
        if self.gpu_id == -1:
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{self.gpu_id}")

    def _refresh_models(self):
        model_dir = self.config.paths.checkpoints
        model_naming_convention = r"^model_run-([0-9a-f]+)_ep-(\d+).pth$"
        models = []
        for file_name in os.listdir(model_dir):
            match = re.match(model_naming_convention, file_name)
            if match:
                models.append({
                    "run_id": match.group(1),
                    "epoch": int(match.group(2)),
                    "name": file_name
                })
        if len(models) > 0:
            logger.info("Found {} models under {}", len(models), model_dir)
        else:
            logger.warning("No models found under {}", model_dir)
        return models

    @property
    def status_string(self):
        if self._status == "NOMODEL":
            return "No model loaded"
        elif self._status == "LOADING":
            return "Loading checkpoint"
        elif self._status == "READY":
            return "Ready"
        elif self._status == "RUNNING":
            return "Inferencing"
        else:
            return "Unknown"

    def list_models(self):
        return self.models
    
    def load_checkpoint(self, checkpoint_name):
        if self.model_name == checkpoint_name:
            logger.debug("Same checkpoint, skipping")
            return
        logger.info("Loading checkpoint {}...", checkpoint_name)
        model_dir = self.config.paths.checkpoints
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model_name = checkpoint_name
        self._status = "READY"
        logger.success("Successfully loaded checkpoint {}", checkpoint_name)

    def _base64_to_bytes(self, b64_string: str):
        return base64.b64decode(b64_string)

    def _base64_image_to_bytes(self, src: str):
        """
        Decode image from base64 string.
        :param src: Encoded image
            eg:
                src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
                    yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
                    ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
                    LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
                    k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
                    ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

        :return: extension, image_data, uuid
        """
        # Extract extension and base64 string
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            b64_string = result.groupdict().get("data")
            assert type(b64_string) == str, "b64_string is not a string"
        else:
            raise Exception("Cannot parse ext and data")

        img_bytes = base64.urlsafe_b64decode(b64_string) # type: bytes

        return img_bytes


    def set_gpu(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._move_model_to_device()
    
    def _move_model_to_device(self):
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            logger.error("Failed to move model to device: {}", e)

    def _uuid(self):
        return str(uuid.uuid4())

    def inference(self, image_bytes: bytes):
        """Inference from raw image bytes.

        @param image_bytes: raw image bytes
        @return: inference result
        """
        logger.info("Inferencing...")
        self._status = "RUNNING"
        rid = self._uuid()
        cache_path = self.config.paths.cache_path
        # Dump image bytes to cache path
        with open(os.path.join(cache_path, f"{rid}.jpg"), "wb") as f:
            f.write(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((256, 256))
        image = np.array(image)
        # Remove alpha channel
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tic = time.time()
            output = self.model(image)
            toc = time.time()
        # Draw skel
        output_bytes = draw_skel(
            image_bytes,
            output,
            font_path=self.config.paths.font,
            threshold=0,
            output_size=512
        )
        self._status = "READY"
        # Detach output and convert to numpy array
        output_detached = output.detach().cpu().numpy()
        return output_detached, output_bytes, toc - tic, rid


flask_app = Flask(__name__)
flask_app.config['JSON_AS_ASCII'] = False
backend = Backend()

cwd = os.getcwd()
def get_abs_path(dir_path: str, create_if_not_exists: bool = False):
    """Gets the absolute path of any path: relative or absolute."""
    if not os.path.isabs(dir_path):
        _path = os.path.join(cwd, dir_path)
        if create_if_not_exists:
            os.makedirs(_path, exist_ok=True)
        return _path
    else:
        if create_if_not_exists:
            os.makedirs(dir_path, exist_ok=True)
        return dir_path

@flask_app.route('/api/v1/set_gpu', methods=['POST'])
def set_gpu():
    request_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = request.get_json()
    tic = time.time()
    backend.set_gpu(params['gpu_id'])
    toc = time.time()
    return jsonify({
        "status": "ok",
        "request_time": request_received_at,
        "time_elapsed": toc - tic
    })

@flask_app.route('/api/v1/models', methods=['GET'])
def list_models():
    request_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tic = time.time()
    models = backend.list_models()
    toc = time.time()
    return jsonify({
        "status": "ok",
        "request_time": request_received_at,
        "time_elapsed": toc - tic,
        "models": models
    })

@flask_app.route('/api/v1/load_checkpoint', methods=['POST'])
def load_model():
    request_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = request.get_json()
    tic = time.time()
    backend.load_checkpoint(params['name'])
    toc = time.time()
    return jsonify({
        "status": "ok",
        "request_time": request_received_at,
        "time_elapsed": toc - tic
    })

@flask_app.route('/api/v1/inference', methods=['POST'])
def inference():
    request_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = request.get_json()
    tic = time.time()
    output_np_array, skel_image, time_elapsed, rid = backend.inference(backend._base64_image_to_bytes(params['image']))
    # output = output_np_array.tolist()
    output_image_path = os.path.join(backend.config.paths.cache_path, f"{rid}_skel.png")
    with open(output_image_path, 'wb') as f:
        f.write(skel_image)
    toc = time.time()
    logger.success("Skel image saved to {}", output_image_path)
    return jsonify({
        "status": "ok",
        "request_time": request_received_at,
        "time_elapsed": toc - tic,
        "inference_time": time_elapsed,
        # "skel_array": output,
        "url": f"/output?id={rid}"
    })
    
@flask_app.route('/output', methods=['GET'])
def output():
    request_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tic = time.time()
    rid = request.args.get('id')
    output_image_path = os.path.join(backend.config.paths.cache_path, f"{rid}_skel.png")
    with open(output_image_path, 'rb') as f:
        image_bytes = f.read()
    toc = time.time()
    return Response(image_bytes, mimetype='image/png')

# Set static folder
flask_app.static_folder = get_abs_path('static')
flask_app.static_url_path = '/static'

@flask_app.route('/index.html', methods=['GET'])
def index():
    # Serve index.html
    return flask_app.send_static_file('index.html')

# Run flask_app
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", default="config.json", help="configuration file")
    config_file = parser.parse_args().config_file
    
    try:
        with open(get_abs_path(config_file), 'rb') as f:
            config = orjson.loads(f.read())
    except orjson.JSONDecodeError as _:
        logger.error("Aborting: invalid {} found under {}", cwd)
        exit(-1)
    except Exception as e:
        logger.error("Aborting: {}", e)
        exit(-1)

    logger.info("Loading checkpoint skeleton...")
    backend.start(config)
    logger.success("Ready to serve!")
    flask_app.run(host=config['host'], port=config['port'])
