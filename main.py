import base64
from loguru import logger

import torch
import cv2
import pyvirtualcam
import numpy as np
from PIL import Image

import tha2.poser.modes.mode_20_wx
from models import TalkingAnime3
from utils import preprocessing_image
from action_animeV2 import ActionAnimeV2
from alive import Alive
from multiprocessing import Value, Process, Queue
from ctypes import c_bool
import os

import queue
import time
import math
import collections
from collections import OrderedDict
from args import args
from tha3.util import torch_linear_to_srgb
from pyanime4k import ac

# 新增：导入 logging 和 Flask 的 Response, render_template_string
import logging
from flask import Flask, Response, render_template_string
from flask_restful import Resource, Api, reqparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

fps_delay = 0.01

app = Flask(__name__)
api = Api(app)


def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0


ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()


class ModelClientProcess(Process):
    def __init__(self, input_image, device, model_process_args):
        super().__init__()
        self.device = device
        self.should_terminate = Value('b', False)
        self.updated = Value('b', False)
        self.data = None
        self.input_image = input_image
        self.output_queue = model_process_args['output_queue']
        self.input_queue = model_process_args['input_queue']
        self.model_fps_number = Value('f', 0.0)
        self.gpu_fps_number = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)

        self.input_image_q = model_process_args['input_image_q']

    def run(self):
        model = TalkingAnime3().to(self.device)
        model = model.eval()
        logger.info("模型进程: 预训练模型已加载")

        eyebrow_vector = torch.empty(1, 12, dtype=torch.half if args.model.endswith('half') else torch.float)
        mouth_eye_vector = torch.empty(1, 27, dtype=torch.half if args.model.endswith('half') else torch.float)
        pose_vector = torch.empty(1, 6, dtype=torch.half if args.model.endswith('half') else torch.float)

        input_image = self.input_image.to(self.device)
        eyebrow_vector = eyebrow_vector.to(self.device)
        mouth_eye_vector = mouth_eye_vector.to(self.device)
        pose_vector = pose_vector.to(self.device)

        model_cache = OrderedDict()
        tot = 0
        hit = 0
        hit_in_a_row = 0
        model_fps = FPS()
        gpu_fps = FPS()
        cur_sec = int(time.perf_counter())
        fps_num = 0
        while True:
            # time.sleep(fps_delay)
            if int(time.perf_counter()) == cur_sec:
                fps_num += 1
            else:
                fps_num = 0
                cur_sec = int(time.perf_counter())

            if not self.input_image_q.empty():
                input_image = self.input_image_q.get_nowait().to(self.device)
                model.face_cache = OrderedDict()
                model_cache = OrderedDict()
                logger.info("模型进程: 角色图像已更新")

            model_input = None
            try:
                while not self.input_queue.empty():
                    model_input = self.input_queue.get_nowait()
            except queue.Empty:
                continue
            if model_input is None:
                continue
            simplify_arr = [1000] * ifm_converter.pose_size
            if args.simplify >= 1:
                simplify_arr = [200] * ifm_converter.pose_size
                simplify_arr[ifm_converter.eye_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_surprised_left_index] = 30
                simplify_arr[ifm_converter.eye_surprised_right_index] = 30
                simplify_arr[ifm_converter.iris_rotation_x_index] = 25
                simplify_arr[ifm_converter.iris_rotation_y_index] = 25
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
            if args.simplify >= 2:
                simplify_arr[ifm_converter.head_x_index] = 100
                simplify_arr[ifm_converter.head_y_index] = 100
                simplify_arr[ifm_converter.eye_surprised_left_index] = 10
                simplify_arr[ifm_converter.eye_surprised_right_index] = 10
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_left_index]
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_right_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_wink_right_index] / 2
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_right_index] / 2

                uosum = model_input[ifm_converter.mouth_uuu_index] + \
                        model_input[ifm_converter.mouth_ooo_index]
                model_input[ifm_converter.mouth_ooo_index] = uosum
                model_input[ifm_converter.mouth_uuu_index] = 0
                is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
                    ifm_converter.mouth_iii_index] + uosum) > 0
                model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
                model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
                model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
                model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
            if args.simplify >= 3:
                simplify_arr[ifm_converter.iris_rotation_x_index] = 20
                simplify_arr[ifm_converter.iris_rotation_y_index] = 20
                simplify_arr[ifm_converter.eye_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_wink_right_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
            if args.simplify >= 4:
                simplify_arr[ifm_converter.head_x_index] = 50
                simplify_arr[ifm_converter.head_y_index] = 50
                simplify_arr[ifm_converter.neck_z_index] = 100
                model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
                model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
                simplify_arr[ifm_converter.iris_rotation_x_index] = 10
                simplify_arr[ifm_converter.iris_rotation_y_index] = 10
                simplify_arr[ifm_converter.eye_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_surprised_left_index] = 8
                simplify_arr[ifm_converter.eye_surprised_right_index] = 8
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_wink_right_index]
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2

                model_input[ifm_converter.eye_surprised_left_index] += model_input[
                    ifm_converter.eye_surprised_right_index]
                model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                           ifm_converter.eye_surprised_left_index] / 2
                model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                          ifm_converter.eye_surprised_left_index] / 2

                model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.mouth_aaa_index] = min(
                    model_input[ifm_converter.mouth_aaa_index] +
                    model_input[ifm_converter.mouth_ooo_index] / 2 +
                    model_input[ifm_converter.mouth_iii_index] / 2 +
                    model_input[ifm_converter.mouth_uuu_index] / 2, 1
                )
                model_input[ifm_converter.mouth_ooo_index] = 0
                model_input[ifm_converter.mouth_iii_index] = 0
                model_input[ifm_converter.mouth_uuu_index] = 0
            for i in range(4, args.simplify):
                simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
            for i in range(0, len(simplify_arr)):
                if simplify_arr[i] > 0:
                    model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]
            input_hash = hash(tuple(model_input))
            cached = model_cache.get(input_hash)
            tot += 1
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            if cached is not None and hit_in_a_row < self.model_fps_number.value:
                self.output_queue.put(cached)
                model_cache.move_to_end(input_hash)
                hit += 1
                hit_in_a_row += 1
            else:
                hit_in_a_row = 0
                if args.eyebrow:
                    for i in range(12):
                        eyebrow_vector[0, i] = model_input[i]
                        eyebrow_vector_c[i] = model_input[i]
                for i in range(27):
                    mouth_eye_vector[0, i] = model_input[i + 12]
                    mouth_eye_vector_c[i] = model_input[i + 12]
                for i in range(6):
                    pose_vector[0, i] = model_input[i + 27 + 12]
                if model is None:
                    output_image = input_image
                else:
                    output_image = model(input_image, mouth_eye_vector, pose_vector, eyebrow_vector, mouth_eye_vector_c,
                                         eyebrow_vector_c,
                                         self.gpu_cache_hit_ratio)
                postprocessed_image = output_image[0].float()
                postprocessed_image = convert_linear_to_srgb((postprocessed_image + 1.0) / 2.0)
                c, h, w = postprocessed_image.shape
                postprocessed_image = 255.0 * torch.transpose(postprocessed_image.reshape(c, h * w), 0, 1).reshape(h, w,
                                                                                                                   c)
                postprocessed_image = postprocessed_image.byte().detach().cpu().numpy()

                self.output_queue.put(postprocessed_image)
                if args.debug:
                    self.gpu_fps_number.value = gpu_fps()
                if args.max_cache_len > 0:
                    model_cache[input_hash] = postprocessed_image
                    if len(model_cache) > args.max_cache_len:
                        model_cache.popitem(last=False)
            if args.debug:
                self.model_fps_number.value = model_fps()
                self.cache_hit_ratio.value = hit / tot


def prepare_input_img(IMG_WIDTH, charc):
    if os.path.exists(charc):
        img = Image.open(charc)
    else:
        img = Image.open(f"data/images/{charc}.png")
    img = img.convert('RGBA')
    wRatio = img.size[0] / IMG_WIDTH
    img = img.resize((IMG_WIDTH, int(img.size[1] / wRatio)))
    for i, px in enumerate(img.getdata()):
        if px[3] <= 0:
            y = i // IMG_WIDTH
            x = i % IMG_WIDTH
            img.putpixel((x, y), (0, 0, 0, 0))
    input_image = preprocessing_image(img.crop((0, 0, IMG_WIDTH, IMG_WIDTH)))
    if args.model.endswith('half'):
        input_image = torch.from_numpy(input_image).half() * 2.0 - 1
    else:
        input_image = torch.from_numpy(input_image).float() * 2.0 - 1
    input_image = input_image.unsqueeze(0)
    extra_image = None
    if img.size[1] > IMG_WIDTH:
        extra_image = np.array(img.crop((0, IMG_WIDTH, img.size[0], img.size[1])))
    logger.info(f"角色图像已加载: {charc}")
    return input_image, extra_image


# 修改 EasyAIV, 增加 web_output_q
class EasyAIV(Process):
    def __init__(self, model_process_args, alive_args, web_output_q):
        super().__init__()
        # self.extra_image = extra_image

        self.model_process_input_queue = model_process_args['input_queue']
        self.model_process_output_queue = model_process_args['output_queue']

        # 新增: Web视频流输出队列
        self.web_output_q = web_output_q

        self.alive_args_is_speech = alive_args['is_speech']
        self.alive_args_speech_q = alive_args['speech_q']

        self.alive_args_is_singing = alive_args['is_singing']
        self.alive_args_is_music_play = alive_args['is_music_play']
        self.alive_args_beat_q = alive_args['beat_q']
        self.alive_args_mouth_q = alive_args['mouth_q']

    @torch.no_grad()
    def run(self):
        IMG_WIDTH = 512

        cam = None
        if args.output_webcam:
            cam_scale = 1
            cam_width_scale = 1
            if args.anime4k:
                cam_scale = 2
            if args.alpha_split:
                cam_width_scale = 2
            try:
                cam = pyvirtualcam.Camera(width=args.output_w * cam_scale * cam_width_scale,
                                          height=args.output_h * cam_scale,
                                          fps=30,
                                          backend=args.output_webcam,
                                          fmt=
                                          {'unitycapture': pyvirtualcam.PixelFormat.RGBA,
                                           'obs': pyvirtualcam.PixelFormat.RGB}[
                                              args.output_webcam])
                logger.info(f'主进程: 虚拟摄像头已启动: {cam.device}')
            except Exception as e:
                logger.error(f"主进程: 启动虚拟摄像头失败: {e}")
                cam = None

        a = None
        if args.anime4k:
            try:
                parameters = ac.Parameters()
                # enable HDN for ACNet
                parameters.HDN = True
                a = ac.AC(
                    managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
                    type=ac.ProcessorType.OpenCL_ACNet,
                )
                a.set_arguments(parameters)
                logger.info("主进程: Anime4K 已加载")
            except Exception as e:
                logger.error(f"主进程: 加载 Anime4K 失败: {e}")
                a = None
                args.anime4k = False

        position_vector = [0, 0, 0, 1]
        model_output = None
        speech_q = None
        mouth_q = None
        beat_q = None

        action = ActionAnimeV2()
        idle_start_time = time.perf_counter()

        logger.info("主进程: 准备就绪。关闭此控制台以退出。")

        while True:
            # time.sleep(fps_delay)

            current_action = "idle"  # 用于日志记录
            idle_flag = False
            if bool(self.alive_args_is_speech.value):  # 正在说话
                if not self.alive_args_speech_q.empty():
                    speech_q = self.alive_args_speech_q.get_nowait()
                eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = action.speaking(speech_q)
                current_action = "speaking"
            elif bool(self.alive_args_is_singing.value):  # 正在唱歌
                if not self.alive_args_beat_q.empty():
                    beat_q = self.alive_args_beat_q.get_nowait()
                if not self.alive_args_mouth_q.empty():
                    mouth_q = self.alive_args_mouth_q.get_nowait()
                eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = action.singing(beat_q, mouth_q)
                current_action = "singing"
            elif bool(self.alive_args_is_music_play.value):  # 摇子
                if not self.alive_args_beat_q.empty():
                    beat_q = self.alive_args_beat_q.get_nowait()
                eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = action.rhythm(beat_q)
                current_action = "rhythm"
            else:  # 空闲状态
                speech_q = None
                mouth_q = None
                beat_q = None
                idle_flag = True
                if args.sleep != -1 and time.perf_counter() - idle_start_time > args.sleep:  # 空闲20秒就睡大觉
                    eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = action.sleeping()
                    current_action = "sleeping"
                else:
                    eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = action.idle()
                    current_action = "idle"

            # 新增日志
            # logger.info(f"主进程: 当前动作 - {current_action}")

            if not idle_flag:
                idle_start_time = time.perf_counter()

            pose_vector_c[3] = pose_vector_c[1]
            pose_vector_c[4] = pose_vector_c[2]

            model_input_arr = eyebrow_vector_c
            model_input_arr.extend(mouth_eye_vector_c)
            model_input_arr.extend(pose_vector_c)

            self.model_process_input_queue.put_nowait(model_input_arr)

            has_model_output = 0
            try:
                new_model_output = model_output
                while not self.model_process_output_queue.empty():
                    has_model_output += 1
                    new_model_output = self.model_process_output_queue.get_nowait()
                model_output = new_model_output
            except queue.Empty:
                pass
            if model_output is None:
                logger.warning("主进程: 尚未收到模型输出，等待1秒...")
                time.sleep(1)
                continue

            postprocessed_image = model_output

            k_scale = 1
            rotate_angle = 0
            dx = 0
            dy = 0
            if args.extend_movement:
                k_scale = position_vector[2] * math.sqrt(args.extend_movement) + 1
                rotate_angle = -position_vector[0] * 10 * args.extend_movement
                dx = position_vector[0] * 400 * k_scale * args.extend_movement
                dy = -position_vector[1] * 600 * k_scale * args.extend_movement
            if args.bongo:
                rotate_angle -= 5
            rm = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_WIDTH / 2), rotate_angle, k_scale)
            rm[0, 2] += dx + args.output_w / 2 - IMG_WIDTH / 2
            rm[1, 2] += dy + args.output_h / 2 - IMG_WIDTH / 2

            postprocessed_image = cv2.warpAffine(
                postprocessed_image,
                rm,
                (args.output_w, args.output_h))

            if args.anime4k:
                alpha_channel = postprocessed_image[:, :, 3]
                alpha_channel = cv2.resize(alpha_channel, None, fx=2, fy=2)

                img1 = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGR)
                a.load_image_from_numpy(img1, input_type=ac.AC_INPUT_BGR)
                a.process()
                postprocessed_image = a.save_image_to_numpy()
                postprocessed_image = cv2.merge((postprocessed_image, alpha_channel))
                postprocessed_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_BGRA2RGBA)
            if args.alpha_split:
                alpha_image = cv2.merge(
                    [postprocessed_image[:, :, 3], postprocessed_image[:, :, 3], postprocessed_image[:, :, 3]])
                alpha_image = cv2.cvtColor(alpha_image, cv2.COLOR_RGB2RGBA)
                postprocessed_image = cv2.hconcat([postprocessed_image, alpha_image])

            # 将帧发送到虚拟摄像头
            if cam:
                result_image = postprocessed_image
                if args.output_webcam == 'obs':
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
                cam.send(result_image)
                cam.sleep_until_next_frame()

            # 新增: 将帧发送到web输出队列
            try:
                # 正確且安全的清空佇列的方法
                try:
                    while True:
                        # 不斷地從佇列中取出項目，直到它拋出 Empty 異常
                        self.web_output_q.get_nowait()
                except queue.Empty:
                    # 當接到 Empty 異常時，代表佇列已經被清空。
                    # 這不是一個錯誤，而是我們期望的結果，所以我們用 pass 忽略它。
                    pass

                # --- 佇列清空完畢，後續的程式碼不變 ---
                bgra_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)

                ret, buffer = cv2.imencode('.png', bgra_image)
                if ret:
                    frame_bytes = buffer.tobytes()
                    self.web_output_q.put_nowait(frame_bytes)

            except queue.Full:
                # 如果佇列已滿（極低機率發生，因為我們剛清空），則忽略此幀
                pass
            except Exception as e:
                # 保留這個區塊，以捕捉其他真正未預期的錯誤
                logger.error(f"主进程: 发送帧到Web队列时发生未知错误: repr={repr(e)}", exc_info=True)


class FlaskAPI(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('type', required=True, help="Type field cannot be blank!")
        parser.add_argument('speech_path', default=None)
        parser.add_argument('music_path', default=None)
        parser.add_argument('voice_path', default=None)
        parser.add_argument('mouth_offset', default=0.0)
        parser.add_argument('beat', default=2)
        parser.add_argument('img', default=None)
        parser.add_argument('audio_data', default=None)  # 用于接收音频流数据
        json_args = parser.parse_args()

        logger.info(f"API: 收到请求, 类型: {json_args['type']}")

        try:
            global alive
            if json_args['type'] == "speak":
                if json_args['speech_path']:
                    alive.speak(json_args['speech_path'])
                else:
                    return {"status": "Need speech_path!! 0.0", "receive args": json_args}, 400

            elif json_args['type'] == "speak_from_stream":
                audio_data = json_args['audio_data']
                if audio_data:
                    audio_bytes = None
                    try:
                        # --- 关键修改：处理两种不同的数据格式 ---

                        # 检查是否是客户端发送的字节列表字符串格式，例如 "[82, 73, ...]"
                        if isinstance(audio_data, str) and audio_data.strip().startswith('['):
                            logger.info("API:检测到字节列表字符串格式，正在尝试解析...")
                            # 移除首尾的 '[' 和 ']'
                            cleaned_str = audio_data.strip()[1:-1]
                            # 按逗号分割，并将每个部分转换成整数
                            # 添加 s.strip() 以处理数字周围的空格
                            # 添加 if s.strip() 以处理由于尾部逗号产生的空字符串
                            byte_list = [int(s.strip()) for s in cleaned_str.split(',') if s.strip()]
                            # 从整数列表创建 bytes 对象
                            audio_bytes = bytes(byte_list)
                            logger.info(f"API:成功从字符串列表中解析了 {len(audio_bytes)} 字节。")

                        # 否则，假定是标准的 Base64 字符串
                        elif isinstance(audio_data, str):
                            logger.info("API:检测到Base64字符串格式，正在尝试解码...")
                            audio_data_b64 = audio_data.replace(' ', '+')
                            missing_padding = len(audio_data_b64) % 4
                            if missing_padding != 0:
                                audio_data_b64 += '=' * (4 - missing_padding)
                                logger.warning(f"API: 收到不正确的Base64 padding, 已自动修复。")
                            audio_bytes = base64.b64decode(audio_data_b64)
                            logger.info(f"API:成功从Base64解码了 {len(audio_bytes)} 字节。")

                        else:
                            error_msg = f"Invalid data type for audio_data. Expected string, but got {type(audio_data).__name__}."
                            logger.error(f"API: {error_msg}")
                            return {"status": "Invalid audio_data format", "error": error_msg}, 400

                        # 如果成功获取了 audio_bytes，则调用方法
                        if audio_bytes is not None:
                            alive.speak_from_stream(audio_bytes)
                        else:
                            # 这是一个备用情况，理论上不应该发生
                            raise ValueError("未能从接收的数据中提取音频字节。")

                    except (ValueError, TypeError, base64.binascii.Error) as e:
                        # 捕获所有可能的解析或解码错误
                        logger.error(
                            f"API: 处理音频流时出错。收到的数据 (前100字符): {str(json_args['audio_data'])[:100]}")
                        logger.error(f"API: 解析/解码错误: {e}", exc_info=True)
                        return {"status": "Error processing audio stream",
                                "error": f"Failed to parse or decode audio data: {e}"}, 400
                else:
                    return {"status": "Need audio_data!! 0.0", "receive args": json_args}, 400

            elif json_args['type'] == "rhythm":
                if json_args['music_path']:
                    alive.rhythm(json_args['music_path'], int(json_args['beat']))
                else:
                    return {"status": "Need music_path!! 0.0", "receive args": json_args}, 400
            elif json_args['type'] == "sing":
                if json_args['music_path'] and json_args['voice_path']:
                    alive.sing(json_args['music_path'], json_args['voice_path'], float(json_args['mouth_offset']),
                               int(json_args['beat']))
                else:
                    return {"status": "Need music_path and voice_path!! 0.0", "receive args": json_args}, 400
            elif json_args['type'] == "stop":
                global alive_args
                alive_args["is_speech"].value = False
                alive_args["is_singing"].value = False
                alive_args["is_music_play"].value = False
                logger.info("API: 所有动作已停止")
            elif json_args['type'] == "change_img":
                if json_args['img']:
                    global model_process_args
                    input_image, _ = prepare_input_img(512, json_args['img'])
                    model_process_args['input_image_q'].put_nowait(input_image)
                    logger.info(f"API: 角色图像已更改为 {json_args['img']}")
                else:
                    return {"status": "Need img!! 0.0", "receive args": json_args}, 400
            else:
                logger.warning(f"API: 未知的请求类型 '{json_args['type']}'")
        except Exception as ex:
            # 增加 exc_info=True 来获取完整的堆栈跟踪，便于调试
            logger.error(f"API: 处理请求时发生错误: {ex}", exc_info=True)
            return {'status': 'error', 'message': str(ex)}, 500

        return {'status': "success"}, 200
# 新增: 用于生成视频流的函数
def gen_frames(q):
    while True:
        try:
            # 从队列中获取帧，设置超时以防永久阻塞
            frame = q.get(timeout=1.0)
            # 使用 multipart/x-mixed-replace 格式产出帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            # 如果队列为空，继续等待下一帧
            logger.debug("Web视频流队列为空，等待中...")
            continue
        except Exception as e:
            logger.error(f"生成视频流帧时出错: {e}")
            break


# 新增: Flask 路由
# 根目录，显示视频播放页面
@app.route('/')
def index():
    """Video streaming home page."""
    html_page = """
    <html>
    <head>
        <title>EasyAIV Web Output</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #2c2c2c; color: #f0f0f0; }
            img { border: 2px solid #555; margin-top: 20px; background-color: #000; }
            h1 { color: #00aaff; }
        </style>
    </head>
    <body>
        <h1>EasyAIV 实时视频流</h1>
        <img src="{{ url_for('video_feed') }}" width="512">
    </body>
    </html>
    """
    return render_template_string(html_page)


# 视频流路由
@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global web_output_q
    # 返回一个流式响应
    return Response(gen_frames(web_output_q),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # 使用 logger 替代 print
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        logger.info(f"设备数量: {torch.cuda.device_count()}")
        logger.info(f"设备名称: {torch.cuda.get_device_name(0)}")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"使用的设备: {device}")

    input_image, extra_image = prepare_input_img(512, args.character)

    # 声明跨进程公共参数
    model_process_args = {
        "output_queue": Queue(maxsize=3),
        "input_queue": Queue(),
        "input_image_q": Queue()
    }
    # 初始化动作模块
    model_process = ModelClientProcess(input_image, device, model_process_args)
    model_process.daemon = True
    model_process.start()

    # 声明跨进程公共参数
    alive_args = {
        "is_speech": Value(c_bool, False),
        "speech_q": Queue(),
        "is_singing": Value(c_bool, False),
        "is_music_play": Value(c_bool, False),
        "beat_q": Queue(),
        "mouth_q": Queue(),
    }
    # 初始化模块
    alive = Alive(alive_args)
    alive.start()

    # 新增: 为Web视频流创建队列，maxsize=2 保证低延迟
    web_output_q = Queue(maxsize=2)

    # 初始化主进程, 传入新的队列
    aiv = EasyAIV(model_process_args, alive_args, web_output_q)
    aiv.start()

    # 添加 API 资源路由
    api.add_resource(FlaskAPI, '/alive')

    # 运行 Flask app
    logger.info(f"Flask 服务器正在启动，请在浏览器中打开 http://127.0.0.1:{args.port}/")
    app.run(host='0.0.0.0', port=args.port, threaded=True)  # threaded=True 允许多个客户端同时连接

    logger.info('所有进程已结束。')