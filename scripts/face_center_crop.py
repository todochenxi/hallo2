from insightface.app import FaceAnalysis
from PIL import Image
import cv2
import numpy as np
import torch
import subprocess
import torch.nn.functional as F
from pathlib import Path
import os
# from hallo.utils.util import convert_video_to_images, extract_audio_from_videos

def convert_video_to_images(video_path, output_dir):
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'fps=25',
        str(output_dir / '%04d.png')
    ]

    try:
        print(f"Running command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting video to images: {e}")
        raise

    return output_dir  

def get_fisrst_frame(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image_path = os.path.join(output_dir, video_file.replace('.mp4', '_first_frame.jpg').replace('.avi', '_first_frame.jpg'))
                cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            cap.release()

def get_face(image_path):
    image_name = os.path.basename(image_path)
    source_image = Image.open(image_path)
    ref_image_pil = source_image.convert("RGB")


    faces = face_analysis.get(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))

    # print(faces[0].keys())
    faces_sorted = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
    # print(f"frame_path==>{image_path}, face_lis==>{len(faces_sorted)}")
    face = faces_sorted[0]  # Select the largest face
    # rimg = face_analysis.draw_on(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR), faces)
    # cv2.imwrite(f"./data{image_name}_draw.jpg", rimg)
    # print(face["bbox"])
    return ref_image_pil, face



def face_center_crop(input_dir, output_dir, height=512, width=512):
    first_frame = True
    fist_frame_center_x = 0
    crop_top, crop_bottom, crop_left, crop_right = 0, 0, 0, 0  # 初始化裁剪边界
    path_lis = []
    pil_lis = []
    i = 0
    for frame in sorted(os.listdir(input_dir)):
        # i += 1
        # if i == 3:
        #     break
        frame_path = os.path.join(input_dir, frame)
        ref_image_pil, face = get_face(frame_path)
        
        ref_image_np = np.array(ref_image_pil)
        x1, y1, x2, y2 = face["bbox"]
        face_width = x2 - x1
        face_height = y2 - y1
        center_x = x1 + face_width // 2
        center_y = y1 + face_height // 2

        # 只在第一帧计算裁剪边界
        if first_frame:
            fist_frame_center_x = center_x
            # 计算offset 和 crop_size
            offset = int(face_width * 1.5)
            # 固定的裁剪大小
            crop_size = max(offset * 2, face_height)

            # 计算裁剪区域的初始边界
            crop_top = int(center_y - crop_size // 2)
            crop_bottom = crop_top + crop_size

            # 检查顶部是否超过边界
            if crop_top < 0:
                crop_bottom += -crop_top  # 向下扩展以适应顶部边界
                crop_top = 0  # 顶部固定在0

            # 检查底部是否超过边界
            if crop_bottom > ref_image_np.shape[0]:
                crop_bottom = ref_image_np.shape[0]  # 底部固定在图像高度
                crop_top = crop_bottom - crop_size  # 重新计算crop_top，确保高度固定

            # 左右边界的处理
            crop_left = int(center_x - crop_size // 2)
            crop_right = crop_left + crop_size

            # 检查左右边界
            if crop_left < 0:
                crop_right += -crop_left  # 向右扩展以适应左边界
                crop_left = 0  # 左边固定在0

            if crop_right > ref_image_np.shape[1]:
                crop_right = ref_image_np.shape[1]  # 右边固定在图像宽度
                crop_left = crop_right - crop_size  # 重新计算crop_left，确保宽度固定



            first_frame = False  # 设置为False，后续帧不再重新计算
        print(center_x)
        if center_x < (fist_frame_center_x - 50) or center_x > (fist_frame_center_x + 50):
            break
        # 计算新的裁剪边界，确保人脸不会被裁剪
        face_x1 = face["bbox"][0]
        face_x2 = face["bbox"][2]
        face_y1 = face["bbox"][1]
        face_y2 = face["bbox"][3]

        # 检查边界并适当扩展，如果无法扩展则退出循环
        if face_x1 < crop_left or face_x2 > crop_right or face_y1 < crop_top or face_y2 > crop_bottom:
            print(f"人脸靠近裁剪边界，停止处理帧: {frame}")
            break  # 退出循环
        print(crop_right-crop_left, crop_bottom - crop_top)
        # 根据新的边界裁剪图像
        ref_image_np = ref_image_np[crop_top:crop_bottom, crop_left:crop_right, :]



        # 最后，调整到512x512
        ref_image_pil = Image.fromarray(ref_image_np)
        ref_image_pil = ref_image_pil.resize([width, height], Image.LANCZOS)

        # 保存处理后的图像
        os.makedirs(output_dir, exist_ok=True)
        frame_path = os.path.join(output_dir, frame)
        path_lis.append(frame_path)
        pil_lis.append(ref_image_pil)
    useful_time = len(path_lis) % 25
    # save
    for path, pil in zip(path_lis[:useful_time*25:], pil_lis[:useful_time*25]):
        pil.save(path)
    # remove
    # for path in path_lis[useful_time*25:]:
    #     os.remove(path)

import shutil


def check_crop(input_dir, output_dir, error_output_dir, height=512, width=512, offset_size=1.5):
    fist_frame_center_x = 0
    crop_top, crop_bottom, crop_left, crop_right = 0, 0, 0, 0  # 初始化裁剪边界
    path_lis = []
    pil_lis = []
    i = 0
    for frame in sorted(os.listdir(input_dir)):
        # i += 1
        # if i == 3:
        #     break
        frame_path = os.path.join(input_dir, frame)
        ref_image_pil, face = get_face(frame_path)
        x1, y1, x2, y2 = face["bbox"]
        face_width = x2 - x1
        face_height = y2 - y1
        center_x = x1 + face_width // 2
        center_y = y1 + face_height // 2

        ref_image_np = np.array(ref_image_pil)

        # offset_size = 1.5
        offset = int(face_width * offset_size)
        # 固定的裁剪大小
        crop_size = max(offset * 2, face_height)

        # 计算裁剪区域的初始边界
        crop_top = int(center_y - crop_size // 2)
        crop_bottom = crop_top + crop_size

        # 检查顶部是否超过边界
        if crop_top < 0:
            crop_bottom += -crop_top  # 向下扩展以适应顶部边界
            crop_top = 0  # 顶部固定在0

        # 检查底部是否超过边界
        if crop_bottom > ref_image_np.shape[0]:
            print("crop_bottom>ref_imagenp.shape[0]")
            crop_bottom = ref_image_np.shape[0]  # 底部固定在图像高度
            crop_top = crop_bottom - crop_size  # 重新计算crop_top，确保高度固定

        # 左右边界的处理
        crop_left = int(center_x - crop_size // 2)
        crop_right = crop_left + crop_size

        # 检查左右边界
        if crop_left < 0:
            crop_right += -crop_left  # 向右扩展以适应左边界
            crop_left = 0  # 左边固定在0

        if crop_right > ref_image_np.shape[1]:
            crop_right = ref_image_np.shape[1]  # 右边固定在图像宽度
            crop_left = crop_right - crop_size  # 重新计算crop_left，确保宽度固定
            # 根据新的边界裁剪图像
        
        ref_image_np = ref_image_np[crop_top:crop_bottom, crop_left:crop_right, :]



        # 最后，调整到512x512
        ref_image_pil = Image.fromarray(ref_image_np)
        ref_image_pil = ref_image_pil.resize((width, height), Image.LANCZOS)
        # 保存处理后的图像
        os.makedirs(output_dir, exist_ok=True)
        crop_frame_path = os.path.join(output_dir, frame)
        ref_image_pil.save(crop_frame_path, "JPEG", quality=95)
        
        # check_face
        try:
            _, face = get_face(crop_frame_path)
        except Exception as e:
            print("Error msg", e)
            i += 1
            print("height, width==>",ref_image_np.shape[0], ref_image_np.shape[1])
            print(crop_left, crop_right, crop_top, crop_bottom)
            print(frame)
            os.makedirs(error_output_dir, exist_ok=True)
            new_path = os.path.join(error_output_dir, frame)
            shutil.copy(frame_path, new_path)
            os.remove(crop_frame_path)
            # continue
        # path_lis.append(frame_path)
        # pil_lis.append(ref_image_pil)
    print("error num", i)
    # useful_time = len(path_lis) % 25
    # # save
    # for path, pil in zip(path_lis[:useful_time*25:], pil_lis[:useful_time*25]):
    #     pil.save(path)
    # remove
    # for path in path_lis[useful_time*25:]:
    #     os.remove(path)

def test_resize(image_path, output_path):
    image_pil = Image.open(image_path)
    image_pil = image_pil.resize([256, 256], Image.NEAREST)
    image_pil.save(output_path, "JPEG", quality=50)


def extract_audio_from_videos(video_path: Path, audio_output_path: Path) -> Path:
    """
    Extract audio from a video file and save it as a WAV file.

    This function uses ffmpeg to extract the audio stream from a given video file and saves it as a WAV file
    in the specified output directory.

    Args:
        video_path (Path): The path to the input video file.
        output_dir (Path): The directory where the extracted audio file will be saved.

    Returns:
        Path: The path to the extracted audio file.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails to execute.
    """
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn', '-acodec',
        "pcm_s16le", '-ar', '16000', '-ac', '2',
        str(audio_output_path)
    ]

    try:
        print(f"Running command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from video: {e}")
        raise

    return audio_output_path

def get_error_index(images_dir):
    print("process images::", images_dir)
    for image in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image)
        try:
            _, _ = get_face(image_path)
        except Exception as e:
            print("==>", image_path)
            return True
    return False
if __name__ == "__main__":

    # convert video to image
    # convert_video_to_images(Path("data/woman.mp4"), Path("data/woman"))


    # video_path = Path("/data2/datasets/hallo/hallo_data/30h/videos/0655.mp4")
    # output_dir = Path("/data2/datasets/hallo/hallo_data/30h/images/0655")
    # convert_video_to_images(video_path, output_dir)

    # input_dir = "/data3/ml/hallo/test_data/test_0001"
    # output_dir = "/data3/ml/hallo/test_data/invalids"

    # input_dir = "/data2/datasets/hallo/hallo_data/first_frame"
    # output_dir = "/data2/datasets/hallo/hallo_data/first_frame_offset_1.2"
    # error_output_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset"

    # input_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset"
    # output_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset_1"
    # error_output_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset2"

    # input_dir = "../examples/test"
    # output_dir = "../examples/test_out"
    # error_output_dir = "../examples/first_frame_error_offset3"

    face_analysis = FaceAnalysis(
        name = "",
        root="../pretrained_models/face_analysis",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    file_lis = sorted(os.listdir("/data3/datasets/hallo/hallo_data/30h/images/"))
    segment_size = len(file_lis) //10
    begin = 10
    end = 10
    del_lis = []
    
    for index in file_lis[segment_size*begin:len(file_lis)]:
        flag = get_error_index(f"/data3/datasets/hallo/hallo_data/30h/images/{index}")
        if flag:
            del_lis.append(index)
    print(del_lis)
        
    # check_crop(input_dir, output_dir, error_output_dir, offset_size=1.5)

    # _, face = get_face("data/test_crop/0001.png")
    # area = (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])
    # x1, y1, x2, y2 = face["bbox"][0], face["bbox"][1], face["bbox"][2], face["bbox"][3]
    # face_width = x2 - x1
    # face_height = y2 - y1
    # print(face_width/face_height)
    # print(area/512/512)

    # get_fisrst_frame("/data2/datasets/hallo/hallo_data/videos", "/data2/datasets/hallo/hallo_data/first_frame")
    
    # test_resize("/data3/ml/hallo/hallo/self_learning/hdtf_fist_frame_crop/RD_Radio1_000_first_frame.jpg","RD_Radio1_000_first_frame.jpg")
    # extract_audio_from_videos("../chen.training.MP4", "../examples/driving_audios/chen.wav")
