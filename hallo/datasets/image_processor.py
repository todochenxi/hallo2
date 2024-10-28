# pylint: disable=W0718
"""
This module is responsible for processing images, particularly for face-related tasks.
It uses various libraries such as OpenCV, NumPy, and InsightFace to perform tasks like
face detection, augmentation, and mask rendering. The ImageProcessor class encapsulates
the functionality for these operations.
"""
import os
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import transforms

from ..utils.util import (blur_mask, get_landmark_overframes, get_mask,
                          get_union_face_mask, get_union_lip_mask)

MEAN = 0.5
STD = 0.5

class ImageProcessor:
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and handles any exceptions that occurred during the processing.
    """
    def __init__(self, img_size, face_analysis_model_path) -> None:
        self.img_size = img_size

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([MEAN], [STD]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def preprocess(self, source_image_path: str, cache_dir: str, face_region_ratio: float):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        source_image = Image.open(source_image_path)
        ref_image_pil = source_image.convert("RGB")
        # 1. image augmentation
        pixel_values_ref_img = self._augmentation(ref_image_pil, self.pixel_transform)

        # 2.1 detect face
        faces = self.face_analysis.get(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))
        if not faces:
            print("No faces detected in the image. Using the entire image as the face region.")
            # Use the entire image as the face region
            face = {
                "bbox": [0, 0, ref_image_pil.width, ref_image_pil.height],
                "embedding": np.zeros(512)
            }
        else:
            # Sort faces by size and select the largest one
            faces_sorted = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
            face = faces_sorted[0]  # Select the largest face

        # 2.2 face embedding
        face_emb = face["embedding"]

        # 2.3 render face mask
        get_mask(source_image_path, cache_dir, face_region_ratio)
        file_name = os.path.basename(source_image_path).split(".")[0]
        face_mask_pil = Image.open(
            os.path.join(cache_dir, f"{file_name}_face_mask.png")).convert("RGB")

        face_mask = self._augmentation(face_mask_pil, self.cond_transform)

        # 2.4 detect and expand lip, face mask
        sep_background_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_background.png"))
        sep_face_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_face.png"))
        sep_lip_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_lip.png"))

        pixel_values_face_mask = [
            self._augmentation(sep_face_mask, self.attn_transform_64),
            self._augmentation(sep_face_mask, self.attn_transform_32),
            self._augmentation(sep_face_mask, self.attn_transform_16),
            self._augmentation(sep_face_mask, self.attn_transform_8),
        ]
        pixel_values_lip_mask = [
            self._augmentation(sep_lip_mask, self.attn_transform_64),
            self._augmentation(sep_lip_mask, self.attn_transform_32),
            self._augmentation(sep_lip_mask, self.attn_transform_16),
            self._augmentation(sep_lip_mask, self.attn_transform_8),
        ]
        pixel_values_full_mask = [
            self._augmentation(sep_background_mask, self.attn_transform_64),
            self._augmentation(sep_background_mask, self.attn_transform_32),
            self._augmentation(sep_background_mask, self.attn_transform_16),
            self._augmentation(sep_background_mask, self.attn_transform_8),
        ]

        pixel_values_full_mask = [mask.view(1, -1)
                                  for mask in pixel_values_full_mask]
        pixel_values_face_mask = [mask.view(1, -1)
                                  for mask in pixel_values_face_mask]
        pixel_values_lip_mask = [mask.view(1, -1)
                                 for mask in pixel_values_lip_mask]

        return pixel_values_ref_img, face_mask, face_emb, pixel_values_full_mask, pixel_values_face_mask, pixel_values_lip_mask

    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


class ImageProcessorForDataProcessing():
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and handles any exceptions that occurred during the processing.
    """
    def __init__(self, face_analysis_model_path, landmark_model_path, step) -> None:
        self.detector = FaceAnalysis(
                name="",
                root=face_analysis_model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        if step == 2:
            self.face_analysis = self.detector
            self.landmarker = None
        else:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            # Create a face landmarker instance with the video mode:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=landmark_model_path),
                running_mode=VisionRunningMode.IMAGE,
            )
            self.landmarker = FaceLandmarker.create_from_options(options)
            self.face_analysis = None

    def get_face(self, image_path):
        # image_name = os.path.basename(image_path)
        source_image = Image.open(image_path)
        ref_image_pil = source_image.convert("RGB")


        faces = self.detector.get(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))

        # print(faces[0].keys())
        faces_sorted = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
        face = faces_sorted[0]  # Select the largest face
        # rimg = face_analysis.draw_on(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR), faces)
        # cv2.imwrite(f"./data{image_name}_draw.jpg", rimg)
        print(face["bbox"])
        return ref_image_pil, face
    
    def face_center_crop(self, input_dir, output_dir, height=512, width=512):
        first_frame = True
        fist_frame_center_x = 0
        crop_top, crop_bottom, crop_left, crop_right = 0, 0, 0, 0  # 初始化裁剪边界
        path_lis = []
        pil_lis = []
        offset_size = 1 if os.path.basename(input_dir) in ['0454', '0455', '0456', '0457', '0458', '0459', '0464', '0465', '0536', '1541', '1543', '1544', '1589', '1590', '1591', '2538', '2691', '2712', '2754', '2777', '2942', '3217', '3219', '3220', '3221', '3222', '3224', '3225', '3226', '3227', '3229', '3230', '3231', '3232', '3235', '3236', '3237', '3238', '3240', '3755', '3954', '3995', '4074', '4110', '4128', '4129', '4130'] else 1.2

        # i = 0
        for frame in sorted(os.listdir(input_dir)):
            # i += 1
            # if i == 27:
            #     break
            frame_path = os.path.join(input_dir, frame)
            ref_image_pil, face = self.get_face(frame_path)
            
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
        useful_time = len(path_lis) // 25
        # save
        for path, pil in zip(path_lis[:useful_time*25:], pil_lis[:useful_time*25]):
            pil.save(path)
        # remove
        all_frames = [os.path.join(input_dir, frame) for frame in sorted(os.listdir(input_dir))]
        for path in all_frames[useful_time*25:]:
            os.remove(path)

    def preprocess(self, source_image_path: str):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        # 1. get face embdeding
        face_mask, face_emb, sep_pose_mask, sep_face_mask, sep_lip_mask = None, None, None, None, None
        if self.face_analysis:
            for frame in sorted(os.listdir(source_image_path)):
                try:
                    source_image_path = os.path.join(source_image_path, frame)
                    _, face = self.get_face(source_image_path)
                    # ref_image_pil = source_image.convert("RGB")
                    # # 2.1 detect face
                    # faces = self.face_analysis.get(cv2.cvtColor(
                    #     np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))
                    # # use max size face
                    # face = sorted(faces, key=lambda x: (
                    #     x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
                    # 2.2 face embedding
                    face_emb = face["embedding"]
                    if face_emb is not None:
                        break
                except Exception as _:
                    continue

        if self.landmarker:
            # face_center_crop
            self.face_center_crop(source_image_path, source_image_path)

            # 3.1 get landmark
            landmarks, height, width = get_landmark_overframes(
                self.landmarker, source_image_path)
            assert len(landmarks) == len(os.listdir(source_image_path))

            # 3 render face and lip mask
            face_mask = get_union_face_mask(landmarks, height, width)
            lip_mask = get_union_lip_mask(landmarks, height, width)

            # 4 gaussian blur
            blur_face_mask = blur_mask(face_mask, (64, 64), (51, 51))
            blur_lip_mask = blur_mask(lip_mask, (64, 64), (31, 31))

            # 5 seperate mask
            sep_face_mask = cv2.subtract(blur_face_mask, blur_lip_mask)
            sep_pose_mask = 255.0 - blur_face_mask
            sep_lip_mask = blur_lip_mask

        return face_mask, face_emb, sep_pose_mask, sep_face_mask, sep_lip_mask

    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
