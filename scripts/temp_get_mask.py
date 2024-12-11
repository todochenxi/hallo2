
file = "/data3/ml/hallo2/examples/reference_images/1.jpg"
import mediapipe as mp
import os
import cv2
import numpy as np
def compute_face_landmarks(detection_result, h, w):
    """
    Compute face landmarks from a detection result.

    Args:
        detection_result (mediapipe.solutions.face_mesh.FaceMesh): The detection result containing face landmarks.
        h (int): The height of the video frame.
        w (int): The width of the video frame.

    Returns:
        face_landmarks_list (list): A list of face landmarks.
    """
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        print("#face is invalid:", len(face_landmarks_list))
        return []
    return [[p.x * w, p.y * h] for p in face_landmarks_list[0]]
def expand_region(region, image_w, image_h, expand_ratio=1.0):
    """
    Expand the given region by a specified ratio.
    Args:
        region (tuple): A tuple containing the coordinates (min_x, max_x, min_y, max_y) of the region.
        image_w (int): The width of the image.
        image_h (int): The height of the image.
        expand_ratio (float, optional): The ratio by which the region should be expanded. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the expanded coordinates (min_x, max_x, min_y, max_y) of the region.
    """

    min_x, max_x, min_y, max_y = region
    mid_x = (max_x + min_x) // 2
    side_len_x = (max_x - min_x) * expand_ratio
    mid_y = (max_y + min_y) // 2
    side_len_y = (max_y - min_y) * expand_ratio
    min_x = mid_x - side_len_x // 2
    max_x = mid_x + side_len_x // 2
    min_y = mid_y - side_len_y // 2
    max_y = mid_y + side_len_y // 2
    if min_x < 0:
        max_x -= min_x
        min_x = 0
    if max_x > image_w:
        min_x -= max_x - image_w
        max_x = image_w
    if min_y < 0:
        max_y -= min_y
        min_y = 0
    if max_y > image_h:
        min_y -= max_y - image_h
        max_y = image_h

    return round(min_x), round(max_x), round(min_y), round(max_y)    
def get_landmark(file):
    """
    This function takes a file as input and returns the facial landmarks detected in the file.

    Args:
        file (str): The path to the file containing the video or image to be processed.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists of floats representing the x and y coordinates of the facial landmarks.
    """
    model_path = "../pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(str(file))
        height, width = image.height, image.width
        face_landmarker_result = landmarker.detect(image)
        face_landmark = compute_face_landmarks(
            face_landmarker_result, height, width)
    # print("============>landmark", face_landmark)
    return np.array(face_landmark), height, width

silhouette_ids = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
def get_face_mask(landmarks, height, width, out_path=None, expand_ratio=1.2):
    """
    Generate a face mask based on the given landmarks.

    Args:
        landmarks (numpy.ndarray): The landmarks of the face.
        height (int): The height of the output face mask image.
        width (int): The width of the output face mask image.
        out_path (pathlib.Path): The path to save the face mask image.
        expand_ratio (float): Expand ratio of mask.
    Returns:
        None. The face mask image is saved at the specified path.
    """
    face_landmarks = np.take(landmarks, silhouette_ids, 0)
    min_xy_face = np.round(np.min(face_landmarks, 0))
    max_xy_face = np.round(np.max(face_landmarks, 0))
    min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1] = expand_region(
        [min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1]], width, height, expand_ratio)
    face_mask = np.zeros((height, width), dtype=np.uint8)
    face_mask[round(min_xy_face[1]):round(max_xy_face[1]),
              round(min_xy_face[0])-100:round(max_xy_face[0]-100)] = 255
    if out_path:
        cv2.imwrite(str(out_path), face_mask)
        return None

    return face_mask
landmarks, height, width = get_landmark(file)
file_name = "1"
face_expand_raio=1.2
cache_dir = "./"
get_face_mask(landmarks, height, width, os.path.join(
        cache_dir, f"{file_name}_face_mask.png"), face_expand_raio)
