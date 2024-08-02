import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

path = 'D:/Code/Python/MediaPipeExample/'
img_path = path + 'image.jpg'

def show_img():

  # 读取图片
  image = cv2.imread(img_path)

  # 检查图片是否正确加载
  if image is None:
      print("Error: Image not found or unable to load.")
  else:
      # 显示图片
      cv2.imshow('Image', image)

      # 等待直到按下任意键
      cv2.waitKey(0)

      # 关闭所有窗口
      cv2.destroyAllWindows()


# run 显示图片测试
# show_img()



task_pose_landmarker_full = path + 'task/pose_landmarker_full.task'
pose_landmarker_heavy = path + 'task/pose_landmarker_heavy.task'
pose_landmarker_lite = path + 'task/pose_landmarker_lite.task'


# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=task_pose_landmarker_full)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(img_path)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# 等待直到按下任意键
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
