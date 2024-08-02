from matplotlib import pyplot as plt
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests


def show_image(img, figsize=(10, 10)):
  """Shows output PIL image."""
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.show()


path = 'D:/Code/Python/MediaPipeExample/pose_csv/'
# image_path = path + 'image.jpg'
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# show_image(img, figsize=(20, 20))




class FullBodyPoseEmbedder(object):
  """将 3D 姿势标志点转换为 3D 嵌入。"""

  def __init__(self, torso_size_multiplier=2.5):
    # 乘数应用于躯干以获得最小身体大小。
    self._torso_size_multiplier = torso_size_multiplier

    # 标志点名称，如预测中出现的名称。
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """标准化姿势标志点并转换为嵌入。
    
    参数:
      landmarks - 形状为 (N, 3) 的 3D 标志点的 NumPy 数组。

    结果:
      形状为 (M, 3) 的姿势嵌入的 Numpy 数组，其中 `M` 是 `_get_pose_distance_embedding` 中定义的成对距离的数量。
    """
    assert landmarks.shape[0] == len(self._landmark_names), '标志点数量不符合预期: {}'.format(landmarks.shape[0])

    # 获取姿势标志点。
    landmarks = np.copy(landmarks)

    # 标准化标志点。
    landmarks = self._normalize_pose_landmarks(landmarks)

    # 获取嵌入。
    embedding = self._get_pose_distance_embedding(landmarks)

    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """标准化标志点的平移和缩放。"""
    landmarks = np.copy(landmarks)

    # 标准化平移。
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # 标准化缩放。
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # 乘以 100 不是必需的，但可以使调试更容易。
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """计算姿势中心作为臀部之间的点。"""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """计算姿势大小。
    
    它是两个值中的最大值：
      * 躯干大小乘以 `torso_size_multiplier`
      * 从姿势中心到任何姿势标志点的最大距离
    """
    # 这种方法仅使用 2D 标志点计算姿势大小。
    landmarks = landmarks[:, :2]

    # 臀部中心。
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # 肩膀中心。
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # 躯干大小作为最小身体大小。
    torso_size = np.linalg.norm(shoulders - hips)

    # 到姿势中心的最大距离。
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
    """将姿势标志点转换为 3D 嵌入。

    我们使用多个成对的 3D 距离来形成姿势嵌入。所有距离都包含带符号的 X 和 Y 分量。
    我们使用不同类型的配对来覆盖不同的姿势类别。可以根据需要移除一些或添加新的配对。
    
    参数:
      landmarks - 形状为 (N, 3) 的 3D 标志点的 NumPy 数组。

    结果:
      形状为 (M, 3) 的姿势嵌入的 Numpy 数组，其中 `M` 是成对距离的数量。
    """
    embedding = np.array([
        # 一个关节。
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

        # 两个关节。

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        # 四个关节。

        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # 五个关节。

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),
        
        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # 交叉身体。

        self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        # 身体弯曲方向。

        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
        #     landmarks[self._landmark_names.index('left_hip')
        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
        #     landmarks[self._landmark_names.index('right_hip')]),
    ])

    return embedding
    
  def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
  def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from
  

class PoseSample(object):

  def __init__(self, name, landmarks, class_name, embedding):
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name
    
    self.embedding = embedding


class PoseSampleOutlier(object):

  def __init__(self, sample, detected_class, all_classes):
    self.sample = sample
    self.detected_class = detected_class
    self.all_classes = all_classes


class PoseClassifier(object):
  """分类姿势标志点。"""

  def __init__(self,
               pose_samples_folder,
               pose_embedder,
               file_extension='csv',
               file_separator=',',
               n_landmarks=33,
               n_dimensions=3,
               top_n_by_max_distance=30,
               top_n_by_mean_distance=10,
               axes_weights=(1., 1., 0.2)):
    self._pose_embedder = pose_embedder
    self._n_landmarks = n_landmarks
    self._n_dimensions = n_dimensions
    self._top_n_by_max_distance = top_n_by_max_distance
    self._top_n_by_mean_distance = top_n_by_mean_distance
    self._axes_weights = axes_weights

    self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                 file_extension,
                                                 file_separator,
                                                 n_landmarks,
                                                 n_dimensions,
                                                 pose_embedder)

  def _load_pose_samples(self,
                         pose_samples_folder,
                         file_extension,
                         file_separator,
                         n_landmarks,
                         n_dimensions,
                         pose_embedder):
    """从给定文件夹加载姿势样本。
    
    所需的文件夹结构:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    所需的 CSV 结构:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
    """
    # 文件夹中的每个文件表示一个姿势类别。
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    pose_samples = []
    for file_name in file_names:
      # 使用文件名作为姿势类别名。
      class_name = file_name[:-(len(file_extension) + 1)]
      
      # 解析 CSV。
      with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
          assert len(row) == n_landmarks * n_dimensions + 1, '值的数量错误: {}'.format(len(row))
          landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
          pose_samples.append(PoseSample(
              name=row[0],
              landmarks=landmarks,
              class_name=class_name,
              embedding=pose_embedder(landmarks),
          ))

    return pose_samples

  def find_pose_sample_outliers(self):
    """针对整个数据库对每个样本进行分类。"""
    # 找到目标姿势中的异常值
    outliers = []
    for sample in self._pose_samples:
      # 找到目标姿势的最近姿势。
      pose_landmarks = sample.landmarks.copy()
      pose_classification = self.__call__(pose_landmarks)
      class_names = [class_name for class_name, count in pose_classification.items() if count == max(pose_classification.values())]

      # 如果最近的姿势属于不同的类别或者检测到的最近姿势类别超过一个，样本就是异常值。
      if sample.class_name not in class_names or len(class_names) != 1:
        outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

    return outliers

  def __call__(self, pose_landmarks):
    """对给定的姿势进行分类。

    分类分两个阶段进行：
      * 首先我们通过 MAX 距离选择前 N 个样本。这允许删除与给定姿势几乎相同的样本，但有几个关节向另一个方向弯曲。
      * 然后我们通过 MEAN 距离选择前 N 个样本。在上一阶段删除异常值后，我们可以选择平均上最接近的样本。
    
    参数:
      pose_landmarks: 形状为 (N, 3) 的 3D 标志点的 NumPy 数组。

    返回:
      字典，包含数据库中最近的姿势样本的数量。样例：
        {
          'pushups_down': 8,
          'pushups_up': 2,
        }
    """
    # 检查提供的姿势和目标姿势具有相同的形状。
    assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), '形状不符合预期: {}'.format(pose_landmarks.shape)

    # 获取给定姿势嵌入。
    pose_embedding = self._pose_embedder(pose_landmarks)
    flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

    # 通过最大距离筛选。
    #
    # 这有助于删除异常值 - 这些姿势与给定姿势几乎相同，但一个关节弯曲到另一个方向，实际上代表不同的姿势类别。
    max_dist_heap = []
    for sample_idx, sample in enumerate(self._pose_samples):
      max_dist = min(
          np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      max_dist_heap.append([max_dist, sample_idx])

    max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
    max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

    # 通过平均距离筛选。
    #
    # 删除异常值后，我们可以通过平均距离找到最近的姿势。
    mean_dist_heap = []
    for _, sample_idx in max_dist_heap:
      sample = self._pose_samples[sample_idx]
      mean_dist = min(
          np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      mean_dist_heap.append([mean_dist, sample_idx])

    mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
    mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

    # 将结果收集到映射中：(class_name -> n_samples)
    class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
    result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

    return result

class EMADictSmoothing(object):
  """平滑姿势分类。"""

  def __init__(self, window_size=10, alpha=0.2):
    self._window_size = window_size
    self._alpha = alpha

    self._data_in_window = []

  def __call__(self, data):
    """平滑给定的姿势分类。

    通过在给定时间窗口内对每个观察到的姿势类别计算指数移动平均值来完成平滑处理。
    缺失的姿势类别用 0 替代。
    
    参数:
      data: 包含姿势分类的字典。样例：
          {
            'pushups_down': 8,
            'pushups_up': 2,
          }

    返回:
      与输入格式相同的字典，但值为平滑后的浮点数而不是整数。样例：
        {
          'pushups_down': 8.3,
          'pushups_up': 1.7,
        }
    """
    # 将新数据添加到窗口的开头，以简化代码。
    self._data_in_window.insert(0, data)
    self._data_in_window = self._data_in_window[:self._window_size]

    # 获取所有键。
    keys = set([key for data in self._data_in_window for key, _ in data.items()])

    # 获取平滑后的值。
    smoothed_data = dict()
    for key in keys:
      factor = 1.0
      top_sum = 0.0
      bottom_sum = 0.0
      for data in self._data_in_window:
        value = data[key] if key in data else 0.0

        top_sum += factor * value
        bottom_sum += factor

        # 更新因子。
        factor *= (1.0 - self._alpha)

      smoothed_data[key] = top_sum / bottom_sum

    return smoothed_data


class RepetitionCounter(object):
  """计算给定目标姿势类别的重复次数。"""

  def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
    self._class_name = class_name

    # 如果姿势计数超过给定阈值，则进入该姿势。
    self._enter_threshold = enter_threshold
    self._exit_threshold = exit_threshold

    # 标记我们是否进入了给定姿势。
    self._pose_entered = False

    # 退出姿势的次数。
    self._n_repeats = 0

  @property
  def n_repeats(self):
    return self._n_repeats

  def __call__(self, pose_classification):
    """计算直到给定帧的重复次数。

    我们使用两个阈值。首先需要超过较高的阈值进入姿势，然后需要低于较低的阈值退出它。
    阈值之间的差异使其对预测抖动稳定（在只有一个阈值的情况下会导致错误计数）。
    
    参数:
      pose_classification: 当前帧上的姿势分类字典。
        示例：
          {
            'pushups_down': 8.3,
            'pushups_up': 1.7,
          }

    返回:
      重复次数的整数计数。
    """
    # 获取姿势置信度。
    pose_confidence = 0.0
    if self._class_name in pose_classification:
      pose_confidence = pose_classification[self._class_name]

    # 在第一帧或我们不在姿势中的情况下，只需检查我们是否在这一帧进入姿势并更新状态。
    if not self._pose_entered:
      self._pose_entered = pose_confidence > self._enter_threshold
      return self._n_repeats

    # 如果我们在姿势中并且正在退出它，则增加计数器并更新状态。
    if pose_confidence < self._exit_threshold:
      self._n_repeats += 1
      self._pose_entered = False

    return self._n_repeats




class PoseClassificationVisualizer(object):
  """跟踪每一帧的分类并渲染它们。"""

  def __init__(self,
               class_name,
               plot_location_x=0.05,
               plot_location_y=0.05,
               plot_max_width=0.4,
               plot_max_height=0.4,
               plot_figsize=(9, 4),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.85,
               counter_location_y=0.05,
               counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
               counter_font_color='red',
               counter_font_size=0.15):
    self._class_name = class_name
    self._plot_location_x = plot_location_x
    self._plot_location_y = plot_location_y
    self._plot_max_width = plot_max_width
    self._plot_max_height = plot_max_height
    self._plot_figsize = plot_figsize
    self._plot_x_max = plot_x_max
    self._plot_y_max = plot_y_max
    self._counter_location_x = counter_location_x
    self._counter_location_y = counter_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count):
    """渲染姿势分类和计数器直到给定帧。"""
    # 扩展分类历史。
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    # 输出带有分类图和计数器的帧。
    output_img = Image.fromarray(frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # 绘制图表。
    img = self._plot_classification_history(output_width, output_height)
    img.thumbnail((int(output_width * self._plot_max_width),
                   int(output_height * self._plot_max_height)),
                  Image.ANTIALIAS)
    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))

    # 绘制计数。
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_height * self._counter_font_size)
      font_request = requests.get(self._counter_font_path, allow_redirects=True)
      self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)

    return output_img

  def _plot_classification_history(self, output_width, output_height):
    fig = plt.figure(figsize=self._plot_figsize)

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
      plt.plot(y, linewidth=7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Classification history for `{}`'.format(self._class_name))
    plt.legend(loc='upper right')

    if self._plot_y_max is not None:
      plt.ylim(top=self._plot_y_max)
    if self._plot_x_max is not None:
      plt.xlim(right=self._plot_x_max)

    # 将图表转换为图像。
    buf = io.BytesIO()
    dpi = min(
        output_width * self._plot_max_width / float(self._plot_figsize[0]),
        output_height * self._plot_max_height / float(self._plot_figsize[1]))
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img






# Bootstrap 助手
class BootstrapHelper(object):
    """帮助初始化图像并过滤用于分类的姿势样本。"""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # 获取姿势类别列表并打印图像统计信息。
        self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

    def bootstrap(self, per_pose_class_limit=None):
        """初始化给定文件夹中的图像。

        所需的输入图像文件夹（输出图像文件夹的使用方式相同）：
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        生成的 CSV 文件夹：
          pushups_up.csv
          pushups_down.csv

        生成的 CSV 结构包含姿势 3D 地标：
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # 创建输出 CSV 文件夹。
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print('初始化 ', pose_class_name, file=sys.stderr)

            # 姿势类别的路径。
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                # 获取图像列表。
                image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                # 初始化每个图像。
                for image_name in tqdm.tqdm(image_names):
                    # 加载图像。
                    input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    # 初始化新的姿势跟踪器并运行它。
                    with mp_pose.Pose(upper_body_only=False) as pose_tracker:
                        result = pose_tracker.process(image=input_frame)
                        pose_landmarks = result.pose_landmarks

                    # 保存带有姿势预测的图像（如果检测到姿势）。
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # 如果检测到姿势，则保存地标。
                    if pose_landmarks is not None:
                        # 获取地标。
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                             for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                        csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str).tolist())

                    # 绘制 XZ 投影并与图像拼接。
                    projection_xz = self._draw_xz_projection(output_frame=output_frame, pose_landmarks=pose_landmarks)
                    output_frame = np.concatenate((output_frame, projection_xz), axis=1)

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None:
            return np.asarray(img)

        # 根据图像宽度缩放半径。
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # 翻转 Z 并将髋关节中心移动到图像中心。
            x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
            x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """确保图像文件夹和 CSV 文件夹具有相同的样本。

        只保留图像文件夹和 CSV 文件夹中的交集样本。
        """
        for pose_class_name in self._pose_class_names:
            # 姿势类别的路径。
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

            # 将 CSV 读取到内存中。
            rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    rows.append(row)

            # CSV 中剩余的图像名称。
            image_names_in_csv = []

            # 重写 CSV，移除没有对应图像的行。
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print('从 CSV 中移除图像：', image_path)

            # 移除没有对应行的图像。
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print('从文件夹中移除图像：', image_path)

    def analyze_outliers(self, outliers):
        """将每个样本与所有其他样本进行分类以找出异常值。

        如果样本的分类与原始类别不同 - 它应该被删除或添加更多相似的样本。
        """
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('异常值')
            print('  样本路径 =    ', image_path)
            print('  样本类别 =   ', outlier.sample.class_name)
            print('  检测类别 = ', outlier.detected_class)
            print('  所有类别 =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """从图像文件夹中移除异常值。"""
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """打印输入图像文件夹的统计信息。"""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """打印输出图像文件夹的统计信息。"""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('每个姿势类别的图像数量：')
        for pose_class_name in pose_class_names:
            n_images = len([n for n in os.listdir(os.path.join(images_folder, pose_class_name)) if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))




# 确定输入和输出文件夹路径
bootstrap_images_in_folder = path + 'fitness_poses_images_in'
bootstrap_images_out_folder =  path + 'fitness_poses_images_out'
bootstrap_csvs_out_folder =  path + 'fitness_poses_csvs_out'

# 创建 BootstrapHelper 实例
bootstrap_helper = BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=bootstrap_csvs_out_folder)

# 打印输入图像文件夹的统计信息
bootstrap_helper.print_images_in_statistics()

# 初始化图像并生成 CSV 文件
bootstrap_helper.bootstrap(per_pose_class_limit=None)

# 对齐图像和 CSV 文件
bootstrap_helper.align_images_and_csvs(print_removed_items=True)

# 打印输出图像文件夹的统计信息
bootstrap_helper.print_images_out_statistics()




# 手动过滤
# 请手动验证预测并删除具有错误姿势预测的样本（图像）。检查是否要求您仅根据预测的地标对姿势进行分类。如果不能 - 将其删除。

# 完成后对齐 CSV 和图像文件夹。

# Align CSVs with filtered images.
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()

# 查找异常值。

# 将姿态标志转换为嵌入向量。
pose_embedder = FullBodyPoseEmbedder()

# 将给定的姿态与姿态数据库进行分类。
pose_classifier = PoseClassifier(
    pose_samples_folder=bootstrap_csvs_out_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

outliers = pose_classifier.find_pose_sample_outliers()
print('异常值数量: ', len(outliers))


# 分析异常值。
# bootstrap_helper.analyze_outliers(outliers)


# 移除所有异常值（如果你不想手动挑选的话）。
# bootstrap_helper.remove_outliers(outliers)


# 移除异常值后，与图像对齐 CSV 文件。
# bootstrap_helper.align_images_and_csvs(print_removed_items=False)
# bootstrap_helper.print_images_out_statistics()


# def dump_for_the_app():
#   pose_samples_folder = 'fitness_poses_csvs_out'
#   pose_samples_csv_path = 'fitness_poses_csvs_out.csv'
#   file_extension = 'csv'
#   file_separator = ','

#   # 文件夹中的每个文件表示一个姿态类别。
#   file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

#   with open(pose_samples_csv_path, 'w') as csv_out:
#     csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
#     for file_name in file_names:
#       # 使用文件名作为姿态类别名称。
#       class_name = file_name[:-(len(file_extension) + 1)]

#       # 一个文件行: `sample_00001,x1,y1,x2,y2,....`。
#       with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
#         csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
#         for row in csv_in_reader:
#           row.insert(1, class_name)
#           csv_out_writer.writerow(row)

#   files.download(pose_samples_csv_path)

def dump_for_the_app(output_path):
    pose_samples_folder = path + 'fitness_poses_csvs_out'
    pose_samples_csv_path = os.path.join(output_path, 'fitness_poses_csvs_out.csv')
    file_extension = 'csv'
    file_separator = ','

    # 文件夹中的每个文件表示一个姿态类别。
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    with open(pose_samples_csv_path, 'w') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
            # 使用文件名作为姿态类别名称。
            class_name = file_name[:-(len(file_extension) + 1)]

            # 一个文件行: `sample_00001,x1,y1,x2,y2,....`。
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)

    print(f'文件已保存到: {pose_samples_csv_path}')


dump_for_the_app(path)