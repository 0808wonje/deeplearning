import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # redpyr

IMAGE_FILE = 'images\cat_and_dog.jpg'

# visualize는 무시, step1 ~ step5 만 기억하면 됌
def visualize(image, detection_result) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


import cv2

img = cv2.imread('images\cat_and_dog.jpg')
# cv2.imshow('test', img)
# cv2.waitKey(0)



# STEP 1: Import the necessary modules.
# 필요한 모듈 임포트
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 모든 태스크에서 가장 기본은 모델을 가져오는 것, 추론기를 만드는 단계
# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite') # 모델 경로 설정
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)  
detector = vision.ObjectDetector.create_from_options(options) # ObjectDetector

# STEP 3: Load the input image.
# 대상이 될 이미지 가져오기
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.
# 이미지에서 객체를 찾아낸다(detect)
detection_result = detector.detect(image)

print('detection_result = ', detection_result)

# STEP 5: Process the detection result. In this case, visualize it.
# 후처리 작업 (가변적)
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
print('img = ', img)
print()
print('rgb_annotated_image = ', rgb_annotated_image)
cv2.imshow('test', rgb_annotated_image)
cv2.waitKey(0)
