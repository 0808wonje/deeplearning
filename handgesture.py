# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# IMAGE_FILENAMES = ['images/thumbs_down.jpg', 'images/victory.jpg', 'images/thumbs_up.jpg', 'images/pointing_up.jpg']

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2.imshow('test', img)
#   cv2.waitKey(0)



# # Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # STEP 2: Create an GestureRecognizer object.
# base_options = python.BaseOptions(model_asset_path='models\gesture_recognizer.task')
# options = vision.GestureRecognizerOptions(base_options=base_options)
# recognizer = vision.GestureRecognizer.create_from_options(options)


# images = []
# results = []
# for image_file_name in IMAGE_FILENAMES:
#   # STEP 3: Load the input image.
#   image = mp.Image.create_from_file(image_file_name)

#   # STEP 4: Recognize gestures in the input image.
#   recognition_result = recognizer.recognize(image)

#   # STEP 5: Process the result. In this case, visualize it.
#   images.append(image)
#   top_gesture = recognition_result.gestures[0][0]
#   hand_landmarks = recognition_result.hand_landmarks
#   results.append((top_gesture, hand_landmarks))


# print(results)


# # display_batch_of_images_with_gestures_and_hand_landmarks(images, results)


####################################


import urllib.request

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
  urllib.request.urlretrieve(url, name)



import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow('test', img)
  cv2.waitKey(0)


# Preview the images.
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
for image_file_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_file_name)

  # STEP 4: Recognize gestures in the input image.
  recognition_result = recognizer.recognize(image)

  # STEP 5: Process the result. In this case, visualize it.
  images.append(image)
  top_gesture = recognition_result.gestures[0][0]
  hand_landmarks = recognition_result.hand_landmarks
  results.append((top_gesture, hand_landmarks))

  for idx, e in enumerate(results):
      print(f'{idx} = {e}\n')


# display_batch_of_images_with_gestures_and_hand_landmarks(images, results)