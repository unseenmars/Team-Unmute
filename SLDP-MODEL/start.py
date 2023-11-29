# Define paths for various components needed for the model
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
MODEL_PATH = WORKSPACE_PATH + '/models'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/'

labels = [
    {'name':'Hello', 'id':1},
    {'name':'Friend', 'id':2},
    {'name':'Yes', 'id':3},
    {'name':'No', 'id':4},
    {'name':'Thank You', 'id':5},
    {'name':'Ok', 'id':6},
    {'name':'Bathroom', 'id':7},
    {'name':'Please', 'id':8},
    {'name':'1000 Years of Death', 'id':9},
]

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' # name of custom model

# Import necessary libraries
import tensorflow as tf
from object_detection.utils import config_util, label_map_util, visualization_utils as vz_utils
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from google.protobuf import text_format
import cv2
import numpy as np
import threading
import queue
import pyttsx3

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()


# Define detection function
@tf.function
def detect_fn(image):
    """Function to detect objects in an image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# Initialize pyttsx3 engine for text-to-speech
engine = pyttsx3.init()
speak_queue = queue.Queue()


def speak_worker():
    """Worker thread function to handle text-to-speech tasks."""
    while True:
        text_to_speak = speak_queue.get()
        if text_to_speak is None:  # None is used to signal the end of the program
            break
        engine.say(text_to_speak)  # Command the engine to say the text
        engine.runAndWait()  # Wait for the speech to finish
        speak_queue.task_done()  # Indicate that the task is completed


# Start the speak worker thread
speak_thread = threading.Thread(target=speak_worker)
speak_thread.start()

# Setting up video capture
cap = cv2.VideoCapture(0)  # '0' for the default camera
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

print('\n\n=============================== BOOT-UP =================================\n')

# Main detection loop
while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    vz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)

    # Text-to-Speech for detected labels with high confidence
    for i in range(num_detections):
        if detections['detection_scores'][i] > 0.85:  # Only announce high confidence detections
            class_id = detections['detection_classes'][i] + label_id_offset
            class_name = category_index[class_id]['name']
            speak_queue.put(class_name)  # Add the class name to the speech queue

    cv2.imshow('Object Detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):