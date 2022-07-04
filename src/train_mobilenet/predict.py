from keras_segmentation.predict import predict_multiple
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#       tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#       tf.config.experimental.set_virtual_device_configuration( gpus[0],
#                                                               [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
#       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU \n")
#     except RuntimeError as e:
#       # Visible devices must be set before GPUs have been initialized
#       print(e)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
gpu = tf.test.gpu_device_name()

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())




predict_multiple( 
checkpoints_path="mobilenet_checkpoints", 
inp_dir="./zed_recording_09_02_2022_session_1",
out_dir="./outputs/zed_recording_09_02_2022_session_1/" 
)

predict_multiple( 
checkpoints_path="mobilenet_segnet_no_aug", 
inp_dir="./zed_recording_09_02_2022_session_1",
out_dir="./outputs/zed_recording_09_02_2022_session_1_no_aug/" 
)
