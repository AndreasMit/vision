#!/usr/bin/env python3

from keras_segmentation.models.segnet import mobilenet_segnet
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
gpu = tf.test.gpu_device_name()

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())


model = mobilenet_segnet(n_classes=2 ,  input_height=224, input_width=224  )


model.summary()

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

model.train(
    train_images =  "ResizedFr/",
    train_annotations = "CMasks/",
    checkpoints_path = "mobilenet_checkpoints/mobilenets" , epochs=50,
    batch_size=10,
    verify_dataset=True,
    validate=True,
    val_images="TestFr",
    val_annotations="TestM",
    val_batch_size=10,
    auto_resume_checkpoint=True,
    load_weights=None,
    steps_per_epoch=512,
    val_steps_per_epoch=512,
    gen_use_multiprocessing=False,
    ignore_zero_class=False,
    optimizer_name='adadelta',
    do_augment=False,
    augmentation_name="aug_all",
)
