import tensorflow as tf
from tensorflow import keras
model= keras.models.load_model("save_at_43.h5")
image_size = (180, 180)
img = keras.preprocessing.image.load_img(
    "ca.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)