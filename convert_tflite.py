import tensorflow as tf
import os
import gc
from tensorflow import keras
from stable_diffusion_tf.diffusion_model import UNetModel

MAX_TEXT_LEN = 77
 
h5_model_file = "../stable-diffusion/diffusion_model.h5"
tf_model_file = "../stable-diffusion/diffusion_model.tflite"

img_height = 512
img_width = 512
n_h = img_height // 8
n_w = img_width // 8
#########################
context = keras.layers.Input((MAX_TEXT_LEN, 768))
t_emb = keras.layers.Input((320,))
latent = keras.layers.Input((n_h, n_w, 4))
unet = UNetModel()
diffusion_model = keras.models.Model(
    [latent, t_emb, context], unet([latent, t_emb, context])
)
#########################
print("Loading weights...")
diffusion_model.load_weights(os.path.abspath(h5_model_file))
#########################
print("Saving hdf5 model...")
diffusion_model.save(
    os.path.abspath("../stable-diffusion/diffusion_model.hdf5")
)
#########################
print("Loading model from disk...")
model = tf.keras.models.load_model(
    os.path.abspath("../stable-diffusion/diffusion_model.hdf5"),
    custom_objects={"UNetModel": UNetModel },
    compile=False
)
gc.collect()
print("Converting model to tflite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(diffusion_model)
# converter.allow_custom_ops=True
#converter.post_training_quantize=True 
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
converter.target_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

print("Saving tflite model...")
open(os.path.abspath(tf_model_file), "wb").write(tflite_model)

print("Done.")
