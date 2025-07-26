from tensorflow.keras.models import load_model
import tensorflow as tf

def load_emotion_model(weights_path):
    base_model = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.load_weights(weights_path)
    return model
