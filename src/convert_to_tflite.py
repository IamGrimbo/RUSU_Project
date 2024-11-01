import tensorflow as tf

model = tf.keras.models.load_model('saved_models/fruit_classifier_model.h5')
def representative_data_gen():
    for _ in range(100):
        yield [tf.random.normal([1, *model.input_shape[1:]])]

@tf.function
def my_model(x):
    return model(x)

concrete_func = my_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.experimental_new_converter = True

tflite_model = converter.convert()

with open('saved_models/fruit_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)