# TensorFlow Liteの使い方
試した条件は以下の通り。
| 項目 | 条件 |
| --- | --- |
| デバイス | Raspberry Pi 4 |
| Pythonのバージョン | 3.7 |
| TFのバージョン | 1.15.0 |
| 変換元モデル | Keras (model.json + model.hdf5)

## ライブラリのインストール
1. Raspberry Pi 4にTensorFlow 1.15.0を入れる。  
    ```
    wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl
    sudo python3 -m pip install tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl
    ```

## 変換方法
1. まず、model.jsonのinputのshapeを固定する。

1. model.hdf5 + model.json -> model.h5を作成する。  
    ```python
    with open('model.json', 'r') as f:
        json = f.read()
    model = tf.keras.model_from_json(json)
    model.load_weights('model.hdf5')
    model.save('model.h5')
    ```

1. model.h5 -> model.tfliteに変換する。この際に量子化も行う。  
    ```python
    # float16の量子化
    converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    # int8の量子化
    def representative_data_gen():
        images = <shape=NHWC, dtype=np.ndarray>
        images = tf.cast(images, tf.float32) / 255.0
        ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
        for input_value in ds.take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    ```
    [Post-training float16 quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant)  
    [Post-training integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

## 推論方法
set_num_threadsで推論に使うコアの数を指定します。

```python
input_data = 入力データ

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.set_num_threads(4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```
[Converter Python API guide](https://www.tensorflow.org/lite/convert/python_api)