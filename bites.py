from flask import Flask, render_template, flash, redirect, request, Response, json
from werkzeug.utils import secure_filename
import os
import secrets
import tensorflow as tf
import numpy as np

UPLOAD_FOLDER = './upload_image/'
ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg', 'bmp', 'gif']
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return render_template('Bite.html')


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('沒有檔案')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_name = os.path.join(app.config['UPLOAD_FOLDER'], f'{secrets.token_hex(10)}_{filename}')
            file.save(save_name)
            result, label = get_prediction(save_name)
            j = {
                'result': result.tolist(),
                'label': label
            }
            return Response(json.dumps(j), mimetype='application/json')


def get_prediction(file_name):

    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255

    model_file = 'model/ckpt.pb'
    label_file = "model/label.txt"

    def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image_file(file_name,
                                    input_height=299,
                                    input_width=299,
                                    input_mean=0,
                                    input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + 'Placeholder'
    output_name = "import/" + 'final_result'
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)
    for i in range(len(results)):
        results[i] *= 100
    labels = load_labels(label_file)
    return results, labels


if __name__ == '__main__':
    app.debug = False
    app.run()
