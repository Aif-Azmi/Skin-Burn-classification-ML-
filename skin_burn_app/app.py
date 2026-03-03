import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import io
import cv2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras._tf_keras.keras.applications.vgg16 import VGG16
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
from flask import session



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['HEATMAP_FOLDER'] = 'heatmaps'
app.secret_key = 'your_secret_key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
model = tf.keras.models.load_model('trained_model/latestbestmodel.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# === LIME Related ===
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)], predictions

def generate_lime_heatmap(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array, model.predict, top_labels=3, hide_color=0, num_samples=1000)
    predicted_class, _ = predict_image_class(image_path)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(f"LIME Explanation for Class: {predicted_class}")
    ax.axis('off')
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png')
    img_io.seek(0)
    return img_io

# === Grad-CAM Related ===
def residual_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def generate_gradcam_heatmap(img_path, model, last_conv_layer_name, pred_index=None):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image_path, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    return superimposed_img

def build_cam_models():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_output = GlobalAveragePooling2D()(vgg.output)
    vgg_output = Dense(256, activation='relu')(vgg_output)
    vgg_output = Dense(3, activation='softmax')(vgg_output)
    vgg_model = Model(vgg.input, vgg_output)

    inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    inception_output = GlobalAveragePooling2D()(inception.output)
    inception_output = Dense(256, activation='relu')(inception_output)
    inception_output = Dense(3, activation='softmax')(inception_output)
    inception_model = Model(inception.input, inception_output)

    res_input = Input(shape=(224, 224, 3))
    x = Conv2D(64, (7, 7), strides=2, padding='same')(res_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    res_model = Model(res_input, x)
    return vgg_model, inception_model, res_model

def generate_gradcams(img_path):
    vgg_model, inception_model, res_model = build_cam_models()

    heatmaps = {}
    for model_name, model_obj, layer_name in [
        ('vgg', vgg_model, 'block5_conv3'),
        ('inception', inception_model, 'mixed10'),
        ('residual', res_model, 'conv2d_102'),
    ]:
        heatmap = generate_gradcam_heatmap(img_path, model_obj, layer_name)
        result_img = overlay_heatmap(heatmap, img_path)
        save_path = os.path.join(app.config['HEATMAP_FOLDER'], f'{model_name}_gradcam.jpg')
        cv2.imwrite(save_path, result_img)
        print(f"Saved {model_name} Grad-CAM at: {save_path}")
        heatmaps[model_name] = save_path  # Save correct path
    return heatmaps

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    image_path = None
    prediction = None
    image_filename = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('classify.html')
        file = request.files['image']
        if file.filename == '':
            return render_template('classify.html')    
        
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            image_path = upload_path
            prediction, _ = predict_image_class(upload_path)
            image_filename = filename
            return render_template('classify.html', image_path=image_path, prediction=prediction, image_filename=image_filename)
        # Store in session
    session['latest_image'] = image_path
    session['predicted_class'] = prediction
    return render_template('classify.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap_route():
    image_path = session.get('latest_image')
    predicted_class = session.get('predicted_class')
    image_path = request.form['image_path']

    # Generate LIME heatmap
    lime_io = generate_lime_heatmap(image_path)
    lime_filename = f"lime_{os.path.basename(image_path)}.png"
    lime_filepath = os.path.join(app.config['HEATMAP_FOLDER'], lime_filename)
    with open(lime_filepath, 'wb') as f:
        f.write(lime_io.getvalue())

    
    # Generate Grad-CAM images
    gradcam_paths = generate_gradcams(image_path)

    # Convert to URL-safe relative filenames
    gradcam_files = {
    model_name: os.path.basename(path).replace("\\", "/")
    for model_name, path in gradcam_paths.items()
    }
    session['lime_heatmap'] = lime_filename 
    return render_template('heatmap.html',
                       lime_heatmap_filename=lime_filename,
                       gradcam_paths=gradcam_files)

@app.route('/generate_gradcam', methods=['POST'])
def generate_gradcam_route():
    image_path = request.form['image_path']
    heatmap_paths = generate_gradcams(image_path)
    
    heatmap_files = {
        model_name: heatmap_path.split('/')[-1]  # Extract filename from full path
        for model_name, heatmap_path in heatmap_paths.items()
    }
    return render_template('heatmap.html', gradcam_paths=heatmap_files)

@app.route('/download_report')
def download_report():
    from flask import send_file  

    output_path = "static/report.pdf"
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "BlazeAid Heatmap Analysis Report")

    # Grad-CAM Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "Grad-CAM Heatmaps")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 120, "VGG Model:")
    c.drawImage("heatmaps/vgg_gradcam.jpg", 50, height - 370, width=200, height=200)

    c.drawString(300, height - 120, "Inception Model:")
    c.drawImage("heatmaps/inception_gradcam.jpg", 300, height - 370, width=200, height=200)

    c.drawString(50, height - 400, "Residual Model:")
    c.drawImage("heatmaps/residual_gradcam.jpg", 50, height - 650, width=200, height=200)

    # Fetch from session
    predicted_class = session.get('predicted_class', 'N/A')
    lime_filename = session.get('lime_heatmap', 'lime_default.png')

    # LIME Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(300, height - 400, "LIME Heatmap")

   # LIME Heatmap Section
    lime_path = os.path.join(app.config['HEATMAP_FOLDER'], lime_filename)
    if os.path.exists(lime_path):
        c.drawImage(lime_path, 300, height - 650, width=200, height=200)
    else:
        c.drawString(50, height - 140, "LIME image not found.")

    c.showPage()
    c.save()

    return send_file(output_path, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

