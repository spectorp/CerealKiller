import pandas as pd
import os

# App packages
import flask
from flask import Flask, render_template, request, send_file
from flask import Response
from werkzeug.utils import secure_filename

# Model-related packages
import sys
sys.path.insert(0, "../scripts")
YOLO_path = '../YOLOv3/TrainYourOwnYOLO'
src_path = os.path.join(YOLO_path, "2_Training", "src")
utils_path = os.path.join(YOLO_path, "Utils")
sys.path.append(src_path)
sys.path.append(utils_path)
from keras_yolo3.yolo import YOLO
from utils import detect_object

# Packages required for image processing and figure generation
from matplotlib.figure import Figure
from image_processing import *
from prediction_utils import *
import base64
from io import BytesIO

#-----------------------------------------------
#                 functions
#-----------------------------------------------

def generate_plot(img, prediction_results, allergies, df):
    """
    Generate the figure **without using pyplot**
    """
    fig = Figure()
    ax = fig.subplots()
    # remove background
    ax.set_axis_off()
    # Plot image
    ax.imshow(img);

    # Plot bonding boxes and annotations
    for result in prediction_results:
        box = result['box']
        label = result['label']
        fontsize = (box[3]-box[1]) / img.size[1]

        # if dangerous
        if df[df.cereal_name==label][allergies].sum().sum() > 0:
            color = 'red'
            symbol = 'X'
        else:
            # if unknown
            if df[df.cereal_name==label][allergies].isnull().values.any():
                color = 'orange'
                symbol = '?'
            # if safe
            else:
                color = 'lime'
                symbol = u'\u2713'

        # plot boxes
        ax.plot([box[0],box[2], box[2],box[0],box[0]],
            [box[1],box[1],box[3],box[3],box[1]],
            linewidth=2, color=color)
        # plot symbol
        ax.text(np.mean([box[0], box[2]]),
            np.mean([box[1],box[3]]),
            symbol,
            fontsize=fontsize*250,
            color=color,
            horizontalalignment='center',
            verticalalignment='center')
        # plot labels
        ax.text(np.mean([box[0], box[2]]),
            np.mean([box[1],box[3]]),
            str(label),
            fontsize=fontsize*20,
            bbox=dict(facecolor=color, alpha=0.5),
            horizontalalignment='center')

    # Save figure to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    figure =  f"data:image/png;base64,{data}"
    return figure

def predict(img, df):
    # Run YOLO detector
    YOLO_predictions, _ = yolo.detect_image(img);

    # Generate vertically-stacked image
    stacked_image, stacked_img_edges = generate_stacked_image(YOLO_predictions, img)

    # Detect text with Cloud Vision API and parse results
    OCR_words_all, OCR_vertices = Vision_API_OCR(stacked_image)

    # loop through detected boxes and identify each
    prediction_results = []
    for ix, box in enumerate(YOLO_predictions):
        # Find OCR text in this cereal box
        OCR_words = set()
        for word, vertices in zip(OCR_words_all, OCR_vertices):
            if (vertices['y'] > stacked_img_edges[ix]).all() & (vertices['y'] < stacked_img_edges[ix+1]).all():
                OCR_words.add(word)
        if len(OCR_words) > 0:
            # predict cereal
            label = get_cereal(df, OCR_words)
            # compile results in dictionary
            prediction_results.append({
                'OCR': OCR_words,
                'box': box[0:4],
                'label': label
            })
    return prediction_results

#-----------------------------------------------
#  load models and query DB during app spinup
#-----------------------------------------------

# load YOLO
print('Loading YOLO')
model_weights = os.path.join(YOLO_path, 'Data', 'Model_Weights', "trained_weights_final_ck2000.h5")
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")
model_classes = os.path.join(YOLO_path, 'Data', 'Model_Weights', "data_classes.txt")
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.25,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

# add Cloud Vision API key to environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../ServiceAccountToken_VisionAPI.json'

# Get cereal info from DB
s = 'select * from cereals'
df = pd.read_sql(s, connect_to_db())


#-----------------------------------------------
#                run app
#-----------------------------------------------

app = Flask(__name__)

@app.route("/")

@app.route('/index', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        # get list of allergies
        allergies = request.form.getlist('allergy')

        # Get uploaded image
        file = request.files['image']
        filename = secure_filename(file.filename)
        img = Image.open(file).convert("RGB")
        new_img = copy.deepcopy(img)

        # process image, predict bounding boxes, predict cereal using OCR
        prediction_results = predict(new_img, df)

        # generate figure with matplotlib
        figure = generate_plot(img, prediction_results, allergies, df)

        return render_template("index.html", figure = figure)
    else:
        return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/slides')
def slides():
    return render_template('slides.html')

#-----------------------------------------------------------------------
#
#-----------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True, threaded=False)
