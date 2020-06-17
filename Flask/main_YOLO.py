import pandas as pd
import os
import unidecode

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
    ax.set_axis_off()        # remove background
    ax.imshow(img);          # Plot image

    # Plot bonding boxes and annotations
    for result in prediction_results:
        box = result['box']
        label = result['label']
        confidence = result['confidence']

        # if no allergies selected, all are safe
        if len(allergies) == 0:
            color = 'lime'
        else:
            # if cereal type is unknown
            if label == '':
                color = 'yellow'
            else:
                # if dangerous
                if df[df.cereal_name==label][allergies].sum().sum() > 0:
                    color = 'red'
                else:
                    # if unknown
                    if df[df.cereal_name==label][allergies].isnull().values.any():
                        color = 'yellow'
                    # if safe
                    else:
                        color = 'lime'

        # if not confident in cereal identification
        if confidence == 0 and label != '':
            color = 'yellow'
            if df[df.cereal_name==label][allergies].sum().sum() > 0:
                label = 'Not safe if:\n' + label
            else:
                if df[df.cereal_name==label][allergies].isnull().values.any():
                    label = label
                else:
                    label = 'Safe if:\n' + label

        fontsize = (box[3]-box[1]) / img.size[1]

        # plot boxes
        ax.plot([box[0],box[2], box[2],box[0],box[0]],
            [box[1],box[1],box[3],box[3],box[1]],
            linewidth=2, color=color)
        # set font color
        if color == 'lime' or color=='yellow':
            font_color = 'black'
        else:
            font_color = 'white'
        # plot labels
        ax.text(np.mean([box[0], box[2]]),
            np.mean([box[1],box[3]]),
            label.replace(" ", "\n"),
            fontsize=np.max([fontsize*20, 6]),
            color=font_color,
            bbox=dict(facecolor=color, alpha=0.6),
            horizontalalignment='center', verticalalignment='center')

    # Save figure to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=300)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    figure =  f"data:image/png;base64,{data}"
    return figure

def predict(img, df):
    # Run YOLO detector
    YOLO_predictions, _ = yolo.detect_image(img);

    # Only permit bounding boxes with height > width and height / width > 2.5
    YOLO_predictions = [box for box in YOLO_predictions if (box[3]-box[1]) > (box[2]-box[0]) and (box[3]-box[1]) / (box[2]-box[0]) < 2.5]

    # Remove bounding boxes smaller than mean_box_area - coefficient*std_box_area
    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in YOLO_predictions]
    YOLO_predictions = [box for box, area in zip(YOLO_predictions, areas) if area > (np.mean(areas)-1.5*np.std(areas))]

    # create empty result list
    prediction_results = []

    # Deal with case that no cereal boxes found
    if len(YOLO_predictions) == 0:
        return prediction_results
    else:
        # Generate vertically-stacked image
        stacked_image, stacked_img_edges = generate_stacked_image(YOLO_predictions, img)

        # Detect text with Cloud Vision API and parse results
        OCR_words_all, OCR_vertices = Vision_API_OCR(stacked_image)

        # loop through detected boxes and identify each
        for ix, box in enumerate(YOLO_predictions):
            # Find OCR text in this cereal box
            OCR_words = []
            OCR_areas = np.empty((0))
            for word, vertices in zip(OCR_words_all, OCR_vertices):
                if (vertices['y'] > stacked_img_edges[ix]).all() & (vertices['y'] < stacked_img_edges[ix+1]).all():
                    word = unidecode.unidecode(word)
                    OCR_words.append(process_string_for_comparison(word))
                    OCR_areas = np.append(OCR_areas, PolygonArea(vertices['x'], vertices['y']))
            if len(OCR_words) > 0:
                # predict cereal
                label, confidence = get_cereal2(df, OCR_words, OCR_areas)
                # compile results in dictionary
                prediction_results.append({
                    'OCR': OCR_words,
                    'box': box[0:4],
                    'label': label,
                    'confidence': confidence
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
s = 'select * from cereals2'
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
