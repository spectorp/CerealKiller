import numpy as np
import pandas as pd
import sqlalchemy
import copy

# App packages
import flask
from flask import Flask, render_template, request, send_file
from flask import Response
from werkzeug.utils import secure_filename

# Model-related packages
import sys
sys.path.insert(0, "../scripts")
import keras_ocr
YOLO_path = '/home/perry/datasci/study_guide/examples/YOLO/TrainYourOwnYOLO'
src_path = os.path.join(YOLO_path, "2_Training", "src")
utils_path = os.path.join(YOLO_path, "Utils")
sys.path.append(src_path)
sys.path.append(utils_path)
from keras_yolo3.yolo import YOLO
from utils import detect_object

# Packages required for image processing and figure generation
from PIL import Image
from matplotlib.figure import Figure
from image_processing import *
import base64
from io import BytesIO


#-----------------------------------------------
#                 functions
#-----------------------------------------------

def generate_plot(img, bboxes, labels, allergies, df):
    """
    Generate the figure **without using pyplot**
    """
    fig = Figure()
    ax = fig.subplots()
    ax.set_axis_off()

    ax.imshow(img);
    for box, label in zip(bboxes, labels):
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







def connect_to_db():
    passwd = open('../db_info','r').readlines()[1].split()[0]
    username = open('../db_info','r').readlines()[0].split()[0]
    dbname = 'cereals'
    db = sqlalchemy.create_engine(f'mysql+pymysql://{username}:{passwd}@localhost/{dbname}')
    conn = db.connect()
    return conn

def load_RCNN(num_classes):
    # Load Faster R-CNN model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # load model weights
    #weights_file = "../notebooks/best_model_weights_singleclass_epoch_1.pt"
    weights_file = "../notebooks/best_model_weights_singleclass_epoch_9_F1_52.pt"
    weights = torch.load(weights_file, map_location=lambda storage, loc:storage)
    model.load_state_dict(weights)
    model.eval()
    return model

def get_jaccard_sim(set1, set2):
    intersect = set1.intersection(set2)
    return len(intersect) / (len(set1) + len(set2) - len(intersect))

def get_cereal(df, ocr_words):
    # add jaccard column to dataframe
    df["jaccard"] = np.nan

    for ix, row in df.iterrows():
        # pre-process cereal name
        cereal = row['cereal_name'] + " " + row['company']
        cereal = cereal.lower().replace("'","").replace("-", " ")
        cereal = set(cereal.split())
        # Get jaccard and add to dataframe
        jaccard = get_jaccard_sim(ocr_words, cereal)
        df.loc[ix, "jaccard"] = jaccard

    return df.sort_values(by=['jaccard'], ascending=False)['cereal_name'].iloc[0]

def find_textboxes_in_cerealbox(box, OCR_results):
    text_in_box = []
    for OCR_result in OCR_results:
        # consider text box to be in cereal bounding box only if all x-y coordinates are within cereal box
        xcoords = OCR_result[1][:,0]
        ycoords = OCR_result[1][:,1]
        if np.all((xcoords > box[0]) & (xcoords < box[2]) & (ycoords > box[1]) & (ycoords < box[3])):
            text_in_box.append(OCR_result)
    return text_in_box

def predict(img, df):
    bboxes, _ = yolo.detect_image(img)


    #bboxes = np.asarray(pred[0]['boxes'])
    #labels = np.asarray(pred[0]['labels'])

    # OCR
    print('Starting OCR')
    OCR_results = pipeline.recognize([np.asarray(img)])[0]

    # figure out labels
    labels = []
    for box in bboxes:
        df2 = copy.deepcopy(df)
        text_in_box = find_textboxes_in_cerealbox(box[0:4], OCR_results)
        ocr_words = set([text[0] for text in text_in_box]) # make set for jaccard sim
        ocr_words
        labels.append(get_cereal(df2, ocr_words))

    return bboxes, labels

#-----------------------------------------------
#  load models and query DB during app spinup
#-----------------------------------------------

# load YOLO
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

# Load keras-OCR
pipeline = keras_ocr.pipeline.Pipeline()

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

        # process image, predict bounding boxes, predict cereal using OCR
        bboxes, labels = predict(img, df)

        # generate figure with matplotlib
        figure = generate_plot(img, bboxes, labels, allergies, df)

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

    app.run(host='127.0.0.1', port=8080, debug=True)
