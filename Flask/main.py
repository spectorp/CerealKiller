import flask
from flask import Flask, render_template, request, send_file
from flask import Response

from PIL import Image
from werkzeug.utils import secure_filename

import numpy as np
import base64
from io import BytesIO
from matplotlib.figure import Figure

import sys
sys.path.insert(0, "../scripts")
from image_processing import *
import sqlalchemy
import pandas as pd

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

#-----------------------------------------------

def generate_plot(img, bboxes, labels):
    """
    Generate the figure **without using pyplot**
    """
    fig = Figure()
    ax = fig.subplots()
    ax.set_axis_off()

    ax.imshow(img.permute(1, 2, 0));
    for box, label in zip(bboxes, labels):
        # plot boxes
        ax.plot([box[0],box[2], box[2],box[0],box[0]],
            [box[1],box[1],box[3],box[3],box[1]],
            linewidth=2, color='lime')
        # plot labels
        ax.text(np.mean([box[0], box[2]]),
            np.mean([box[1],box[3]]),
            str(label),
            fontsize=15,
            bbox=dict(facecolor='lime', alpha=0.5),
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

def load_model(num_classes):
    # Load Faster R-CNN model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # load model weights
    weights_file = "../notebooks/best_model_weights_epoch_20.pt"
    weights = torch.load(weights_file, map_location=lambda storage, loc:storage)
    model.load_state_dict(weights)
    model.eval()
    return model

#-----------------------------------------------

app = Flask(__name__)

@app.route("/")

@app.route('/index', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        # Get cereal info from DB
        s = 'select cereal_name, short_name, label_id from cereals'
        df = pd.read_sql(s, connect_to_db())

        # load model
        model = load_model(num_classes=len(df['label_id']))

        # Get uploaded image
        file = request.files['image']
        filename = secure_filename(file.filename)
        img = Image.open(file).convert("RGB")
        print(filename)
        print(type(img))

        # set output image size
        #out_px = 224
        # resize and crop
        #img = resize_bg_img(img, out_px)

        # convert to PyTorch tensor
        img = F.to_tensor(img)

        preprocessed_img = img.unsqueeze(0)

        with torch.no_grad():                                     # Without tracking gradients,
            pred = model(preprocessed_img)

        bboxes = np.asarray(pred[0]['boxes'])
        labels = np.asarray(pred[0]['labels'])

        # generate figure with matplotlib
        figure = generate_plot(img, bboxes, labels)

        return render_template("index.html", figure = figure)
    else:
        return render_template('index.html')

#-----------------------------------------------------------------------
#
#-----------------------------------------------------------------------
if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8080, debug=True)
