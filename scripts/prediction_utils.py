import copy
import numpy as np
from PIL import Image
import io
from google.cloud.vision import types, ImageAnnotatorClient
import sqlalchemy


# Get cereal info from DB
def connect_to_db():
    passwd = open('../db_info','r').readlines()[1].split()[0]
    username = open('../db_info','r').readlines()[0].split()[0]
    dbname = 'cereals'
    db = sqlalchemy.create_engine(f'mysql+pymysql://{username}:{passwd}@localhost/{dbname}')
    conn = db.connect()
    return conn


def generate_stacked_image(YOLO_predictions, img):
    """ Generate composite image of stacked cereal boxes to
    improve OCR results"""
    widths = []
    heights = []
    for box in YOLO_predictions:
        heights.append(box[3] - box[1])
        widths.append(box[2]-box[0])
    stacked_image = Image.new('RGB', (max(widths), sum(heights)))

    stacked_img_edges = np.concatenate([np.array([0]), np.cumsum(heights)[:-1], np.array([sum(heights)])])
    for box, y_top in zip(YOLO_predictions, stacked_img_edges[:-1]):
        subimg = copy.deepcopy(img)
        subimg = subimg.crop((box[0], box[1], box[2], box[3]))
        stacked_image.paste(subimg, (0,y_top))

    return stacked_image, stacked_img_edges


def Vision_API_OCR(stacked_image):
    """ Send image to Google Cloud Vision text detection
    API and parse result"""

    with io.BytesIO() as output:
        stacked_image.save(output, format="GIF")
        content = output.getvalue()

    # Instantiates a client
    OCR_client = ImageAnnotatorClient()

    image = types.Image(content=content)

    response = OCR_client.text_detection(image=image)

    texts = response.text_annotations

    OCR_words_all = []
    OCR_vertices = []
    for text in texts[1:]:
        OCR_words_all.append(text.description.lower())
        x = np.empty([0], dtype=int)
        y = np.empty([0], dtype=int)
        for vertex in text.bounding_poly.vertices:
            x = np.append(x,vertex.x)
            y = np.append(y,vertex.y)
        vertices = {
            "x": x,
            "y": y
        }
        OCR_vertices.append(vertices)

    return OCR_words_all, OCR_vertices

def process_string_for_comparison(string):
    return string.lower().replace("'","").replace("-", " ")

def jaccard_similarity(set1, set2):
    """ Calculate Jaccard similarity between two sets of strings"""
    intersect = set1.intersection(set2)
    return len(intersect) / (len(set1) + len(set2) - len(intersect))


def get_cereal(df, ocr_words):
    # add jaccard column to dataframe
    df["jaccard"] = np.nan

    for ix, row in df.iterrows():
        # pre-process cereal name
        cereal = row['cereal_name'] + " " + row['company']
        cereal = process_string_for_comparison(cereal)
        cereal = set(cereal.split())
        # Get jaccard and add to dataframe
        jaccard = jaccard_similarity(ocr_words, cereal)
        df.loc[ix, "jaccard"] = jaccard

    # if no jaccard greater than zero, return empty string
    if df['jaccard'].max() == 0:
        predicted_cereal = ''
    else:
        predicted_cereal = df['cereal_name'][df['jaccard'].idxmax()]

    return predicted_cereal
