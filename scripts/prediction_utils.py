
# Utilities for identifying a cereal box given an image
#
# Perry Spector, 2020, spectorp@gmail.com


import copy
import numpy as np
from PIL import Image
import io
from google.cloud.vision import types, ImageAnnotatorClient
import sqlalchemy

# Get cereal info from DB
def connect_to_db():
    """ Return connection to MySQL DB"""
    passwd = open('../db_info','r').readlines()[1].split()[0]
    username = open('../db_info','r').readlines()[0].split()[0]
    dbname = 'cereals'
    db = sqlalchemy.create_engine(f'mysql+pymysql://{username}:{passwd}@localhost/{dbname}')
    conn = db.connect()
    return conn

def generate_stacked_image(YOLO_predictions, img):
    """
    Generate composite image of vertically-stacked cereal boxes. The basic idea here is that
    if you send an image with multiple cereal boxes on the shelf to Google Cloud Vision's
    API, it sometimes groups text spanning multiple cereal boxes into a single text box. One
    solution would be to crop out each cereal box and send it to the API individually, but that's
    slow. Here, we create a vertical "film strip" image, so that we can send multiple cereal
    boxes to the API while only getting text from each box.
    """
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
    """ Send image to Google Cloud Vision APi for text detection and parse result"""
    with io.BytesIO() as output:
        stacked_image.save(output, format="JPEG")
        content = output.getvalue()
    OCR_client = ImageAnnotatorClient() # Instantiates a client
    image = types.Image(content=content)
    # Detect text using API. Note, setting language_hints to english. This seems to have fixed
    # a problem where it was interpreting some characters as written in cyrillic.
    response = OCR_client.text_detection(image=image, image_context={"language_hints": ["en"]})
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
        vertices = {"x": x, "y": y}
        OCR_vertices.append(vertices)
    return OCR_words_all, OCR_vertices

def process_string_for_comparison(string):
    """ minor string pre-processing"""
    return string.lower().replace("'","").replace("-", " ")

def modified_jaccard_similarity(ocr_words, set2, ocr_areas=None, use_fontsize=True):
    """
    Calculate modified Jaccard similarity between two sets of strings. Accounts for
    fontsize by scaling the Jaccard similarity by the sum of the fractional area of
    the detected text boxes in the intersection (the numerator in the Jaccard
    similarity).

    Optional to just calculate Jaccard similarity by setting use_fontsize to False.
    """
    intersect = set(ocr_words).intersection(set2)
    if use_fontsize == True:
        total_area = np.sum(ocr_areas)
        fractional_area = 0
        for word in intersect:
            indices = [i for i in range(len(ocr_words)) if ocr_words[i] == word]
            fractional_area += np.max(ocr_areas[indices] / total_area) # take max in case word occurs multiple times
        similarity = fractional_area * len(intersect) / (len(ocr_words) + len(set2) - len(intersect))
    else:
        similarity = len(intersect) / (len(ocr_words) + len(set2) - len(intersect))
    return similarity

def get_cereal(df, ocr_words, ocr_areas):
    """
    Named-entity recognition to identify cereal name given OCR-detected text.
    Outputs a string of predicted cereal name and a binary 0 or 1 confidence level.
    """
    # add jaccard column to dataframe
    df["jaccard"] = np.nan

    for ix, row in df.iterrows():
        # pre-process cereal name
        cereal = row['cereal_name'] + " " + row['company']
        cereal = process_string_for_comparison(cereal)
        cereal = set(cereal.split())
        # Get jaccard and add to dataframe
        jaccard = modified_jaccard_similarity(ocr_words, cereal, ocr_areas)
        df.loc[ix, "jaccard"] = jaccard

    # if no jaccard greater than zero OR multiple cereals share max jaccard, return empty string
    if df['jaccard'].max() == 0 or len(df.loc[df['jaccard']==df['jaccard'].max()]) > 1:
        predicted_cereal = ''
    else:
        predicted_cereal = df['cereal_name'][df['jaccard'].idxmax()]

    confidence = 1

    # only return positive identification if company name in OCR words
    predicted_company = df['company'][df['jaccard'].idxmax()]
    predicted_company = set(process_string_for_comparison(predicted_company).split())
    if len(predicted_company.intersection(set(ocr_words))) < len(predicted_company):
        confidence = 0
    # same for cereal name
    predicted_cereal_set = set(process_string_for_comparison(predicted_cereal).split())
    if len(predicted_cereal_set.intersection(set(ocr_words))) < len(predicted_cereal_set):
        confidence = 0

    return predicted_cereal, confidence

def PolygonArea(x,y):
    """Calculate area of a polygon given its vertices"""
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
