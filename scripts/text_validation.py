# Generate performance metrics for text recognition and string matching
#
# Perry Spector, 2020, spectorp@gmail.com

import pandas as pd
from prediction_utils import *
import os
from sklearn import metrics
import unidecode

# add Cloud Vision API key to environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../ServiceAccountToken_VisionAPI.json'

test_dir = '../data/test_text'
test_imgs = os.listdir(test_dir)

# Get cereal info from DB
passwd = open('../db_info','r').readlines()[1].split()[0]
username = open('../db_info','r').readlines()[0].split()[0]
dbname = 'cereals'
db = sqlalchemy.create_engine(f'mysql+pymysql://{username}:{passwd}@localhost/{dbname}')
conn = db.connect()
s = 'select * from cereals2'
df = pd.read_sql(s, connect_to_db())

gt_label_ids = []
gt_labels = []
predicted_labels = []
predicted_labels_id = []

np.random.seed(17)

# main loop
for iteration in range(100):

    # get random file ids
    indicies = np.random.randint(len(test_imgs), size=(1,np.random.randint(10, 20))).tolist()[0]
    print(f"Running iteration {iteration}")

    # load images, get labels
    imgs = []
    widths = []
    heights = []
    for index in indicies:
        test_img = test_imgs[index]
        gt_label_ids.append(int(test_img[:3]))
        gt_labels.append(df.loc[df['label_id']==gt_label_ids[-1]]['cereal_name'].to_string(index=False).strip())

        img = Image.open(os.path.join(test_dir, test_img))
        imgs.append(img)
        heights.append(img.size[1])
        widths.append(img.size[0])
    stacked_image = Image.new('RGB', (max(widths), sum(heights)))
    stacked_img_edges = np.concatenate([np.array([0]), np.cumsum(heights)[:-1], np.array([sum(heights)])])

    # make stacked image
    for img, y_top in zip(imgs, stacked_img_edges[:-1]):
        stacked_image.paste(img, (0,y_top))

    # Detect text with Cloud Vision API and parse results
    OCR_words_all, OCR_vertices = Vision_API_OCR(stacked_image)

    for ix, index in enumerate(imgs):
        OCR_words = []
        OCR_areas = np.empty((0))
        for word, vertices in zip(OCR_words_all, OCR_vertices):
            if (vertices['y'] > stacked_img_edges[ix]).all() & (vertices['y'] < stacked_img_edges[ix+1]).all():
                word = unidecode.unidecode(word)
                OCR_words.append(word)
                OCR_areas = np.append(OCR_areas, PolygonArea(vertices['x'], vertices['y']))

        # strip any whitespace
        OCR_words = list(map(str.strip, OCR_words))

        if len(OCR_words) > 0:
            label, confidence = get_cereal(df, OCR_words, OCR_areas)
            if confidence == 0: label = ''
            predicted_labels.append( label )
            predicted_labels_id.append(df.loc[df['cereal_name']==predicted_labels[-1]]['label_id'].to_numpy())

true = []
predict = []
for gt, label in zip(gt_labels, predicted_labels):
    if label != '':
        true.append(gt)
        predict.append(label)
# Print the precision and recall, among other metrics
print(metrics.classification_report(true, predict, digits=3))
