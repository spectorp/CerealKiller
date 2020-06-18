# Generate artificial images of cereal boxes superimposed on scenes of
# grocery store shelves. Images are used for training an object
# detection model.
#
# Perry Spector, 2020, spectorp@gmail.com

from image_processing import *

# directory of background images
bg_dir_path = os.path.join('..', 'data', 'raw', 'shelves')

# directories of foreground images
fg_dirs = [ f.path for f in os.scandir(os.path.join('..', 'data', 'raw')) if f.is_dir()]
fg_dirs = [fg_dir for fg_dir in fg_dirs if 'shelves' not in fg_dir]

# output directory
synth_dir = os.path.join('..', 'data', 'synthesized')

# Connect to database
passwd = open('../db_info','r').readlines()[1].split()[0]
username = open('../db_info','r').readlines()[0].split()[0]
dbname = 'cereals'
db = sqlalchemy.create_engine(f'mysql+pymysql://{username}:{passwd}@localhost/{dbname}')
conn = db.connect()

# set output image size
out_px = 224

# set max attempts to superimpose image
max_attempts = 100

# number of training images to generate
n_train_img = 2000

# generate and save images
all_targets = {}
for iter in range(n_train_img):
    print('Image ', iter)

    # generate image
    bg_img, target_list = make_training_image(bg_dir_path, fg_dirs, conn, out_px, max_attempts)

    # save image
    fname = f"{iter:05d}" + '.png'
    bg_img.save(os.path.join(synth_dir, 'PNG', fname))

    #all_targets.append(target)
    all_targets.update({fname: target_list})

# save target
np.save(os.path.join(synth_dir, 'targets.npy'), all_targets)
