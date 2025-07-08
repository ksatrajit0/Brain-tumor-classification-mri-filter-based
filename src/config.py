IMG_SIZE = (256, 256)
BATCH_SIZE = 16
CHANNELS = 3
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)

TRAIN_DIR = '/content/cleaned/Training'
TEST_DIR = '/content/cleaned/Testing'

CSV_TRAIN_PATH = '/content/training_global_average_pool_2d_efnetb3_trained_weights_new.csv'
CSV_TEST_PATH = '/content/testing_global_average_pool_2d_efnetb3_trained_weights_new.csv'

LABELS_DICT = {'glioma': 1, 'notumor': 0, 'meningioma': 2, 'pituitary': 3}
