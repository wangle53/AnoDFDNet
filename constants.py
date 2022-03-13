# training configuration
EPOCH = 100
ISIZE = 256
BATCH_SIZE = 4
LR = 0.0002
DISPLAY = True # if display training images on visdom
RESUME = False  # if training from the last epoch
PRINT_STEP = 5

# dirs
DATA_PATH = 'H:\\ScLeadChangeDetectionDataset\\LPT-2DCD-paper\\scleadChangeDetectionDatasetV2\\experiment_paper'
TXT_PATH = './data'
TRAIN_TXT = './data/train.txt'
TEST_TXT = './data/test.txt'
VAL_TXT = './data/validation.txt'
IM_SAVE_DIR = './outputs/tainging_images'
WEIGHTS_SAVE_DIR = './outputs/model'

# val
CRITERIA = 'f1'
METRIC = 'roc'
THRESHOLD = 0.5

# init network
NC = 3 #input image channel size 
NZ = 1024 #size of the latent z vector
NDF = 64
NGF = 64
EXTRALAYERS = 1
Z_SIZE = 16
GT_C = 1
D_IN = 1

# init transformer
USE_TR = True
HIDDEN_SIZE = NZ
INPUT_FEATURES_DIMS = NZ  # output channel dim of backbone
HEADS = 16
GRID_SIZE = 16
DOWNSAMPLING_RATE = 16
ENCODER_LAYERS = 2


