class Configer:

  #set pre-trained model path
  EAST_MODEL_CHECKPOINT_FOLD = r'C:\Users\LSC-110\Desktop\east_icdar2015_resnet_v1_50_rbox'

  #set logging path
  LOGGING_FILE_PATH = r''

  #set gpu list, if multiple GPUs, list their index,start from 0
  GPU_LIST = 0

  #set test data parameters
  TEST_IMAGES_PATH = r''
  TEST_RESULTS_FOLD = r''
  RESULT_IMAGES_WRITE_FLAG = True

  #set images loaded batch
  IMAGES_BATCH_IN_QUEUE = 10

