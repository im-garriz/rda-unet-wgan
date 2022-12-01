from configuration_flags import *

"""
Script de configuracion de dataset en Pytorch ##
    Iñaki Martinez: 18/02/2021
"""

# GPUs que se van a emplear para entrenar
USING_GPUS = GPU_0  # NO_GPU, GPU_0, GPU_1, GPU_2, GPU_3, ALL

# Indicador que determina si se va a usar data augmentation o no
DATA_AUG = AUGMENTATION_NO # AUGMENTATION_YES, AUGMENTATION_NO

# Conifguracion del conjunto de datos

# Rutas necesarias para los conjuntos de datos
TRAIN_DATA_DIR = "/workspace/shared_files/Dataset_BUSI_with_GT/"
TRAIN_DATA_ANNOTATIONS_FILE = "train_bus_images.csv"
VAL_DATA_FILE = "/workspace/shared_files/Dataset_BUSI_with_GT/"
VAL_DATA_ANNOTATIONS_FILE = "val_bus_images.csv"

# Tamaño de los batches
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

# Numero de nucleos a usar por los DataLoader
DATA_LOADERS_CORE_NUMBER = 9 # Aqui interesa poner todos los core que permita el PC sin que envie SIGKILL

# MODELO

# Segun se va a emplear transfer learning con un modelo preentrenado
PRETRAINED_WEIGHTS = PRETRAINED_FALSE # PRETRAINED_TRUE,PRETRAINED_FALSE

# Si vamos a entrenar desde un checkpoint. Tiene predomiancia sobre PRETRAINED_WEIGHTS
TRAIN_FROM_CHECKPOINT = CHECKPOINT_FALSE # CHECKPOINT_TRUE, CHECKPOINT_FALSE
CHECKPOINT_WEIGHTS_PATH = "../pesos/weights_2021-02-26 06:57:22.448560"

# Las primeras x capas que se van a freezear (para el caso de transfer learning), si -1 no se freezear nada
FREEZE_UNTIL_LAYER = -1

# Cantidad de neuronas de salida (en caso de salida fully connected)
N_OUTPUTS = 2

# Ruta donde se van a guardar los pesos del modelo, si "" no se guardan
MODEL_WEIGHTS_SAVE_PATH = "./weights_bus_gan"

# Hiperparametros
LEARNING_RATE = 0.0005
NUMBER_OF_EPOCHS = 120

# Ruta donde guardar los logs de tensorboard
TENSORBOARD_LOGDIR = "../runs"

# Para las GANs
OUT_WIDTH = 500
OUT_HEIGHT = 500
OUT_DEPTH = 3
AUTOENCODER_DEPTH = 3
