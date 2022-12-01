import torch
import pandas as pd
import os
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import cv2


"""
Script de configuracion de dataset en Pytorch ##
    Iñaki Martinez: 18/02/2021
"""

# Rutas necesarias para los conjuntos de datos
TRAIN_DATA_DIR = "/workspace/shared_files/Dataset_BUSI_with_GT/"
TRAIN_DATA_ANNOTATIONS_FILE = "gan_train_bus_images.csv"
VAL_DATA_FILE = "/workspace/shared_files/Dataset_BUSI_with_GT/"
VAL_DATA_ANNOTATIONS_FILE = "gan_val_bus_images.csv"

# Tamaño de los batches
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE

# Numero de nucleos a usar por los DataLoader
DATA_LOADERS_CORE_NUMBER = 9 # Aqui interesa poner todos los core que permita el PC sin que envie SIGKILL

class Dataset(torch.utils.data.Dataset):

    """
    Para usar un dataset en pytorch, este tiene que heredar de torch.utils.data.Dataset. Ademas, hay que implementar
    los metodos __len__() de manera que este retorne el tamaño del dastaset y __getitem__ de manera que se pueda
    indexar (ej: dataset[i])
    """

    def __init__(self, csv_file, data_root_dir, transform=None, augmentation_pipeline=None):
        """
        En el constructor simplemente se almecenan los datos

        :param csv_file: archivo con las anotaciones
        :param data_root_dir: directorio de las imagenes
        :param transform: transformacion a aplicar a las imagenes
        """
        self.data = pd.read_csv(os.path.join(data_root_dir, csv_file))
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self):
        """
        En este caso cada fila del csv es una imagen
        :return: longitud del dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Lee de disco y retorna la imagen dataset[idx]
        :param idx: indice a retornar
        :return: imagen y labels correspondientes al indice idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        full_image_name = os.path.join(self.data_root_dir, self.data.iloc[idx, 0])
        image = Image.open(full_image_name).convert("RGB")
        ground_truth_image_name = os.path.join(self.data_root_dir, self.data.iloc[idx, 1])
        ground_truth = Image.open(ground_truth_image_name).convert("L")

        #image = cv2.imread(full_image_name)
        #image = image.astype(np.float) / 255.0

        #ground_truth = cv2.imread(ground_truth_image_name)[:,:,0]
        #ground_truth = ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1)
        ##ground_truth = ground_truth == 255
        #ground_truth = ground_truth.astype(np.float) / 255.0

        #cv2.imwrite(f"{idx}_test.png", image * ground_truth)

        #print(np.all(ground_truth[:,:,0] == ground_truth[:,:,2]))
        #print(np.max(ground_truth))
        #print( f"SUM: {np.sum(ground_truth == 1.)}")
        #print(ground_truth.shape)

        if self.augmentation_pipeline:
            image, ground_truth = self.augmentation_pipeline(np.array(image), np.array(ground_truth))
            #img = np.vstack([image, ground_truth.draw(size=ground_truth.shape)[0]])
            #cv2.imwrite(f"{idx}_test.png", img)
            #cv2.imwrite(f"{idx}_test.png", image*ground_truth.draw(size=ground_truth.shape)[0])
            #   print(ground_truth.draw(size=ground_truth.shape)[0].shape)
            image = Image.fromarray(image)
            ground_truth = Image.fromarray(ground_truth.draw(size=ground_truth.shape)[0]).convert('L')
            #print("guardo")

        else:
            pass
            #ground_truth = Image.fromarray((ground_truth * 255).astype(np.uint8).reshape(ground_truth.shape[:2]))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray((image * 255).astype(np.uint8))


        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        # En este caso como es clasificacion no hay que transformar labels, si fuera una mascara habria que retornarlos
        # juntos, como una tupla o diccionario: {'image': image, 'mask': mask}
        return image, ground_truth, self.data.iloc[idx, 0]


def load_data(train_data_transform, val_data_trasform, augmentation_pipeline, print_batches_info=False):
    """
    Funcion que genera los DataLoaders correspondientes a cada uno de los conjuntos

    :param data_transform: transformacion a aplicar
    :param print_batches_info: si queremos que muetre en pantalla el tamaño de los batches

    :return: trainloader y valloader
    """
    trainset = Dataset(TRAIN_DATA_ANNOTATIONS_FILE,
                       TRAIN_DATA_DIR,
                       transform=train_data_transform,
                       augmentation_pipeline=augmentation_pipeline)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                              num_workers=DATA_LOADERS_CORE_NUMBER)

    valset = Dataset(VAL_DATA_ANNOTATIONS_FILE,
                     VAL_DATA_FILE,
                     transform=val_data_trasform,
                     augmentation_pipeline=None)

    valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True,
                                             num_workers=DATA_LOADERS_CORE_NUMBER)

    if print_batches_info:
        print(f"Train set length: {len(trainset)}")
        print(f"Val set length: {len(valset)}")
        print("Mini-batches size:")
        print(f"\tTrain set: {len(trainloader)}")
        print(f"\tTest set: {len(valloader)}")

    return trainloader, valloader
