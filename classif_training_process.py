from config import *

from dataset import load_data
from GPU_config import GPU_config
from img_transformations import get_train_img_transform, get_val_img_transform, UnNormalize
from model_handler import ModelHandler
from tensorboard_logger import TensorboardLogger
from progress_logger import ProgressLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

import time
import PIL
from torchvision import transforms
import os


class ClassifTrainingProcess:
    """
    Clase que representa el proceso de entrenamiento
    """

    def __init__(self):
        """
        Constructor: inicializa todos los elementos que forman el proceso de entrenamiento
        """

        # Carga lo correspondiente a los datos
        self.train_data_transform = get_train_img_transform()
        self.val_data_transform = get_val_img_transform()
        self.trainloader, self.valloader = load_data(self.train_data_transform, self.val_data_transform,
                                                     print_batches_info=True)
        self.un_normalizer = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # Carga lo correspondiente al modelo
        self.gpu_config = GPU_config(USING_GPUS)
        self.model_handler = ModelHandler(gpu_config=self.gpu_config)

        # Carga lo correspondiente al entrenamiento del modelo
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model_handler.model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        # Carga el logger
        self.tensorboard = TensorboardLogger()

        # Cargo variables auxiliares para calcular accuracys y losses
        self.total = 0
        self.correct = 0
        self.running_loss = 0
        self.running_val_loss = 0

        # Creo un objeto logger para prints de progreso
        self.logger = ProgressLogger(len(self.trainloader), len(self.valloader))

    def learn(self):
        """
        Proceso de entrnamiento
        :return:
        """

        for epoch in range(NUMBER_OF_EPOCHS):

            self.logger.print_epoch(epoch)

            # Train step
            self.train_step(epoch)
            train_accuracy = 100.0 * self.correct / self.total

            # Val step
            self.val_step()
            val_accuracy = 100.0 * self.correct / self.total

            # Logs a tensorboard
            self.tensorboard.update_per_epoch_values(self.running_loss, self.running_val_loss,
                                                     train_accuracy, val_accuracy, epoch)

            self.logger.finish_epoch()

            # Guardo los pesos (o no, segun config.py)
            self.model_handler.save_model_weights()

    def train_step(self, epoch):

        """
        Un paso de train

        :param epoch: epoch actual
        :return:
        """

        self.total = 0
        self.correct = 0

        self.running_loss = 0

        self.logger.initialize_train_bar()

        for i, data in enumerate(self.trainloader):
            # data -> lista de tuplas (inputs, labels)
            input, labels = self.gpu_config.data_to_GPU(data[0]), self.gpu_config.data_to_GPU(data[1])
            # resetea los gradientes
            self.optimizer.zero_grad()
            # Forward + backward + optimize
            self.model_handler.model.train()
            outputs = self.model_handler.model(input)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Devuelve values y indices, interesa indices
            self.total += labels.size(0)
            self.correct += (predicted == labels).sum().item()

            self.tensorboard.update_per_batch_train_loss_curve(loss, epoch * len(self.trainloader) + i)

            self.logger.update_bar()

        self.running_loss /= len(self.trainloader)

    def val_step(self):
        """
        Un paso de val

        :return:
        """

        self.total = 0
        self.correct = 0

        self.running_val_loss = 0

        self.logger.initialize_val_bar()

        with torch.no_grad():
            for i, data in enumerate(self.valloader):

                # data -> lista de tuplas (inputs, labels)
                input, labels = self.gpu_config.data_to_GPU(data[0]), self.gpu_config.data_to_GPU(data[1])

                prediction_val = self.model_handler.model(input)
                loss_val = self.criterion(prediction_val, labels)

                self.running_val_loss += loss_val.item()

                self.optimizer.zero_grad()

                _, predicted = torch.max(prediction_val.data, 1) # Devuelve values y indices, interesa indices
                self.total += labels.size(0)
                self.correct += (predicted == labels).sum().item()

                self.logger.update_bar()

        self.running_val_loss /= len(self.valloader)

    def inference(self):

        with torch.no_grad():
            for i, data in enumerate(self.valloader):

                # data -> lista de tuplas (inputs, labels)
                input, labels = self.gpu_config.data_to_GPU(data[0]), self.gpu_config.data_to_GPU(data[1])

                prediction_val = self.model_handler.model(input)
                self.optimizer.zero_grad()

                _, predicted = torch.max(prediction_val.data, 1) # Devuelve values y indices, interesa indices

                correct = (predicted == labels)
                trans = transforms.ToPILImage()

                for j, image in enumerate(data[0]):
                    if j == 0:
                        self.tensorboard.add_images(data[0])

                    if correct[j]:
                        folder = "../Inferencia/OK"
                    else:
                        folder = "../Inferencia/NOK"

                    image = self.un_normalizer(image)
                    image = trans(image)

                    #print(data[2][j])
                    name = data[2][j].split('/')

                    image.save(os.path.join(folder, f"{predicted[j]}_{labels[j]}_{name[0]}_{name[1]}_{name[2]}.png"))

    def save_model_onnx_format(self):
        dummy_input = torch.randn(1, 3, 384, 512)
        dummy_input = self.gpu_config.data_to_GPU(dummy_input)

        torch.onnx.export(self.model_handler.model.module, dummy_input, "onnx_resnet_ebaki.onnx", input_names=['input'],
                          output_names=['output'], export_params=True)

        print('ONNX model succesfully saved')
