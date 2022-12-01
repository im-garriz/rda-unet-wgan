import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "malignant"
color = (0, 128, 255)

LABELSIZE = 18
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=LABELSIZE)
plt.rc('ytick', labelsize=LABELSIZE)
plt.rc('axes', labelsize=LABELSIZE)


def generate_images_grid(folder):

    image_names = os.listdir(folder)
    grid = np.zeros([128*8, 128*8, 3])

    for image_name in image_names:

        image = cv2.imread(os.path.join(folder, image_name))
        _row, _col = image_name.split('.')[0].split('_')

        row = int(_row)
        col = int(_col)


        if row != 0:
            seg = image[128*2:, :, 0]
            image = image[:128, :, :]
            contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)

            grid[row*128:(row+1)*128, col*128:(col+1)*128, :] = image

        else:
            image_ = np.zeros(image.shape)

            radio = image[:128, :, :].copy()
            radio_2 = radio.copy()
            gt = image[128:128*2, :, 0]
            seg = image[128*2:, :, 0]

            contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(radio, contours, -1, color, 2)

            contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(radio_2, contours, -1, color, 2)

            image_[:128, :, :] = image[:128, :, :].copy()
            image_[128:128*2, :, :] = radio.copy()
            image_[128*2:, :, :] = radio_2.copy()

            grid[row*128:(row+3)*128, col*128:(col+1)*128, :] = image_

    cv2.imwrite("grid_malignant.png", grid)


def learning_curves_dice(path="modelos finales/gradient_penalty"):

    dices_df = pd.read_csv(os.path.join(path, "run-.-tag-Metrics_dice coeff.csv"))
    dices_x = dices_df["Step"]
    dices_y = dices_df["Value"]

    generator_loss_df = pd.read_csv(os.path.join(path,
                                    "run-.-tag-Per epoch train loss_generator.csv"))
    generator_loss_df_x = generator_loss_df["Step"]
    generator_loss_df_y = generator_loss_df["Value"]

    discriminator_loss_df = pd.read_csv(os.path.join(path,
                                    "run-.-tag-Per epoch train loss_discriminator.csv"))
    discriminator_loss_df_x = discriminator_loss_df["Step"][1:]
    discriminator_loss_df_y = discriminator_loss_df["Value"][1:]

    figure, axes = plt.subplots(1,2, figsize=(12, 5))

    #axes[0].set_title("Generator/critic loss vs epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Wasserstein loss")
    axes[0].plot(generator_loss_df_x, generator_loss_df_y, label="Generator")
    axes[0].plot(discriminator_loss_df_x, discriminator_loss_df_y, label="Critic")
    axes[0].legend(loc="lower right", fontsize=LABELSIZE)

    #axes[1].set_title("Dice value on validation data vs epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice value on validation data")
    axes[1].plot(dices_x, dices_y)


    figure.tight_layout()

    plt.savefig("progess_curves_gradient_penalty.png")


def learning_curves_classif():

    folders = os.listdir("modelos finales_classif")

    figure, axes = plt.subplots(1,2, figsize=(12, 5))

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CCR (\%)")

    for folder in folders:

        loss_df = pd.read_csv(os.path.join("modelos finales_classif", folder,
                                    "run-.-tag-loss_per_epoch_val.csv"))
        loss_df_x = loss_df["Step"]
        loss_df_y = loss_df["Value"]

        ccr_df = pd.read_csv(os.path.join("modelos finales_classif", folder,
                                        "run-.-tag-accuracy_per_epoch_val.csv"))
        ccr_df_x = ccr_df["Step"]
        ccr_df_y = ccr_df["Value"]

        axes[0].plot(loss_df_x, loss_df_y, label=folder)
        axes[1].plot(ccr_df_x, ccr_df_y, label=folder)

    axes[0].legend(loc="upper right", fontsize=LABELSIZE)
    axes[1].legend(loc="lower right", fontsize=LABELSIZE)

    figure.tight_layout()
    plt.savefig("progess_curves_class.png")


learning_curves_classif()

