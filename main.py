# Ficheros propios
from rdau_net import *
from discriminator import *
from net import *
from augmentation import AugmentationPipeline
from img_transformations import UnNormalize
from metrics import *

# Librerias generales
import cv2
from dataset import *
import PIL
import random
import imgaug
from imgaug import augmenters as iaa

# Paquete pytorch
import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from loss import dice_loss

#https://github.com/eriklindernoren/PyTorch-GAN
#https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
# https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7

# Semilla de aleatoriedad fija
RANDOM_SEED = 42
print("Random Seed: ", RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
imgaug.seed(RANDOM_SEED)

# Hiperparametros de entrenamiento de la red
N_EPOCHS = 450
GENERATOR_LEARNING_RATE = 0.000025
DISCRIMINATOR_LEARNING_RATE = 0.000025

# Configuracion del hardware
ngpu = 1 # Numero de GPUs a emplear (0 -> cpu)
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Ruta en la que se van a guardar/cargar los pesos de las redes
WEIHGHTS_PATH = "unet"
PRETRAINED_WEIGHTS = False # Si empleamos pesos preentrenados previamente True, sino False

# En caso de empleo de gradient clipping
GRADIENT_CLIPPING = True
CLIPPING = (-0.2, 0.2)
GRADIENT_PENALTY = False

# En caso de empleo de data augmentation
DATA_AUG = False

# El generador se entrena cada GENERATOR_TRAIN_EVERY_EPOCHS veces
GENERATOR_TRAIN_EVERY_EPOCHS = 3

COLOR = False

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def unet_train():
    # Para visualizar empleando tensorboard
    writer = SummaryWriter()

    # Carga las transformaciones a hacer a las imagenes
    train_data_transform, val_data_transform = load_img_transforms()

    # Carga los cargadores de imagenes de train y val
    if DATA_AUG:
        augmentation_pipeline = AugmentationPipeline()
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=augmentation_pipeline)
    else:
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=None)

    # Inicializa el generador
    generator = RDAU_NET().to(DEVICE)
    #generator = NetS(ngpu).to(DEVICE)
    #generator.apply(weights_init)

    optimizerG = optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.9, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    # Listas para almacenar los costes de cada mini-batch, para asi sacar el medio para cada epoch
    G_losses = []
    D_losses = []

    ### BUCLE DE ENTRENAMIENTO ###

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch}")
        epoch_loss = 0

        # Por cada mini-batch
        for i, data in enumerate(trainloader):

            # Segmentaciones reales
            #optimizerD.zero_grad()
            optimizerG.zero_grad()
            generator.train()

            # Cargamos los datos
            images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

            segmentations = generator(images)
            #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)

            loss = calc_loss(segmentations, ground_truth)

            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + i)

            loss.backward()
            nn.utils.clip_grad_value_(generator.parameters(), 0.1)
            optimizerG.step()

        epoch_loss = epoch_loss / len(trainloader)
        writer.add_scalar('Per epoch loss/train', epoch_loss, epoch)

        val_epoch_loss = 0

        with torch.no_grad():
            generator.eval()
            for i, data in enumerate(valloader):
                images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

                segmentations = generator(images)
                #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)

                loss = criterion(ground_truth, segmentations)

                val_epoch_loss += loss.item()

            val_epoch_loss = val_epoch_loss / len(valloader)
            writer.add_scalar('Per epoch loss/val', val_epoch_loss, epoch)

        torch.save(generator.state_dict(), os.path.join(WEIHGHTS_PATH, "unet"))

def wgan():

    """
    Funcion principal
    :return:
    """

    # Para visualizar empleando tensorboard
    writer = SummaryWriter()

    # Carga las transformaciones a hacer a las imagenes
    train_data_transform, val_data_transform = load_img_transforms()

    # Carga los cargadores de imagenes de train y val
    if DATA_AUG:
        augmentation_pipeline = AugmentationPipeline()
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=augmentation_pipeline)
    else:
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=None)

    # Inicializa el generador
    generator = RDAU_NET().to(DEVICE)
    #generator = NetS(ngpu).to(DEVICE)
    generator.apply(weights_init)

    # Inicializa el discriminador
    discriminator = Discriminator().to(DEVICE)
    discriminator.apply(weights_init)

    # Si queremos partir de una red entrenada previamente cargamos los pesos
    if PRETRAINED_WEIGHTS:
        generator.load_state_dict(torch.load(os.path.join(WEIHGHTS_PATH, "unet")))
        discriminator.load_state_dict(torch.load(os.path.join(WEIHGHTS_PATH, "discriminator_weights")))

    # Carga los optimizadores para cada red
    optimizerD = optim.RMSprop(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)
    optimizerG = optim.RMSprop(generator.parameters(), lr=GENERATOR_LEARNING_RATE)

    #optimizerD = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, betas=(0.5, 0.999))
    #optimizerG = optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.5, 0.999))

    # Listas para almacenar los costes de cada mini-batch, para asi sacar el medio para cada epoch
    G_losses = []
    D_losses = []

    ### BUCLE DE ENTRENAMIENTO ###

    for epoch in range(N_EPOCHS):


        ## COMENZAMOS POR EL DISCRIMINADOR ##

        # Por cada mini-batch
        for i, data in enumerate(trainloader):

            # Segmentaciones reales
            #optimizerD.zero_grad()
            optimizerD.zero_grad()

            # Cargamos los datos
            images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

            images_with_gt = merge_images_with_masks(images, ground_truth).to(DEVICE)
            #images_with_gt = images.clone() * ground_truth

            segmentations = generator(images).detach()
            segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
            #images_with_segmentations = images.clone() * segmentations

            images_with_segmentations = merge_images_with_masks(images, segmentations).to(DEVICE)

            loss_D = -torch.mean(discriminator(images_with_gt)) + torch.mean(discriminator(images_with_segmentations))

            if GRADIENT_PENALTY:
                _gradient_penalty = gradient_penalty(discriminator, images_with_gt, images_with_segmentations, 10, DEVICE)
                loss_D += _gradient_penalty

            # Calculamos gradientes
            loss_D.backward()
            optimizerD.step()

            # Gradient clipping
            if GRADIENT_CLIPPING:
                for p in discriminator.parameters():
                    p.data.clamp_(CLIPPING[0], CLIPPING[1])

            ## UNA VEZ ACABADO EL DISCRIMINADOR, SE ENTRENA EL GENERADOR ##

            if epoch % GENERATOR_TRAIN_EVERY_EPOCHS == 0:

                optimizerG.zero_grad()

                segmentations = generator(images)
                #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
                generated_segmentations = merge_images_with_masks(images, segmentations).to(DEVICE)
                #generated_segmentations = images * generator(images)

                loss_G = -torch.mean(discriminator(generated_segmentations))

                loss_G.backward()
                optimizerG.step()

                # Logemos en tensorboard coste por mini-batch
                writer.add_scalar("Train loss/generator", loss_G.item(), epoch * len(trainloader) + i)
                writer.add_scalar("Train loss/discriminator", loss_D.item(), epoch * len(trainloader) + i)

                if i % 5 == 0:
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, N_EPOCHS, i % len(trainloader), len(trainloader), loss_D.item(), loss_G.item())
                    )

                # Encolamos los costes para calcular el medio de la epoch cuando esta termine
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

            else:
                if i % 5 == 0:
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: -]"
                    % (epoch, N_EPOCHS, i % len(trainloader), len(trainloader), loss_D.item())
                    )

                writer.add_scalar("Train loss/discriminator", loss_D.item(), epoch * len(trainloader) + i)
                D_losses.append(loss_D.item())


        # Logemos en tensorboard coste por epoch
        if len(G_losses) > 0:
            writer.add_scalar("Per epoch train loss/generator", np.mean(np.array(G_losses)), epoch)
        writer.add_scalar("Per epoch train loss/discriminator", np.mean(np.array(D_losses)), epoch)

        # Vaciamos las listar para comenzar la siguiente epoch
        G_losses = []
        D_losses = []

        # Guardamos los pesos de la red
        torch.save(generator.state_dict(), os.path.join(WEIHGHTS_PATH, "generator_weights"))
        torch.save(discriminator.state_dict(), os.path.join(WEIHGHTS_PATH, "discriminator_weights"))

        if epoch % 5 == 0:
            with torch.no_grad():
                while_training_inference(valloader, generator, epoch, writer)


def weights_init(m):

    """
    Inicializacion de los pesos de la red

    :param m: red
    :return:
    """
    classname = m.__class__.__name__

    if classname == 'DilationConvolutionsChain':
        pass
    else:
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def load_img_transforms():

    """
    Funcion que carga las transformaciones

    :return:
    """
    train_data_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor()
    ])

    val_data_transform = train_data_transform

    return train_data_transform, val_data_transform


def merge_images_with_masks(images, masks):

    """
    Genera las imagenes de 4 canales que se pasan al discriminador (3 de la imagen original + 1 con
    la mascara de segmentacion

    :param images: imagenes
    :param masks: mascaras de segmentacion
    :return: tensor con imagenes de 4 canales
    """

    batch_size = images.shape[0]
    img_dim = images.shape[2]
    merged = torch.rand(batch_size, 4, img_dim, img_dim)

    for i in range(batch_size):
        merged[i] = torch.cat((images[i], masks[i]))

    return merged

def gradient_penalty(critic, real_data, fake_data, penalty, device):

    n_elements = real_data.nelement()
    batch_size = real_data.size()[0]
    colors = real_data.size()[1]
    image_width = real_data.size()[2]
    image_height = real_data.size()[3]
    alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
    alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

    fake_data = fake_data.view(batch_size, colors, image_width, image_height)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty

    return gradient_penalty

def test():

    writer = SummaryWriter()

    generator = NetS(1).to(DEVICE)
    #discriminator = Discriminator().to(DEVICE)

    # Carga las transformaciones a hacer a las imagenes
    train_data_transform, val_data_transform = load_img_transforms()

    # Carga los cargadores de imagenes de train y val
    if DATA_AUG:
        augmentation_pipeline = AugmentationPipeline()
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=augmentation_pipeline)
    else:
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=None)

    optimizerG = optim.SGD(generator.parameters(), lr=0.005)

    generator.train()

    for p in generator.parameters():
        p.requires_grad = True

    for epoch in range(50):

        for i, data in enumerate(trainloader):

            optimizerG.zero_grad()
            generator.zero_grad()
            generator.train()

            images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)
            batch_size = images.shape[0]

            segmentations = generator(images)
            segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
            loss_G = -torch.mean(torch.sum(segmentations.view(batch_size, -1), axis=1))

            loss_G.backward()
            optimizerG.step()

            # Logemos en tensorboard coste por mini-batch
            writer.add_scalar("Train loss/generator", loss_G.item(), epoch * len(trainloader) + i)


def inference():

    # Carga las transformaciones a hacer a las imagenes
    train_data_transform, val_data_transform = load_img_transforms()

    # Carga los cargadores de imagenes de train y val
    if DATA_AUG:
        augmentation_pipeline = AugmentationPipeline()
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=augmentation_pipeline)
    else:
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=None)

    generator = RDAU_NET().to(DEVICE)
    #generator = NetS(1).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    generator.load_state_dict(torch.load(os.path.join(WEIHGHTS_PATH, "generator_weights")))
    #generator.load_state_dict(torch.load(os.path.join(WEIHGHTS_PATH, "unet")))
    discriminator.load_state_dict(torch.load(os.path.join(WEIHGHTS_PATH, "discriminator_weights")))

    accuracies = []
    recalls = []
    specifities = []
    precisions = []
    f1_scores = []
    dice_coeffs = []
    intersection_over_unions = []

    with torch.no_grad():

        for i, data in enumerate(valloader):

            images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

            segmentations = generator(images)
            segmentations = torch.autograd.Variable((segmentations > 0.5).float())

            trans = transforms.ToPILImage()

            for j in range(images.shape[0]):

                image, gt = images[j], ground_truth[j]

                name = data[2][j].split('/')[-1]

                accuracy, recall, specifity, precision, f1_score = get_evaluation_metrics(segmentations[j].detach().to("cpu").numpy(),
                                                                                          gt.detach().to("cpu").numpy())

                if accuracy != np.nan:
                    accuracies.append(accuracy)

                if recall != np.nan:
                    recalls.append(recall)

                if specifity != np.nan:
                    specifities.append(specifity)

                if precision != np.nan:
                    precisions.append(precision)

                if f1_score != np.nan:
                    f1_scores.append(f1_score)

                dice_coefficient = dice_coeff(segmentations[j].detach().to("cpu").numpy(), gt.detach().to("cpu").numpy())
                dice_coeffs.append(dice_coefficient)

                _intersection_over_union = intersection_over_union(segmentations[j].detach().to("cpu").numpy(),
                                                               gt.detach().to("cpu").numpy())
                intersection_over_unions.append(_intersection_over_union)

                image = trans(image)
                gt = trans(gt)
                segmentation = trans(segmentations[j])

                opencv_image = np.array(image)
                opencv_image = opencv_image[:, :, ::-1].copy()
                opencv_gt = np.array(gt)
                opencv_segmentation = np.array(segmentation)

                if not COLOR:
                    img = np.vstack((cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY), opencv_gt, opencv_segmentation))

                    cv2.imwrite(os.path.join("inf", f"{name}.png"), img)
                else:
                    contours_gt, hierarchy = cv2.findContours(opencv_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_seg, hierarchy = cv2.findContours(opencv_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    cv2.drawContours(opencv_image, contours_gt, -1, (0, 255, 0), 1)
                    cv2.drawContours(opencv_image, contours_seg, -1, (0, 0, 255), 1)

                    cv2.imwrite(f"inf/{name}.png", opencv_image)


    mean_accuracy = np.nansum(np.array(accuracies)) / len(accuracies)
    std_accuracy = np.std(np.array(accuracies))

    mean_recall = np.nansum(np.array(recalls)) / len(recalls)
    std_recall = np.std(np.array(recalls))

    mean_specifity = np.nansum(np.array(specifities)) / len(specifities)
    std_specifity = np.std(np.array(specifities))

    precisions = np.array(precisions)[~np.isnan(np.array(precisions))]
    mean_precision = np.nansum(np.array(precisions)) / len(precisions)
    std_precision = np.std(np.array(precisions))

    f1_scores = np.array(f1_scores)[~np.isnan(f1_scores)]
    mean_f1_score = np.nansum(np.array(f1_scores)) / len(f1_scores)
    std_f1_score = np.std(np.array(f1_scores))

    mean_dice_coeff = np.nansum(np.array(dice_coeffs)) / len(dice_coeffs)
    std_dice_coeff = np.std(np.array(dice_coeffs))

    mean_iou = np.nansum(np.array(intersection_over_unions)) / len(intersection_over_unions)
    std_iou = np.std(np.array(intersection_over_unions))

    print("\nAccuracy: {:.4f} +- {:.4f}\nRecall: {:.4f} +- {:.4f}\nSpecifity: {:.4f} +- {:.4f}\nPrecision: {:.4f} +- {:.4f}\nF1_score: {:.4f} +- {:.4f}\nIntersection over union: {:.4f} +- {:.4f}\nDice coeff: {:.4f} +- {:.4f}".format
          (mean_accuracy, std_accuracy, mean_recall, std_recall, mean_specifity, std_specifity, mean_precision, std_precision, mean_f1_score, std_f1_score, mean_iou, std_iou, mean_dice_coeff, std_dice_coeff))



def while_training_inference(valloader, generator, epoch, writer):

    folder = os.path.join("train_inference", f"epoch_{epoch}")

    if not os.path.isdir(folder):
        os.mkdir(folder)

    accuracies = []
    recalls = []
    specifities = []
    precisions = []
    f1_scores = []
    dice_coeffs = []
    intersection_over_unions = []

    for i, data in enumerate(valloader):
        images, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

        segmentations = generator(images)
        segmentations = (segmentations > 0.5).float()
        trans = transforms.ToPILImage()

        for j in range(images.shape[0]):

            image, gt = images[j], ground_truth[j]

            name = data[2][j].split('/')[-1]

            accuracy, recall, specifity, precision, f1_score = get_evaluation_metrics(segmentations[j].detach().to("cpu").numpy(),
                                                                                          gt.detach().to("cpu").numpy())

            if accuracy != np.nan:
                accuracies.append(accuracy)

            if recall != np.nan:
                recalls.append(recall)

            if specifity != np.nan:
                specifities.append(specifity)

            if precision != np.nan:
                precisions.append(precision)

            if f1_score != np.nan:
                f1_scores.append(f1_score)

            dice_coefficient = dice_coeff(segmentations[j].detach().to("cpu").numpy(), gt.detach().to("cpu").numpy())
            dice_coeffs.append(dice_coefficient)

            _intersection_over_union = intersection_over_union(segmentations[j].detach().to("cpu").numpy(),
                                                               gt.detach().to("cpu").numpy())
            intersection_over_unions.append(_intersection_over_union)

            image = trans(image)
            gt = trans(gt)
            segmentation = trans(segmentations[j])

            opencv_image = np.array(image)
            opencv_image = opencv_image[:, :, ::-1].copy()
            opencv_gt = np.array(gt)
            opencv_segmentation = np.array(segmentation)

            if not COLOR:
                img = np.vstack((cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY), opencv_gt, opencv_segmentation))
                cv2.imwrite(os.path.join(folder, f"{name}.png"), img)

            else:
                contours_gt, hierarchy = cv2.findContours(opencv_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_seg, hierarchy = cv2.findContours(opencv_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cv2.drawContours(opencv_image, contours_gt, -1, (0, 255, 0), 1)
                cv2.drawContours(opencv_image, contours_seg, -1, (0, 0, 255), 1)

                cv2.imwrite(os.path.join(folder, f"{name}.png"), opencv_image)

    mean_accuracy = np.nansum(np.array(accuracies)) / len(accuracies)
    std_accuracy = np.std(np.array(accuracies))

    mean_recall = np.nansum(np.array(recalls)) / len(recalls)
    std_recall = np.std(np.array(recalls))

    mean_specifity = np.nansum(np.array(specifities)) / len(specifities)
    std_specifity = np.std(np.array(specifities))

    precisions = np.array(precisions)[~np.isnan(np.array(precisions))]
    mean_precision = np.nansum(np.array(precisions)) / len(precisions)
    std_precision = np.std(np.array(precisions))

    f1_scores = np.array(f1_scores)[~np.isnan(f1_scores)]
    mean_f1_score = np.nansum(np.array(f1_scores)) / len(f1_scores)
    std_f1_score = np.std(np.array(f1_scores))

    mean_dice_coeff = np.nansum(np.array(dice_coeffs)) / len(dice_coeffs)
    std_dice_coeff = np.std(np.array(dice_coeffs))

    mean_iou = np.nansum(np.array(intersection_over_unions)) / len(intersection_over_unions)
    std_iou = np.std(np.array(intersection_over_unions))

    writer.add_scalar("Metrics/accuracy", mean_accuracy, epoch)
    writer.add_scalar("Metrics/recall", mean_recall, epoch)
    writer.add_scalar("Metrics/specifity", mean_specifity, epoch)
    writer.add_scalar("Metrics/precision", mean_precision, epoch)
    writer.add_scalar("Metrics/f1 score", mean_f1_score, epoch)
    writer.add_scalar("Metrics/intersection over union", mean_iou, epoch)
    writer.add_scalar("Metrics/dice coeff", mean_dice_coeff, epoch)


def graph():

    train_data_transform, val_data_transform = load_img_transforms()

    # Carga los cargadores de imagenes de train y val
    if DATA_AUG:
        augmentation_pipeline = AugmentationPipeline()
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=augmentation_pipeline)
    else:
        trainloader, valloader = load_data(train_data_transform,
                                           val_data_transform, print_batches_info=True,
                                           augmentation_pipeline=None)

    generator = RDAU_NET().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)


    writer = SummaryWriter()

    for i, data in enumerate(valloader):
        images, ground_truth = data[0].to, data[1]
        break

    writer.add_graph(generator, images)
    writer.close()



def test2():

    generator = NetS(1).to(DEVICE)
    #print(generator)

    data = torch.rand([32, 3, 128, 128]).to(DEVICE)

    output = generator(data)

    print(output.shape)


if __name__ == '__main__':
    #unet_train()
    #wgan()
    inference()
    #test()
    #graph()
