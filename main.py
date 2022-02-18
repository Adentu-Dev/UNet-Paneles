from torch.utils.data import Dataset, DataLoader
from skimage.color import label2rgb
from skimage.util import random_noise

import torchvision.transforms.functional as trans
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import torch
import fnmatch
import cv2
import os


'''
                UNet
'''


class Unet(nn.Module):

    def __init__(self, n, batch_size=50):

        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*n, out_channels=2*n, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*n, out_channels=4*n, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*n, out_channels=8*n, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=8*n, out_channels=16*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16*n, out_channels=16*n, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(in_channels=16*n, out_channels=8*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*n, out_channels=8*n, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels=8*n, out_channels=4*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*n, out_channels=4*n, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(in_channels=4*n, out_channels=2*n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*n, out_channels=2*n, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.conv_9 = nn.Sequential(
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.conv_A = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=1, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )

        self.up_6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16*n, out_channels=8*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8*n, out_channels=4*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=4*n, out_channels=2*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=2, padding='same'),
            nn.ReLU()
        )


    def forward(self, input_):
        conv_1 = self.conv_1(input_)
        pool_1 = nn.MaxPool2d(kernel_size=(2,2))(conv_1)

        conv_2 = self.conv_2(pool_1)
        pool_2 = nn.MaxPool2d(kernel_size=(2,2))(conv_2)

        conv_3 = self.conv_3(pool_2)
        pool_3 = nn.MaxPool2d(kernel_size=(2,2))(conv_3)

        conv_4 = self.conv_4(pool_3)
        pool_4 = nn.MaxPool2d(kernel_size=(2,2))(conv_4)

        conv_5 = self.conv_5(pool_4)

        up_6    = self.up_6(conv_5)
        merge_6 = torch.cat([conv_4, up_6], dim=1)
        conv_6  = self.conv_6(merge_6)

        up_7    = self.up_7(conv_6)
        merge_7 = torch.cat([conv_3, up_7], dim=1)
        conv_7  = self.conv_7(merge_7)

        up_8    = self.up_8(conv_7)
        merge_8 = torch.cat([conv_2, up_8], dim=1)
        conv_8  = self.conv_8(merge_8)

        up_9    = self.up_9(conv_8)
        merge_9 = torch.cat([conv_1, up_9], dim=1)
        conv_9  = self.conv_9(merge_9)

        return self.conv_A(conv_9)

class UnetDataset(Dataset):

    def __init__(self, img_x, img_y):
        super().__init__()
        self.img_x = img_x
        self.img_y = img_y

    def __len__(self):
        return len(self.img_x)

    def __getitem__(self, item):
        return torch.from_numpy(self.img_x[item]), torch.from_numpy(self.img_y[item])


'''
                Methods
'''


def train_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        img, seg = batch
        img = img.to(device)
        seg = seg.to(device)

        predictions = model(img)

        loss = criterion(predictions, seg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(iterator)


def eval_epoch(model, iterator, criterion, device):
    #initialize every epoch
    epoch_loss = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:

            img, seg = batch
            img = img.to(device)
            seg = seg.to(device)

            #convert to 1d tensor
            predictions = model(img)

            #compute loss
            loss = criterion(predictions, seg)

            #keep track of loss
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_history(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Lectura de imágenes a color y segmentación para definir conjuntos
# de training, validación y testing
def load_images(fpath,
                img_names,
                i1, i2, st,
                transformations,
                nl=32, nc=32, echo='off'):

    print(f'reading {st} ...')
    n = i2 - i1

    imgs_data = np.empty(
        (n*(len(transformations) + 1), nl, nc), dtype='uint8'
        ) if transformations else np.empty((n, nl, nc), dtype='uint8'
    )
    mask_data = np.empty(
        (n*(len(transformations) + 1), nl, nc), dtype='uint8'
        ) if transformations else np.empty((n, nl, nc), dtype='uint8'
    )

    for img_indx in range(n):

        i = i1 + img_indx

        img_path = f'{fpath}color/{img_names[i]}'
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (nl, nc))

        mask_path = f'{fpath}seg/{img_names[i]}'
        mask_path = mask_path[:-3] + 'png'
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (nl, nc))

        indx = img_indx*(len(transformations)+1)

        imgs_data[indx] = img
        mask_data[indx] = mask


        if transformations:
            for trans_indx, transformation in enumerate(transformations):
                '''
                trans.hflip,
                trans.vflip,
                random_noise,
                trans.rotate
                '''

                if transformation.__name__ == 'rotate':
                    imgs_data[indx+trans_indx+1] =  transformation(torch.from_numpy(img),
                                                                   180
                                                    )[:, :, :, 0]
                    mask_data[indx+trans_indx+1] =  transformation(torch.from_numpy(mask),
                                                                   180
                                                    )[:, :, :, 0]

                elif transformation.__name__ == 'random_noise':
                    imgs_data[indx+trans_indx+1] = transformation(img,
                                                        mode='s&p',
                                                        seed=1,
                                                        clip=True,
                                                        )

                    mask_data[indx+trans_indx+1] = mask

                else:
                    imgs_data[indx+trans_indx+1] = transformation(torch.from_numpy(img))
                    mask_data[indx+trans_indx+1] = transformation(torch.from_numpy(mask))


        if echo == 'on':
            print('\nindx', indx)
            print('processing files ' + img_path)
            print('processing files ' + mask_path)

    imgs_data_norm = np.float32(imgs_data)/255.0
    mask_data_norm = np.float32(mask_data)/255.0
    imgs_data_norm = imgs_data_norm.reshape(
                              n*(len(transformations)+1), 1, nl, nc)
    mask_data_norm = mask_data_norm.reshape(
                              n*(len(transformations)+1), 1, nl, nc)

    return imgs_data_norm, mask_data_norm

def regionview(img, mask):
    img_color = np.dstack((img, img, img))
    img_color = label2rgb(mask, image=img_color, bg_label=0)
    return img_color

def dirfiles(img_path, img_ext):
    return fnmatch.filter(sorted(os.listdir(img_path)), img_ext)

def num2fixstr(x,d):
  # example num2fixstr(2,5) returns '00002'
  # example num2fixstr(19,3) returns '019'
  st = '%0*d' % (d,x)
  return st


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Device:', device)


    '''
                    Definitions
    '''

    # UNet size
    n           = 64 # size of the Unet

    # Training batch size
    batch_size  = 1

    # learning rate
    lr          = 1e-4

    # Resize parameters of layers
    # nl, nc      = 1024, 1024
    nl, nc      = 512, 512

    model       = Unet(n).to(device)
    criterion   = nn.MSELoss().to(device)
    optimizer   = optim.Adam(model.parameters(), lr=lr)



    '''
                    Training/Validation/Testing
    '''


    transformations = [
        trans.hflip,
        trans.vflip,
        random_noise,
        trans.rotate
    ][:-1]

    fpath = 'PanelesRGB/'

    img_names = dirfiles(fpath + 'color/', '*.jpg')

    # Split: Train/Val/Testing
    ax, ay = load_images(fpath, img_names,
                    0, 30, 'train',
                    transformations,
                    nl=nl, nc=nc, echo='off')
    print('number of train images: ', ax.shape[0], end='\n\n')

    vx, vy = load_images(fpath, img_names,
                    30, 35, 'val',
                    transformations,
                    nl=nl, nc=nc, echo='off')
    print('number of val images: ', vx.shape[0], end='\n\n')

    qx, qy = load_images(fpath, img_names,
                    35, 40, 'test',
                    transformations,
                    nl=nl, nc=nc, echo='off')
    print('number of test images: ', qx.shape[0], end='\n\n')


    train_dataset    = UnetDataset(ax, ay)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size)


    val_dataset      = UnetDataset(vx, vy)
    val_dataloader   = DataLoader(dataset=val_dataset,
                                  batch_size=batch_size)


    '''
                    Entrenamiento
    '''

    num_epochs      = 30

    train_loss_hist = []
    val_loss_hist   = []

    for epoch in range(num_epochs):

        if epoch % 5 == 0:
            lr /= 10
            print('Actualizando lr: ', lr)
            optimizer = optim.Adam(model.parameters(), lr=lr)


        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_loss_hist.append(train_loss)

        st = f'Training Loss: {train_loss:.4f}'
        val_loss = eval_epoch(model, val_dataloader, criterion, device)
        val_loss_hist.append(val_loss)
        sv = f'Validation Loss: {val_loss:.4f}'
        print('['+num2fixstr(epoch+1, 3)+'/'+num2fixstr(num_epochs, 3)+'] ' + st + ' | '+sv)


    torch.save(model.state_dict(), 'Tracker_Segmentation.pt')


    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label=' Val Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
