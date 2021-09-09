"""
This python file contains the Autoencoder models as classes
per model. Architectures include linear, convolution, transpose
convolution, upampling, and ResNet type of NN/layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Upsampling_model(nn.Module):
    ### NOTE: after extensive testing, we decided that upsampling does not produce
    ###     accurate images nor does it learn as fast. This model is no longer used.

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8, stride=2, kernel_size=4):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(Upsampling_model, self).__init__()
        self.img_width = self.img_height = img_dim
        #self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(phy_dim,
                      128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16 * 8 * 8, 16 * 12 * 12, bias=False),
            nn.BatchNorm1d(16*12*12),
            nn.ReLU(),
            nn.Dropout(dropout),

            #Last linear layer
            nn.Linear(16 * 12 * 12, 16 * 16 * 16, bias=False),
            nn.ReLU()
        )
        self.dec_transconv = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(16, 16, kernel_size, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(8, 4, kernel_size, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),

            ###output layer
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid()
        )

    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(phy)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='nearest')
        return z


class Dev_Forward_AE(nn.Module):

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8, stride=2, kernel_size=4):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(Dev_Forward_AE, self).__init__()
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim

        # Linear layers
        self.dec_linear = nn.Sequential(
            nn.Linear(phy_dim,
                      128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16 * 8 * 8, 16 * 16 * 8, bias=False),
            nn.BatchNorm1d(16*16*8),
            nn.ReLU(),
            nn.Dropout(dropout),

            #Last linear layer
            nn.Linear(16 * 16 * 8, 16 * 16 * 16, bias=False),
            nn.ReLU()
        )

        #convolutional layers
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16,  kernel_size, stride=stride, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(16, 16, kernel_size, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),

            nn.Conv2d(16, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size, stride=stride, bias=False,
                               output_padding=1, padding=0),

            nn.Conv2d(8, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Conv2d(8, 4, kernel_size, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),

            ###output layer
            nn.ConvTranspose2d(4, 4, kernel_size, stride=stride, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid()
        )

    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(phy)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='nearest')
        return z



class Conv_Forward_AE(nn.Module):

    def __init__(self, img_dim=187, dropout=.2, in_ch=1, phy_dim=8,
        stride=2, kernel_size=4, numb_conv=5, numb_lin=5, a_func=nn.ReLU()):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        stride     : int
            stride step size (must be >1)
        kernel_size: int
            kernel size for the convolutional layers
        numb_conv  : int
            number of convolutional layers in the model. 5 by defalult.
            Works best for <=8 layers, anything higher might result in errors.
            NOTE: Does not support 0 convolutional layers as it will always have
            one minimum, by construction.
        numb_lin   : int
            number of linear layers in the model. 5 by defalult.
        """

        super(Conv_Forward_AE, self).__init__()
        self.in_ch = in_ch
        self.img_dim = img_dim

        # Linear layers
        h_ch = 2
        if (numb_lin > 4):
            in_ch = 8
        else:
            in_ch = 16
        self.lin = nn.Sequential(
        nn.Linear(phy_dim,  in_ch * h_ch * h_ch, bias=False),
        nn.BatchNorm1d(in_ch * h_ch * h_ch),
        a_func,
        nn.Dropout(dropout)
        )
        i_ch = h_ch
        h_ch *= 2

        for i in range(numb_lin - 1):
            self.lin.add_module(
            "linear_%i" % (i+1),
            nn.Linear(in_ch * i_ch * i_ch, in_ch * h_ch * h_ch, bias=False)
            )

            if (i != numb_lin - 2):

                self.lin.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm1d(in_ch * h_ch * h_ch)
                )

                self.lin.add_module(
                "activation_%i" % (i + 1),
                a_func
                )

                self.lin.add_module(
                "Dropout_%i" % (i + 2),
                nn.Dropout(dropout)
                )
                i_ch = h_ch
                if (i >= 2 and numb_lin > 4):
                    h_ch += 4
                else:
                    h_ch *= 2

            else:
                self.lin.add_module(
                "activation_output",
                a_func
                )

        self.h_ch = h_ch
        self.i_ch = in_ch

        #Convolutional layers
        self.conv = nn.Sequential()
        i_ch = in_ch

        for i in range(numb_conv - 1):
            if (i%2 == 0):
                self.conv.add_module(
                "ConvTranspose2d_%i" % (i+1),
                nn.ConvTranspose2d(i_ch, i_ch, kernel_size, stride=stride, bias=False,
                                    output_padding=1, padding=0)
                )
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, i_ch, kernel_size, bias=False)
                )
                self.conv.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm2d(i_ch, momentum=0.005)
                )
                self.conv.add_module(
                "activation_%i" % (i + 1),
                a_func
                )
                #i_ch = o_ch
                o_ch = i_ch//2

            else:
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, o_ch, kernel_size, bias=False),
                )
                self.conv.add_module(
                    "bn_%i" % (i + 1), nn.BatchNorm2d(o_ch, momentum=0.005)
                )
                self.conv.add_module("activation_%i" % (i + 1), a_func)
                i_ch = o_ch
                #o_ch = o_ch//2

        #output layers
        self.conv.add_module(
        "ConvTransposed2d_output",
        nn.ConvTranspose2d(i_ch, 4, kernel_size, stride=stride, bias=False,
                            output_padding=1, padding=0))

        self.conv.add_module(
        "Conv2d_output",
        nn.Conv2d(4, self.in_ch, kernel_size)
        )

        self.conv.add_module(
        "Sigmoid",
        nn.Sigmoid()
        )


    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.lin(phy)
        z = z.view(-1, self.i_ch, self.h_ch, self.h_ch)
        z = self.conv(z)

        z = F.interpolate(z, size=(self.img_dim, self.img_dim),
                          mode='nearest')
        return z

class Conv_Forward_2(nn.Module):

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8,
        stride=2, kernel_size=4, numb_conv=5, numb_lin=5, a_func=nn.ReLU()):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        stride     : int
            stride step size (must be >1)
        kernel_size: int
            kernel size for the convolutional layers
        numb_conv  : int
            number of convolutional layers in the model. 5 by defalult.
            Works best for <=8 layers, anything higher might result in errors.
            NOTE: Does not support 0 convolutional layers as it will always have
            one minimum, by construction.
        numb_lin   : int
            number of linear layers in the model. 5 by defalult.
        """

        super(Conv_Forward_2, self).__init__()
        self.in_ch = in_ch
        self.img_dim = img_dim

        # Linear layers
        h_ch = 2
        self.lin = nn.Sequential(
        nn.Linear(phy_dim,  16 * h_ch * h_ch, bias=False),
        nn.BatchNorm1d(16 * h_ch * h_ch),
        a_func,
        nn.Dropout(dropout)
        )
        i_ch = h_ch
        h_ch *= 2

        for i in range(numb_lin - 1):
            self.lin.add_module(
            "linear_%i" % (i+1),
            nn.Linear(16 * i_ch * i_ch, 16 * h_ch * h_ch, bias=False)
            )

            if (i != numb_lin - 2):

                self.lin.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm1d(16 * h_ch * h_ch)
                )

                self.lin.add_module(
                "activation_%i" % (i + 1),
                a_func
                )

                self.lin.add_module(
                "Dropout_%i" % (i + 2),
                nn.Dropout(dropout)
                )
                i_ch = h_ch
                if (numb_lin > 4 and i >= 3):
                    h_ch += 2
                else:
                    h_ch *= 2

            else:
                self.lin.add_module(
                "activation_output",
                a_func
                )

        self.h_ch = h_ch

        #Convolutional layers
        self.conv = nn.Sequential()
        i_ch = 16
        #o_ch = i_ch * (numb_conv - 1)

        for i in range(numb_conv - 1):
            if (i%2 == 0):
                self.conv.add_module(
                "ConvTranspose2d_%i" % (i+1),
                nn.ConvTranspose2d(i_ch, i_ch, kernel_size, stride=stride, bias=False,
                                    output_padding=1, padding=0)
                )
                self.conv.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm2d(i_ch, momentum=0.005)
                )
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, i_ch, kernel_size, bias=True)
                )
                self.conv.add_module(
                "activation_%i" % (i + 1),
                a_func
                )
                #i_ch = o_ch
                o_ch = i_ch//2

            else:
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, o_ch, kernel_size, bias=False),
                )
                self.conv.add_module(
                    "bn_%i" % (i + 1), nn.BatchNorm2d(o_ch, momentum=0.005)
                )
                self.conv.add_module("activation_%i" % (i + 1), a_func)
                i_ch = o_ch
                #o_ch = o_ch//2

        #output layers
        self.conv.add_module(
        "ConvTransposed2d_output",
        nn.ConvTranspose2d(i_ch, 4, kernel_size, stride=stride, bias=False,
                            output_padding=1, padding=0))

        self.conv.add_module(
        "Conv2d_output",
        nn.Conv2d(4, in_ch, kernel_size)
        )

        self.conv.add_module(
        "Sigmoid",
        nn.Sigmoid()
        )


    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.lin(phy)
        z = z.view(-1, 16, self.h_ch, self.h_ch)
        z = self.conv(z)

        z = F.interpolate(z, size=(self.img_dim, self.img_dim),
                          mode='nearest')
        return z

def conv_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k) / st + 1)


def pool_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k) / st + 1)

""" Latest AE model. """
class ConvLinTrans_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size,
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined,
    with a final linear layer to return the latent code.
    The decoder is build using transpose convolution and normal convolution layers.
    ...
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """

    def __init__(
        self,
        latent_dim=32,
        img_dim=28,
        dropout=0.2,
        in_ch=1,
        kernel=3,
        n_conv_blocks=5,
        phy_dim=0,
        feed_phy=True,
    ):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ConvLinTrans_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim
        self.feed_phy = feed_phy

        # Encoder specification
        self.enc_conv_blocks = nn.Sequential()
        h_ch = in_ch
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module(
                "conv2d_%i1" % (i + 1),
                nn.Conv2d(h_ch, h_ch * 2, kernel_size=kernel, bias=False),
            )
            self.enc_conv_blocks.add_module(
                "bn_%i1" % (i + 1), nn.BatchNorm2d(h_ch * 2, momentum=0.005)
            )
            self.enc_conv_blocks.add_module("relu_%i1" % (i + 1), nn.ReLU())
            self.enc_conv_blocks.add_module(
                "conv2d_%i2" % (i + 1),
                nn.Conv2d(h_ch * 2, h_ch * 2, kernel_size=kernel, bias=False),
            )
            self.enc_conv_blocks.add_module(
                "bn_%i2" % (i + 1), nn.BatchNorm2d(h_ch * 2, momentum=0.005)
            )
            self.enc_conv_blocks.add_module("relu_%i2" % (i + 1), nn.ReLU())
            self.enc_conv_blocks.add_module(
                "maxpool_%i" % (i + 1), nn.MaxPool2d(2, stride=2)
            )
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = pool_out(img_dim, 2, 2)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim ** 2 + phy_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim + (phy_dim if feed_phy else 0), 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 8 * 8, 16 * 16 * 16, bias=False),
            nn.ReLU(),
        )

        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(
                16, 16, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(16, 16, 4, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(16, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 8, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(8, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(8, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, 4, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, 4, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid(),
        )

    def encode(self, x, phy=None):
        """
        Encoder side of autoencoder.
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv_blocks(x)
        x = x.flatten(1)
        if self.phy_dim > 0 and phy is not None:
            x = torch.cat([x, phy], dim=1)
        x = self.enc_linear(x)
        return x

    def decode(self, z, phy=None):
        """
        Decoder side of autoencoder.
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        if self.phy_dim > 0 and self.feed_phy and phy is not None:
            z = torch.cat([z, phy], dim=1)
        z = self.dec_linear(z)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height), mode="nearest")
        return z

    def forward(self, x, phy=None):
        """
        Autoencoder forward pass.
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x, phy=phy)
        xhat = self.decode(z, phy=phy)
        return xhat, z
