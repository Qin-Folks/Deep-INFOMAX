#-----------------------------------------------------------------------------
# model definition
#-----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class InfoMaxEncoder(torch.nn.Module):
    def __init__(self, input_sz, stt_ch=4, layer_num='deduction'):
        super().__init__()

        if layer_num == 'deduction':
            the_layer_num = math.log2(input_sz)
            assert the_layer_num.is_integer()
            the_layer_num = int(the_layer_num)
        else:
            the_layer_num = layer_num

        self.the_M = nn.Sequential()
        self.the_Y = nn.Sequential()

        tmp = 1

        for i in range(0, the_layer_num-tmp):
            if i == 0:
                self.the_M.add_module(
                    name="MConv{:d}".format(i),
                    module=nn.Conv2d(input_sz, stt_ch * (2 ** i), 4, 2, 1))
                self.the_M.add_module(name="MBN{:d}".format(i),
                                    module=nn.BatchNorm2d(stt_ch * (2 ** i), 1.e-3))
                self.the_M.add_module(name="MAct{:d}".format(i), module=self.relu)
            else:
                self.the_M.add_module(
                    name="MConv{:d}".format(i),
                    module=nn.Conv2d(input_sz * (2 ** (i - 1)), stt_ch * (2 ** i), 4, 2, 1))
                self.the_M.add_module(name="MBN{:d}".format(i),
                                    module=nn.BatchNorm2d(stt_ch * (2 ** i), 1.e-3))
                self.the_M.add_module(name="MAct{:d}".format(i), module=self.relu)

        for i in range(the_layer_num-tmp, the_layer_num):
            self.the_Y.add_module(
                name="YConv{:d}".format(i),
                module=nn.Conv2d(input_sz * (2 ** (i - 1)), stt_ch * (2 ** i), 4, 2, 1))
            self.the_Y.add_module(name="YBN{:d}".format(i),
                                  module=nn.BatchNorm2d(stt_ch * (2 ** i), 1.e-3))
            self.the_Y.add_module(name="YAct{:d}".format(i), module=self.relu)

    def forward(self, x):
        mapM = self.the_M(x)
        mapY = self.the_Y(mapM)
        return mapY, mapM


class GlobalDiscriminator(torch.nn.Module):
    r"""
    input of GlobalDiscriminator is the `M` in Encoder.forward, so with
    channels : num_feature * 2, in_channels
    shape    : (input_shape[0]-3*2, input_shape[1]-3*2), M_shape
    """
    def __init__(self, M_channels, M_shape, E_size, interm_size=512):
        super().__init__()

        in_channels = M_channels; out_channels = in_channels // 2
        self.c0 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)
        in_channels = out_channels; out_channels = in_channels // 2
        self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # input of self.l0 is the concatenate of E and flattened output of self.c1 (C)
        in_feature = out_channels * (M_shape[0]-2*2) * (M_shape[1]-2*2) + E_size
        self.l0 = torch.nn.Linear(in_feature, interm_size)
        self.l1 = torch.nn.Linear(interm_size, interm_size)
        self.l2 = torch.nn.Linear(interm_size, 1)

    def forward(self, E, M):

        C = F.relu(self.c0(M))
        C = self.c1(C)
        C = C.view(E.shape[0], -1)
        out = torch.cat((E, C), dim=1)
        out = F.relu(self.l0(out))
        out = F.relu(self.l1(out))
        out = self.l2(out)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # output of Table 5
        return out


class LocalDiscriminator(torch.nn.Module):
    r"""
    the local discriminator with architecture described in
    Figure 4 and Table 6 in appendix 1A of https://arxiv.org/pdf/1808.06670.pdf.
    input is the concatenate of
    "replicated feature vector E (with M_shape now)" + "M"

    replicated means that all pixels are the same, they are just copies.
    """
    def __init__(self, M_channels, E_size, interm_channels=512):
        super().__init__()

        in_channels = E_size + M_channels
        self.c0 = torch.nn.Conv2d(in_channels, interm_channels, kernel_size=1)
        self.c1 = torch.nn.Conv2d(interm_channels, interm_channels, kernel_size=1)
        self.c2 = torch.nn.Conv2d(interm_channels, 1, kernel_size=1)

    def forward(self, x):

        score = F.relu(self.c0(x))
        score = F.relu(self.c1(score))
        score = self.c2(score)

        return score

class PriorDiscriminator(torch.nn.Module):
    r"""
    the Prior discriminator with architecture described in
    Figure 6 and Table 9 in appendix 1A of https://arxiv.org/pdf/1808.06670.pdf.

    input will be Real feature vector E and Fake feature vector E_fake (E_like shape),
    This discriminator is trained to distinguish Real and Fake inputs.
    So the Encoder is trained to "fool" this discriminator. (idea of GAN)
    """
    def __init__(self, E_size, interm_size=(1000,200)):
        super().__init__()
        assert isinstance(interm_size, tuple), "tuple of integers."

        self.l0 = torch.nn.Linear(E_size, interm_size[0])
        self.l1 = torch.nn.Linear(interm_size[0], interm_size[1])
        self.l2 = torch.nn.Linear(interm_size[1], 1)

    def forward(self, x):

        score = F.relu(self.l0(x))
        score = F.relu(self.l1(score))
        score = torch.sigmoid(self.l2(score))

        return score

class DeepInfoMaxLoss(torch.nn.Module):

    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.get_models()

    def get_models(self, input_shape=(32,32), num_feature=64, out_size=64, interm_size_G=512, interm_channels_L=512, interm_size_P=(1000,200) ):

        self.encoder = InfoMaxEncoder(input_sz=64)
        self.global_D = GlobalDiscriminator(M_channels=self.encoder.M_channels, M_shape=self.encoder.M_shape, E_size=out_size, interm_size=interm_size_G)
        self.local_D = LocalDiscriminator(M_channels=self.encoder.M_channels, E_size=out_size, interm_channels=interm_channels_L)
        self.prior_D = PriorDiscriminator(E_size=out_size, interm_size=interm_size_P)

    def forward(self, Y, M, M_fake):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        Y_replicated = Y.unsqueeze(-1).unsqueeze(-1)
        Y_replicated = Y_replicated.expand(-1, -1, 26, 26)

        Y_cat_M = torch.cat((M, Y_replicated), dim=1)
        Y_cat_M_fake = torch.cat((M_fake, Y_replicated), dim=1)

        # local loss
        # 2nd term in equation (8) in https://arxiv.org/pdf/1808.06670.pdf
        Ej = -F.softplus(-self.local_D(Y_cat_M)).mean()
        Em = -F.softplus(-self.local_D(Y_cat_M_fake)).mean()
        local_loss = (Em - Ej) * self.beta

        # global loss
        # 1st term in equation (8) in https://arxiv.org/pdf/1808.06670.pdf
        Ej = -F.softplus(-self.global_D(Y, M)).mean()
        Em = -F.softplus(-self.global_D(Y, M_fake)).mean()
        global_loss= (Em - Ej) * self.alpha

        # prior loss
        # 3rd term in equation (8) in https://arxiv.org/pdf/1808.06670.pdf
        prior = torch.rand_like(Y)
        # 1st term in equation (7) in https://arxiv.org/pdf/1808.06670.pdf
        term_a = torch.log(self.prior_D(prior)).mean()
        # 2nd term in equation (7) in https://arxiv.org/pdf/1808.06670.pdf
        term_b = torch.log(1 - self.prior_D(Y)).mean()
        prior_loss = - (term_a + term_b) * self.gamma

        return local_loss + global_loss + prior_loss
