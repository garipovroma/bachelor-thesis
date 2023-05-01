import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int, z_length: int):
        super(Generator, self).__init__()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.z_length = z_length

        # in (?, z_length, 1, 1)
        # out (?, data_size * 4, 4, 4)
        self.fc_z = nn.ConvTranspose2d(in_channels=self.z_length,
                                       out_channels=self.data_size * 4, kernel_size=4, stride=1, padding=0)
        # in (?, meta_length, 1, 1)
        # out (?, data_size * 4, 4, 4)
        self.fc_meta = nn.ConvTranspose2d(in_channels=self.meta_length,
                                          out_channels=self.data_size * 4, kernel_size=4, stride=1, padding=0)
        # in (?, data_size * 8, 4, 4)
        # out (?, data_size * 4, 8, 8)
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.data_size * 8,
                                          out_channels=self.data_size * 4, kernel_size=4, stride=2, padding=1)
        # out (?, data_size * 2, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.data_size * 4,
                                          out_channels=self.data_size * 2, kernel_size=4, stride=2, padding=1)
        # out (?, data_size, 32, 16)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.data_size * 2,
                                          out_channels=self.data_size, kernel_size=(4, 1), stride=(2, 1),
                                          padding=(1, 0))
        # out (?, data_size / 2, 64, 16)
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.data_size,
                                          out_channels=classes_size, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, z, meta):
        fc_z = F.leaky_relu(self.fc_z(z), 0.2)
        fc_meta = F.leaky_relu(self.fc_meta(meta), 0.2)

        fc = torch.cat((fc_z, fc_meta), 1)
        deconv1 = F.leaky_relu(self.deconv1(fc), 0.2)
        deconv2 = F.leaky_relu(self.deconv2(deconv1), 0.2)
        deconv3 = F.leaky_relu(self.deconv3(deconv2), 0.2)
        deconv4 = F.sigmoid(self.deconv4(deconv3))
        return deconv4


class Discriminator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int,
                 lambda_length: int):
        super(Discriminator, self).__init__()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.lambda_length = lambda_length

        # in (?, classes_size, instances_size, features_size)
        # out (?, data_size / 2, 32, 16)
        self.conv_1 = nn.Conv2d(in_channels=classes_size,
                                out_channels=self.data_size, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        # in (?, data_size, 32, 16)
        # out (?, data_size * 2, 16, 16)
        self.conv_2 = nn.Conv2d(in_channels=self.data_size,
                                out_channels=self.data_size * 2, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        # out (?, data_size * 4, 8, 8)
        self.conv_3 = nn.Conv2d(in_channels=self.data_size * 2,
                                out_channels=self.data_size * 4, kernel_size=4, stride=2, padding=1)
        # out (?, data_size * 8, 4, 4)
        self.conv_4 = nn.Conv2d(in_channels=self.data_size * 4,
                                out_channels=self.data_size * 8, kernel_size=4, stride=2, padding=1)
        # out (?, self.data_size * 16, 1, 1)
        self.conv_5 = nn.Conv2d(in_channels=self.data_size * 8,
                                out_channels=self.data_size * 16, kernel_size=4, stride=1, padding=0)

        self.fc = nn.Linear(in_features=self.data_size * 16 + self.meta_length, out_features=self.lambda_length + 1)

    def forward(self, data, meta):
        conv1 = F.leaky_relu(self.conv_1(data), 0.2)
        conv2 = F.leaky_relu(self.conv_2(conv1), 0.2)
        conv3 = F.leaky_relu(self.conv_3(conv2), 0.2)
        conv4 = F.leaky_relu(self.conv_4(conv3), 0.2)
        conv5 = F.leaky_relu(self.conv_5(conv4), 0.2)
        concat = torch.cat((conv5, meta), 1)
        result = self.fc(concat.squeeze())
        return result

class DeepSetFlattenedModel(torch.nn.Module):
    def __init__(self, hidden_size_0=1, hidden_size_1=1, predlast_hidden_size=1, meta_size = 27, out_classes=1):
        super().__init__()
        self.hidden_size_0 = hidden_size_0
        self.hidden_size_1 = hidden_size_1
        self.predlast_hidden_size = predlast_hidden_size
        self.meta_size = meta_size
        self.out_classes = out_classes
        self.inv_0 = deepsetlayers.InvLinear(hidden_size_0, hidden_size_1)
        self.equiv_0 = deepsetlayers.EquivLinear(1, hidden_size_0)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_1 + meta_size, 2 * predlast_hidden_size),
            torch.nn.BatchNorm1d(2 * predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.FIRST_DROPOUT_RATE),
            torch.nn.Linear(2 * predlast_hidden_size, predlast_hidden_size),
            torch.nn.BatchNorm1d(predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.SECOND_DROPOUT_RATE),
            torch.nn.Linear(predlast_hidden_size, out_classes),
        )

    def forward(self, X, y):
        # (batch_size, N, M)
        X = X.unsqueeze(-1)
        X = X.flatten(1, 2)
        # (batch_size, N * M, 1)
        X = self.equiv_0(X)
        # (batch_size, N * M, hidden_size_0)
        X = self.inv_0(X)
        # (batch_size, hidden_size_1)

        y = y.view(-1, self.meta_size)
        # (batch_size, meta_size)

        # vec0 = torch.abs(pos_vec - neg_vec)
        # vec1 = pos_vec + neg_vec
        x = torch.hstack([X, y])
        # x = torch.hstack([vec0, vec1, y])

        # (batch_size, hidden_size_1 + meta_size)
        x = self.classifier(x)
        # (batch_size, out_classes)
        return x

class CONFIG:
    # DATA
    features = 16
    instances = 64
    classes = 2

    # TRAIN
    num_epochs = 50
    train_batch_size = 1024
    learning_rate = 0.002
    criterion = 'torch.nn.BCEWithLogitsLoss'
    FIRST_DROPOUT_RATE = 0.7
    SECOND_DROPOUT_RATE = 0.5

    test_batch_size = 1024

    # MODEL
    hidden_size_0=64
    hidden_size_1=128
    predlast_hidden_size=256
    meta_size=27
    out_classes=3
    standardscaler=True
    meta_standardscaler=True
    transpose=True

    # device = torch.device("cuda:0" if torch.cuda.is_available() else if torch.backends.mps.is_available() then "mps" else "cpu")
    # device is cuda is available else mps is available else cpu
    device_mps_or_cpu = ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else device_mps_or_cpu)

    # OTHER
    seed = 42
    num_workers = 8


from deepsets import deepsetlayers
import torch.nn.functional as F


class DeepSetModelV5(torch.nn.Module):
    def __init__(self, hidden_size_0=1, hidden_size_1=1, predlast_hidden_size=1, meta_size=27, out_classes=1):
        super().__init__()
        self.hidden_size_0 = hidden_size_0
        self.hidden_size_1 = hidden_size_1
        self.predlast_hidden_size = predlast_hidden_size
        self.meta_size = meta_size
        self.out_classes = out_classes
        self.inv_0 = deepsetlayers.InvLinear(hidden_size_0, hidden_size_0)
        self.inv_1 = deepsetlayers.InvLinear(2 * hidden_size_1, 4 * hidden_size_1)
        self.equiv_0 = deepsetlayers.EquivLinear(1, hidden_size_0)
        self.equiv_1 = deepsetlayers.EquivLinear(hidden_size_0, 2 * hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(8 * hidden_size_1 + meta_size, 2 * predlast_hidden_size),
            torch.nn.BatchNorm1d(2 * predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.FIRST_DROPOUT_RATE),
            torch.nn.Linear(2 * predlast_hidden_size, predlast_hidden_size),
            torch.nn.BatchNorm1d(predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.SECOND_DROPOUT_RATE),
            torch.nn.Linear(predlast_hidden_size, out_classes),
        )

    def _forward(self, x):
        x = x.unsqueeze(-1)
        # (batch_size, N, M, 1)
        N = x.shape[1]
        x = x.flatten(0, 1)
        # (batch_size * N, M, 1)
        x = self.equiv_0(x)
        # (batch_size * N, M, hidden_size_0)
        x = self.relu(x)
        x = self.inv_0(x)
        # (batch_size * N, hidden_size_0)
        x = x.reshape(-1, N, self.hidden_size_0)
        # (batch_size, N, hidden_size_0)

        x = self.equiv_1(x)
        # (batch_size, N, hidden_size_1)
        x = self.relu(x)
        x = self.inv_1(x)
        # (batch_size, hidden_size_1)
        return x

    def forward(self, pos, neg, y):
        pos_vec = self._forward(pos)
        neg_vec = self._forward(neg)

        y = y.view(-1, self.meta_size)
        # (batch_size, meta_size)

        # vec0 = torch.abs(pos_vec - neg_vec)
        # vec1 = pos_vec + neg_vec
        x = torch.hstack([pos_vec, neg_vec, y])
        # x = torch.hstack([vec0, vec1, y])

        # (batch_size, hidden_size_1 + meta_size)
        x = self.regressor(x)
        # (batch_size, out_classes)
        return x


class DeepSetModelV6(torch.nn.Module):
    def __init__(self, hidden_size_0=1, hidden_size_1=1, predlast_hidden_size=1, meta_size=27, out_classes=1):
        super().__init__()
        self.hidden_size_0 = hidden_size_0
        self.hidden_size_1 = hidden_size_1
        self.predlast_hidden_size = predlast_hidden_size
        self.meta_size = meta_size
        self.out_classes = out_classes

        # pos
        self.inv_0_pos = deepsetlayers.InvLinear(hidden_size_0, hidden_size_0)
        self.inv_1_pos = deepsetlayers.InvLinear(hidden_size_1, 3 * hidden_size_1 // 2)
        self.equiv_0_pos = deepsetlayers.EquivLinear(1, hidden_size_0)
        self.equiv_1_pos = deepsetlayers.EquivLinear(hidden_size_0, hidden_size_1)

        # neg
        self.inv_0_neg = deepsetlayers.InvLinear(hidden_size_0, hidden_size_0)
        self.inv_1_neg = deepsetlayers.InvLinear(hidden_size_1, 3 * hidden_size_1 // 2)
        self.equiv_0_neg = deepsetlayers.EquivLinear(1, hidden_size_0)
        self.equiv_1_neg = deepsetlayers.EquivLinear(hidden_size_0, hidden_size_1)

        # all
        self.inv_0_all = deepsetlayers.InvLinear(hidden_size_0, hidden_size_0)
        self.inv_1_all = deepsetlayers.InvLinear(hidden_size_1, 3 * hidden_size_1 // 2)
        self.equiv_0_all = deepsetlayers.EquivLinear(1, hidden_size_0)
        self.equiv_1_all = deepsetlayers.EquivLinear(hidden_size_0, hidden_size_1)

        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(3 * (3 * hidden_size_1 // 2) + meta_size, 2 * predlast_hidden_size),
            torch.nn.BatchNorm1d(2 * predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.FIRST_DROPOUT_RATE),
            torch.nn.Linear(2 * predlast_hidden_size, predlast_hidden_size),
            torch.nn.BatchNorm1d(predlast_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(CONFIG.SECOND_DROPOUT_RATE),
            torch.nn.Linear(predlast_hidden_size, out_classes),
        )

    def _forward(self, x, mode='all'):
        if mode == 'pos':
            equiv_0 = self.equiv_0_pos
            equiv_1 = self.equiv_1_pos
            inv_0 = self.inv_0_pos
            inv_1 = self.inv_1_pos
        elif mode == 'neg':
            equiv_0 = self.equiv_0_neg
            equiv_1 = self.equiv_1_neg
            inv_0 = self.inv_0_neg
            inv_1 = self.inv_1_neg
        elif mode == 'all':
            equiv_0 = self.equiv_0_all
            equiv_1 = self.equiv_1_all
            inv_0 = self.inv_0_all
            inv_1 = self.inv_1_all

        x = x.unsqueeze(-1)
        # (batch_size, N, M, 1)
        N = x.shape[1]
        x = x.flatten(0, 1)
        # (batch_size * N, M, 1)
        x = equiv_0(x)
        # (batch_size * N, M, hidden_size_0)
        x = self.relu(x)
        x = inv_0(x)
        # (batch_size * N, hidden_size_0)
        x = x.reshape(-1, N, self.hidden_size_0)
        # (batch_size, N, hidden_size_0)

        x = equiv_1(x)
        # (batch_size, N, hidden_size_1)
        x = self.relu(x)
        x = inv_1(x)
        # (batch_size, hidden_size_1)
        return x

    def forward(self, pos, neg, y):

        pos_vec = self._forward(pos, 'pos')
        neg_vec = self._forward(neg, 'neg')

        all = torch.cat([pos, neg], dim=2)
        all_vec = self._forward(all, 'all')

        y = y.view(-1, self.meta_size)

        # (batch_size, meta_size)

        # vec0 = torch.abs(pos_vec - neg_vec)
        # vec1 = pos_vec + neg_vec
        x = torch.hstack([pos_vec, neg_vec, all_vec, y])
        # x = torch.hstack([vec0, vec1, y])

        # (batch_size, hidden_size_1 + meta_size)
        x = self.classifier(x)
        # (batch_size, out_classes)
        return x

def get_discriminator():
    features = CONFIG.features
    instances = CONFIG.instances
    classes = CONFIG.classes
    metas_length = CONFIG.meta_size
    lambdas_length = CONFIG.out_classes
    discriminator = Discriminator(features, instances, classes, metas_length, lambdas_length)
    return discriminator