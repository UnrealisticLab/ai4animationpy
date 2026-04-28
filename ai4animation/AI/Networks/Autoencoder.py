# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn
from ai4animation.AI import Losses, Modules, Stats


class Model(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, dropout):
        super(Model, self).__init__()

        self.Stats = Stats.RunningStats(feature_dim)

        self.Encoder = Modules.LinearEncoder(
            feature_dim, hidden_dim, latent_dim, dropout
        )
        self.Decoder = Modules.LinearEncoder(
            latent_dim, hidden_dim, feature_dim, dropout
        )

    def feature_dim(self):
        return self.Encoder.L1.InputSize

    def latent_dim(self):
        return self.Decoder.L1.InputSize

    def forward(self, x):
        x = self.Stats.Normalize(x)
        z = self.Encoder(x)
        z = self.Decoder(z)
        y = self.Stats.Denormalize(z)
        return y

    def learn(self, features, update_stats):
        if update_stats:
            self.Stats.Update(features)

        features = self.Stats.Normalize(features)
        prediction = self.Decoder(self.Encoder(features))

        loss = Losses.MSE(prediction, features)

        return {"MSE": loss}
