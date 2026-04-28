# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn
from ai4animation.AI import Manifolds, Modules, Stats


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_dim,
        channels,
        dimensions,
        decoder_dim,
        output_dim,
        dropout,
    ):
        super(Model, self).__init__()

        self.InputStats = Stats.RunningStats(input_dim)
        self.OutputStats = Stats.RunningStats(output_dim)

        self.InputDim = input_dim
        self.Channels = channels
        self.Dimensions = dimensions
        self.OutputDim = output_dim

        self.Encoder = Modules.LinearEncoder(
            input_dim, encoder_dim, channels * dimensions, dropout
        )
        self.Decoder = Modules.LinearEncoder(
            channels * dimensions, decoder_dim, output_dim, dropout
        )

    def forward(self, x, sample=False):
        x = self.InputStats.Normalize(x)
        z = self.Encoder(x)
        if sample:
            z = Manifolds.gumbel(z, self.Dimensions)
        else:
            z = Manifolds.softmax(z, self.Dimensions)
        y = self.Decoder(z)
        y = self.OutputStats.Denormalize(y)
        return y

    def learn(self, inputs, outputs, update_stats):
        if update_stats:
            self.InputStats.Update(inputs)
            self.OutputStats.Update(outputs)

        inputs = self.InputStats.Normalize(inputs)
        outputs = self.OutputStats.Normalize(outputs)

        z = self.Encoder(inputs)
        c = Manifolds.gumbel(z, self.Dimensions)
        y = self.Decoder(c)

        mse_fn = nn.MSELoss()
        loss = mse_fn(y, outputs)

        y = self.OutputStats.Denormalize(y)

        return {"Y": y}, {"MSE": loss}
