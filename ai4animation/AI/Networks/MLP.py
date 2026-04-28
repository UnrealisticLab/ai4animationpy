# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn
from ai4animation.AI import Losses, Modules, Stats


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super(Model, self).__init__()

        self.InputStats = Stats.RunningStats(input_dim)
        self.OutputStats = Stats.RunningStats(output_dim)

        self.Layers = Modules.LinearEncoder(input_dim, hidden_dim, output_dim, dropout)

    def input_dim(self):
        return self.InputStats.Dim

    def output_dim(self):
        return self.OutputStats.Dim

    def forward(self, x):
        z = self.InputStats.Normalize(x)
        z = self.Layers(z)
        y = self.OutputStats.Denormalize(z)
        return y

    def learn(self, input, output, update_stats):
        if update_stats:
            self.InputStats.Update(input)
            self.OutputStats.Update(output)

        input = self.InputStats.Normalize(input)
        output = self.OutputStats.Normalize(output)
        prediction = self.Layers(input)

        loss = Losses.MSE(prediction, output)

        return {"MSE Loss": loss}
