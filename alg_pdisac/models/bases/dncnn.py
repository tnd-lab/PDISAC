import torch
import torch.nn as nn
import torch.nn.init as init


class DnCNN(nn.Module):
    def __init__(self, depth=20, n_filters=64, kernel_size=3, img_channels=3):
        """DnCNN baseline (residual learning: the network predicts the disturbance).

        Parameters
        ----------
        depth : int
            Number of convolutional layers (configs/pdisac_system.json: dncnn_depth).
        n_filters : int
            Number of filters of each hidden convolutional layer (filters).
        kernel_size : int
            Square kernel size (kernel_size).
        img_channels : int
            Input / output channels; 2 for the real and imaginary parts of an RD map
            (image_channels).
        """
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False,
                )
            )
            # layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))

        # self.lastconv = nn.Conv2d(in_channels=n_filters, out_channels=img_channels, kernel_size=kernel_size,
        #                         padding=1, bias=False)

        layers.append(
            nn.Conv2d(
                in_channels=n_filters,
                out_channels=img_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            # nn.Tanh(inplace=True),
        )
        self.dncnn = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, Z_rd):
        noise = self.dncnn(Z_rd)
        output = Z_rd - noise

        return output

    def predict(self, Z_rd):
        return self.forward(Z_rd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
