

import numpy as np
import tfunet
from tfunet.image.generator import GrayScaleDataProvider
from tfunet.train import Trainer


np.random.seed(2018)


generator = GrayScaleDataProvider(nx=572, ny=572, cnt=20, rectangles=False)


print(f"n_channels: {generator.channels}")
print(f"n_classes: {generator.n_class}")

net = tfunet.TFUnet(n_channels=generator.channels,
                    n_classes=generator.n_class,
                    n_layers=3,
                    n_filters=16)

trainer = tfunet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, "./unet_trained",
                     training_iters=32,
                     epochs=5,
                     dropout=0.75,  # probability to keep units
                     display_step=2)



