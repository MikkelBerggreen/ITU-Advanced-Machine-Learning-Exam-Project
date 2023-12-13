from .imports import *

class PretrainedModel(nn.Module):
    def __init__(self, model_str):
        super(PretrainedModel, self).__init__()

        if model_str == "AlexNet":
            base_model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(256 * 6 * 6, 4096),
                nn.Dropout(0.75),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.Dropout(0.75),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
            )

        elif model_str == "ResNet18":
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = base_model.avgpool
            self.classifier = nn.Linear(base_model.fc.in_features, 8428)

        elif model_str == "ResNet50":
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = base_model.avgpool
            self.classifier = nn.Linear(base_model.fc.in_features, 8428)

        else:
            raise ValueError("Invalid model name")

    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward_with_intermediate(self, x):
        outputs = {}

        for i, layer in enumerate(self.features):
            x = layer(x)
            outputs[f'features_{i}'] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Check if the classifier is a single layer or a sequence of layers
        if isinstance(self.classifier, nn.Sequential):
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                outputs[f'classifier_{i}'] = x
        else:
            # For a single layer classifier
            x = self.classifier(x)
            outputs['classifier'] = x

        return x, outputs
    
class AlexNetVanilla(nn.Module):
    def __init__(self, dimension_length, num_outputs):
        super(AlexNetVanilla, self).__init__()

        # This model takes inspiration from: https://github.com/dansuh17/alexnet-pytorch/blob/master/model.py
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 30 x 30)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 14 x 14)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 14 x 14)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 6 x 6)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 6 x 6)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 6 x 6)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 2 x 2)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_outputs),
        )


    def forward(self, x):
        #x = x.unsqueeze(1)  # Add a channel dimension for grayscale images

        x = self.net(x)
        x = x.view(-1, 256 * 2 * 2)  # reduce the dimensions for linear layer input
        return self.classifier(x)

    def forward_with_intermediate(self, x):
        outputs = {}  # Dictionary to store intermediate outputs

        #x = x.unsqueeze(1)  # Add a channel dimension for grayscale images
        for i, layer in enumerate(self.net):
            x = layer(x)
            outputs[f'net_{i}'] = x

        x = x.view(-1, 256 * 2 * 2)  # Flatten the output for the classifier
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            outputs[f'classifier_{i}'] = x

        return x, outputs