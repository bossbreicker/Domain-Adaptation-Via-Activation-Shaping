import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
class ActivationShapingLayer(nn.Module):
    def __init__(self):
        super(ActivationShapingLayer, self).__init__()
        # Initialize any additional parameters or layers if needed

    def forward(self, A, M):
        zero_ratio = 0
        M = torch.randn_like(A) < zero_ratio 
        A_binarized = (A > 0).float()  # Binarize A
        M_binarized = (M > 0).float()  # Binarize M
        return A_binarized * M_binarized

# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
def activation_shaping_hook(module, input, output):
    A = output
    zero_ratio = 0
    M = torch.randn_like(A) < zero_ratio  # Replace with your method to generate M

    # Binarize A and M
    A_binarized = (A > 0).float()

    # Element-wise product
    return A_binarized * M.float()

class ModifiedResNet(BaseResNet18):
    def __init__(self, *args, **kwargs):
        super(ModifiedResNet, self).__init__(*args, **kwargs)
        #self.conv1.register_forward_hook(activation_shaping_hook)
        self.layer1.register_forward_hook(activation_shaping_hook)
        #self.layer2.register_forward_hook(activation_shaping_hook)
        #self.layer3.register_forward_hook(activation_shaping_hook)
        #self.layer4.register_forward_hook(activation_shaping_hook)
        #self.avgpool.register_forward_hook(activation_shaping_hook)
        #self.fc.register_forward_hook(activation_shaping_hook)

#IF you encounteer problems with those lines just try the following one by removing hashtag and put new ones from lines 32 to 41.
#class ModifiedResNet(BaseResNet18):
    #def __init__(self):
        #super(ModifiedResNet, self).__init__()
        #self.resnet.conv1.register_forward_hook(activation_shaping_hook)
        ##self.resnet.layer1.register_forward_hook(activation_shaping_hook)
        ##self.resnet.layer2.register_forward_hook(activation_shaping_hook)
        ##self.resnet.layer3.register_forward_hook(activation_shaping_hook)
        ##self.resnet.layer4.register_forward_hook(activation_shaping_hook)
        ##self.resnet.avgpool.register_forward_hook(activation_shaping_hook)
        ##self.resnet.fc.register_forward_hook(activation_shaping_hook)

######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ASHResNet18, self).__init__()
        # Load the base ResNet-18 model
        self.base_model = resnet18(pretrained=True)

        # Replace the fully connected layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

        # Initialize the activation shaping layer(s)
        self.activation_shaping_layer = ActivationShapingLayer()

    def forward(self, x, M=None):
        # Forward pass through the base ResNet-18 model
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)

        # Apply the activation shaping layer after a specific layer
        # Note: Ensure M is provided and has the correct shape
        if M is not None:
            x = self.activation_shaping_layer(x, M)

        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        return x
    def record_activation_maps(self, x):
        x = x.to(device=self.base_model.conv1.weight.device, dtype=self.base_model.conv1.weight.dtype)
        activation_maps = []
        with torch.no_grad():  # Ensure to not track gradients
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            activation_maps.append(x) 
        return activation_maps
    
    def forward_with_activation_shaping(self, x, Mt):
        x = x.to(device=self.base_model.conv1.weight.device, dtype=self.base_model.conv1.weight.dtype)
        # Forward pass with activation shaping
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)

        # Apply activation shaping using the provided activation map Mt
        # Assuming Mt is the activation map you want to apply after the first layer
        x = x * Mt[0]  # Element-wise multiplication with the first activation map

        # Continue the forward pass through the remaining layers
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        return x
######################################################
