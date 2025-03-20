import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights

class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ModifiedVGG16, self).__init__()
        # Load pre-trained VGG16
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights)
        
        # Freeze all layers except the final classifier
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify and unfreeze the classifier
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        # Only the new layer will have requires_grad=True by default
    
    def forward(self, x):
        return self.model(x)

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ModifiedResNet50, self).__init__()
        # Load pre-trained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify and unfreeze the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        # Only the new layer will have requires_grad=True by default
    
    def forward(self, x):
        return self.model(x)

def get_model(model_name, num_classes=100, pretrained=True):
    """
    Factory function to get the specified model.
    
    Args:
        model_name (str): Name of the model ('vgg16' or 'resnet50')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: The specified model with all layers frozen except the final classifier
    """
    if model_name.lower() == 'vgg16':
        return ModifiedVGG16(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'resnet50':
        return ModifiedResNet50(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Choose 'vgg16' or 'resnet50'")

if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    vgg = get_model('vgg16').to(device)
    resnet = get_model('resnet50').to(device)
    
    # Verify only final layer is trainable
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"VGG16 trainable parameters: {count_trainable_params(vgg)}")
    print(f"ResNet50 trainable parameters: {count_trainable_params(resnet)}")
    
    # Test with a sample input
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test forward pass
    with torch.no_grad():
        vgg_output = vgg(sample_input)
        resnet_output = resnet(sample_input)
    
    print(f"VGG16 output shape: {vgg_output.shape}")      # Should be [1, 100]
    print(f"ResNet50 output shape: {resnet_output.shape}")  # Should be [1, 100] 