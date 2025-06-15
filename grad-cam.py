# gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()

        # Pool gradients: shape [C, H, W]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # shape: [C]
        activations = self.activations.squeeze(0)  # shape: [C, H, W]

        # Weight activations
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= torch.max(heatmap) + 1e-8  # Normalize to [0, 1]

        return heatmap.numpy()
