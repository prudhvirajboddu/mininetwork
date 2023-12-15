import torch
import torch.nn.functional as F

a_b = torch.randn(8,5,4)
a_c = torch.randn(8,5)

b_b = torch.randn(8,5,4)
b_c = torch.randn(8,5)

def calculate_loss(a_b, a_c, b_b, b_c):
    # Regression Loss (MSE)
    regression_loss = F.smooth_l1_loss(a_b, b_b)

    # Classification Loss (Cross-Entropy)
    classification_loss = F.cross_entropy(a_c, b_c.argmax(dim=1))

    total_loss = regression_loss + classification_loss

    return total_loss
    

print(calculate_loss(a_b, a_c, b_b, b_c))
    