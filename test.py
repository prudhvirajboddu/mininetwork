import torch
import torch.nn as nn

loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=1.0)

loss_cl = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=1.0)

a = torch.randn(8,5,4)

print(a.shape)
p = torch.randn(8,5,4)
n = torch.randn(8,5,4)

a_c = torch.randn(8,5)
p_c = torch.randn(8,5)
n_c = torch.randn(8,5)

loss_ap = loss(a,p,n)

loss_ac = loss_cl(a_c,p_c,n_c)

print(loss_ap.item()) 

print(loss_ac.item())