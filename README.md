# TorchLosses
## IoULoss
### Highlights
- Calculates both positive IoU and negative IoU
- Ignores all-negative channels for positive IoU calculation

## OHEMLoss
### Highlights
- Cooperates with any loss (set reduction='none')
- Sorts in any dimention
- Dynamic hard example proportion (cosine annealing)
