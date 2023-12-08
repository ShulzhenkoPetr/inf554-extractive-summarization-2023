import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def calc_cosine(self, x1, x2):
        return torch.nn.functional.cosine_similarity(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses_euclidean = torch.relu(distance_positive - distance_negative + self.margin)

        dist_cos_pos = self.calc_cosine(anchor, positive)
        dist_cos_neg = self.calc_cosine(anchor, negative)
        losses_cosine = torch.relu(dist_cos_neg - dist_cos_pos + .5)
        return losses_cosine.mean()
