import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0, bidirectional: bool = False, eps: float = 1e-10):
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional
        self.eps = eps

    def calc_cosine(self, x1, x2):
        return torch.nn.functional.cosine_similarity(x1, x2, dim=-1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_cosine(anchor, positive)
        anchor_norm = anchor / anchor.norm(dim=-1, keepdim=True)
        neg_normalized = negatives / negatives.norm(dim=-1, keepdim=True)
        distance_negative_forward = torch.mm(anchor_norm, neg_normalized.t())

        # Calculate exponentials for positive and negative distances
        distance_positive_exp = torch.exp(distance_positive / self.temperature)
        distance_negative_exp_forward = torch.exp(distance_negative_forward / self.temperature).sum(dim=-1) + distance_positive_exp

        # Calculate forward loss
        loss_forward = -torch.log(distance_positive_exp / distance_negative_exp_forward + self.eps)

        if self.bidirectional:
            # Calculate backward distances
            positive_norm = positive / positive.norm(dim=-1, keepdim=True)
            distance_negative_backward = torch.mm(positive_norm, neg_normalized.t())
            distance_negative_exp_backward = torch.exp(distance_negative_backward / self.temperature).sum(dim=-1) + distance_positive_exp

            # Calculate backward loss
            loss_backward = -torch.log(distance_positive_exp / distance_negative_exp_backward + self.eps)

            # Calculate total loss
            loss = torch.mean(loss_forward) + torch.mean(loss_backward)
        else:
            loss = torch.mean(loss_forward)

        return loss