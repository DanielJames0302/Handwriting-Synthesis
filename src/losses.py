import torch
import torch.nn.functional as F


def dis_criterion(fake_op, real_op, label_smoothing=0.1):
    """
    Hinge loss function for discriminator with optional label smoothing
    
    Args:
        fake_op: Discriminator output for fake samples
        real_op: Discriminator output for real samples
        label_smoothing: Amount of label smoothing (0.0 = no smoothing, 0.1 = 10% smoothing)
    """
    # Apply label smoothing: instead of expecting real_op >= 1.0, expect >= (1.0 - label_smoothing)
    # Instead of expecting fake_op <= -1.0, expect <= (-1.0 + label_smoothing)
    real_target = 1.0 - label_smoothing
    fake_target = -1.0 + label_smoothing
    return torch.mean(F.relu(real_target - real_op)) + torch.mean(F.relu(fake_op - fake_target))


def gen_criterion(dis_preds, ctc_loss):
    """
    Hinge loss function for generator
    """
    return ctc_loss - torch.mean(dis_preds)
    # return -torch.mean(dis_preds)


def compute_ctc_loss(criterion, ip, tgt, tgt_lens):
    """
    CTC loss function for the OCR network
    """
    ip_lens = torch.full(size=(ip.shape[1],), fill_value=ip.shape[0])
    return criterion(ip, tgt, ip_lens, tgt_lens)
