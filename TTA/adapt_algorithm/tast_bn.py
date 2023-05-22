from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class TAST_BN(nn.Module):
    """
    "Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization"
    """
    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.args = args
        # store supports examples and corresponding labels. we store test example itself instead of the feature representations.
        self.supports = None
        self.labels = None
        self.ent = None

        self.filter_K = args.filter_K
        self.num_classes = get_num_classes(args)
        # we restrict the size of support set
        if self.filter_K * self.num_classes >150 :
            self.filter_K = int(150 / self.num_classes)

        self.steps = steps

        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.optimizer = optimizer

        self.classifier = get_classifier(self.args, self.model)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


        self.softmax = torch.nn.Softmax(-1)

        self.tau = 10
        self.n_outputs = get_n_outputs(args)
        self.steps = 1

        self.k = 1 # 1 2 4 8

    def forward(self, x):
        # if self.episodic:
        #     self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(self, x, self.model, self.classifier, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_classifier(args, model):
    return model.classifier


def get_num_classes(args):
    print(args.dataset, 'args.dataset')
    if args.dataset == "uci":
        num_classes = 6
    elif args.dataset == 'unimib':
        num_classes = 17
    elif args.dataset == 'oppo':
        num_classes = 17
    else:
        print('not this dataset')

    return num_classes


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(self, x, model, classifier, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """

    output = model(x)
    p_supports = self.model(x)

    if isinstance(output, tuple):
        p_supports, _ = p_supports
        yhat = torch.nn.functional.one_hot(p_supports.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p_supports)


    if self.supports is None:
        self.supports = x
        self.labels = yhat
        self.ent = ent

    else:
        self.supports = self.supports.to(x.device)
        self.labels = self.labels.to(x.device)
        self.ent = self.ent.to(x.device)
        self.supports = torch.cat([self.supports, x])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])


    supports, labels = select_supports(self)


    outputs = tast_adapt(self, x, supports, labels)


    return outputs


@torch.enable_grad()
def tast_adapt(self, x, supports, labels):
    feats = torch.cat((x, supports), 0)
    
    output = self.model(feats)
    if isinstance(output, tuple):
        _, feats = output

    z, supports = feats[:x.size(0)], feats[x.size(0):]
    del feats

    with torch.no_grad():
        targets, outputs = target_generation(self, z, supports, labels)



    # PL with Ensemble
    logits = compute_logits(self, z, supports, labels)
    loss = F.kl_div(logits.log_softmax(-1), targets)

    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    return outputs  # outputs

def target_generation(self, z, supports, labels):
    dist = cosine_distance_einsum(self, z, supports)
    W = torch.exp(-dist)  # [B, N]
    temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes
    k = min(self.k, temp_k)

    values, indices = torch.topk(W, k, sorted=False)  # [B, k]
    topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else
    temp_labels = compute_logits(self, supports, supports, labels)  # [N, C]
    temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [N, C]
    temp_labels_outputs = torch.softmax(temp_labels, -1)  # [N, C]

    targets = topk_indices @ temp_labels_targets
    outputs = topk_indices @ temp_labels_outputs

    targets = targets / (targets.sum(-1, keepdim=True) + 1e-12)
    outputs = outputs / (outputs.sum(-1, keepdim=True) + 1e-12)

    return targets, outputs


def compute_logits(self, z, supports, labels):
    B, dim = z.size()
    N, dim_ = supports.size()

    temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports

    # normalize
    temp_z = torch.nn.functional.normalize(z, dim=1)
    temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)

    logits = self.tau * temp_z @ temp_centroids.T  # [B,C]

    return logits


# from https://jejjohnson.github.io/research_journal/snippets/numpy/euclidean/
def euclidean_distance_einsum(self, X, Y):
    # X, Y [n, dim], [m, dim] -> [n,m]
    XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
    YY = torch.einsum('md, md->m', Y, Y)  # [m]
    XY = 2 * torch.matmul(X, Y.T)  # [n,m]
    return XX + YY - XY

def cosine_distance_einsum(self, X, Y):
    # X, Y [n, dim], [m, dim] -> [n,m]
    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)
    XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
    YY = torch.einsum('md, md->m', Y, Y)  # [m]
    XY = 2 * torch.matmul(X, Y.T)  # [n,m]
    return XX + YY - XY



def select_supports(self):
    ent_s = self.ent
    y_hat = self.labels.argmax(dim=1).long()
    filter_K = self.filter_K
    if filter_K == -1:
        indices = torch.LongTensor(list(range(len(ent_s))))
    else:
        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

    self.supports = self.supports[indices]
    self.labels = self.labels[indices]
    self.ent = self.ent[indices]

    return self.supports, self.labels






def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # model.train() # 使用新的bn
    model.train()
    # # disable grad, frozen model
    model.requires_grad_(False)
    # # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


def get_n_outputs(args):
    print(args.dataset, 'args.dataset')
    if args.dataset == "uci":
        n_outputs = 15360
    elif args.dataset == 'unimib':
        n_outputs = 8448
    elif args.dataset == 'oppo':
        n_outputs = 1024
    else:
        print('not this dataset')

    return n_outputs