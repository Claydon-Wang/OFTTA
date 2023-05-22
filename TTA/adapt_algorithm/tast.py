from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# motivated from "https://github.com/cs-giung/giung2/blob/c8560fd1b5/giung2/layers/linear.py"
class BatchEnsemble(nn.Module):
    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.in_zs = indim
        self.out_zs = outdim

        # register parameters
        self.register_parameter(
            "weight", nn.Parameter(
                torch.Tensor(self.out_zs, self.in_zs)
            )
        )
        self.register_parameter(
            "bias", nn.Parameter(
                torch.Tensor(self.out_zs)
            )
        )

        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_zs)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_zs)
            )
        )

        use_ensemble_bias = True
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_zs)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        self.init_mode = init_mode
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1 = x.size()
        r_x = x.unsqueeze(0).expand(self.ensemble_size, B, D1) #
        r_x = r_x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def reset(self):
        init_details = [0,1]
        initialize_tensor(self.weight, self.init_mode, init_details)
        initialize_tensor(self.alpha_be, self.init_mode, init_details)
        initialize_tensor(self.gamma_be, self.init_mode, init_details)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")
        if self.bias is not None:
            initialize_tensor(self.bias, "zeros")

def initialize_tensor(
        tensor: torch.Tensor,
        initializer: str,
        init_values: list[float] = [],
    ) -> None:
    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )
    elif initializer == 'xavier_normal':
        torch.nn.init.xavier_normal_(tensor)

    elif initializer == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(tensor)
    else:
        raise NotImplementedError(
            f"Unknown initializer: {initializer}"
        )


class TAST(nn.Module):
    """
    "Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization"
    """
    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        self.args = args
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        
        self.optimizer = optimizer

        self.classifier = get_classifier(self.args, self.model)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # T3A warm up
        self.num_classes = get_num_classes(args)
        # print(self.classifier[0])
        self.warmup_supports = self.classifier.weight.data #(10,640)
        # print(self.warmup_supports.shape,'warmup_supports' )
        warmup_prob = self.classifier(self.warmup_supports) 
        self.warmup_ent = softmax_entropy(warmup_prob)
        # print(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        # print(self.warmup_labels)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.filter_K
        self.softmax = torch.nn.Softmax(-1)

        self.num_ensemble = 10 # [1, 5, 10, 20]
        self.tau = 10
        self.init_mode = 'kaiming_normal'
        self.n_outputs = get_n_outputs(args)
        self.steps = 1
        self.mlps = BatchEnsemble(self.n_outputs, self.n_outputs // 4, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.args.lr)
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
    # if args.MODEL.ARCH == 'Standard_z':
    return model.classifier
    # if args.MODEL.ARCH == 'Hendrycks2020AugMix_ResNeXt':
    #     return model.classifier

    # else:
    #     print('not this model')

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
    # forward
    # print(x.shape)
    output = model(x)

    if isinstance(output, tuple):
        _, z = output
        # print(classifier.weight.data)
        p_label = classifier(z)    # output class-wise possibility
        yhat = torch.nn.functional.one_hot(p_label.argmax(1), num_classes=self.num_classes).float() # possibility to one-hot
        ent = softmax_entropy(p_label)

        # print(self.supports.shape)
        self.supports = self.supports.to(z.device)
        # print(self.supports.shape, 'support')
        self.labels = self.labels.to(z.device)
        # # print(self.labels.shape, 'labels')
        self.ent = self.ent.to(z.device)

        # if self.flag <10:
        self.supports = torch.cat([self.supports, z]) # (class, z_dim) + batchsize(32)
        # print(self.supports.shape, 'support')
        self.labels = torch.cat([self.labels, yhat]) #  7,7 + batchsize(32)
        self.ent = torch.cat([self.ent, ent]) #  7 + batchsize(32) # 全部添加进来
        # self.flag+=1

        supports, labels = select_supports(self) # ranking and choose low-entropy



        # # T3A
        # supports = torch.nn.functional.normalize(supports, dim=1)
        # weights = (supports.T @ (labels))
        # outputs =  z @ torch.nn.functional.normalize(weights, dim=0)

        # TAST
        outputs = tast_adapt(self, z, supports, labels)


    else:
        outputs = output

    return outputs


@torch.enable_grad()
def tast_adapt(self, z, supports, labels):
    # targets : pseudo labels, outputs: for prediction
    with torch.no_grad():
        targets, outputs = target_generation(self, z, supports, labels)

    self.optimizer.zero_grad()

    loss = None
    logits = compute_logits(self, z, supports, labels, self.mlps)

    for ens in range(self.num_ensemble):
        if loss is None:
            loss = F.kl_div(logits[ens].log_softmax(-1), targets[ens])
        else:
            loss += F.kl_div(logits[ens].log_softmax(-1), targets[ens])

    loss.backward()
    self.optimizer.step()

    return outputs  # outputs

def target_generation(self, z, supports, labels):
    # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
    dist = cosine_distance_einsum(self, z, supports)
    W = torch.exp(-dist)  # [B, N]

    temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes
    k = min(self.k, temp_k)

    values, indices = torch.topk(W, k, sorted=False)  # [B, k]
    topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else
    temp_labels = compute_logits(self, supports, supports, labels, self.mlps)  # [ens, N, C]
    temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
    temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

    topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

    # targets for pseudo labels. we use one-hot class distribution
    targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
    targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
    #targets = targets.mean(0)  # [B,C]

    # outputs for prediction
    outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
    outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
    outputs = outputs.mean(0)  # [B,C]

    return targets, outputs


def compute_logits(self, z, supports, labels, mlp):
    '''
    :param z: unlabeled test examples
    :param supports: support examples
    :param labels: labels of support examples
    :param mlp: multiple projection heads
    :return: classification logits of z
    '''
    B, dim = z.size()
    N, dim_ = supports.size()

    mlp_z = mlp(z)
    mlp_supports = mlp(supports)

    assert (dim == dim_)

    logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
    for ens in range(self.num_ensemble):
        temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                ens * N: (ens + 1) * N]

        # normalize
        temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
        temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)

        logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

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

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            # print(indices2)
            indices.append(indices1[y_hat==i][indices2][:filter_K]) # 筛选前多少个熵小于的
        indices = torch.cat(indices)


        self.supports = self.supports[indices] #加入到supports里
        # print(self.supports.shape, 'support')
        self.labels = self.labels[indices]  #按照顺序排的
        # print(self.labels)
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
    model.eval()
    # # disable grad, frozen model
    model.requires_grad_(False)
    # # configure norm for tent updates: enable grad + force batch statisics
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         # m.requires_grad_(True)
    #         # force use of batch stats in train and eval modes
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
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