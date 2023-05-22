from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit


def find_bns(model, priors):
    replace_mods = []
    queue = [(model, '')]
    index = 0
    while queue:
        parent, name = queue.pop(0)
        for child_name, child in parent.named_children():
            module_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.BatchNorm2d):
                module = Weighted_BN(child, priors[index]).cuda()
                index +=1
                replace_mods.append((parent, child_name, module))
            else:
                queue.append((child, module_name))
    return replace_mods


class Weighted_BN(nn.Module):
    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.prior = prior
        self.running_mean = deepcopy(layer.running_mean.detach().clone())
        self.running_std = deepcopy(torch.sqrt(layer.running_var.detach().clone()) + 1e-5)

    def forward(self, input):
        batch_mean = input.mean([0, 2, 3])
        batch_std = torch.sqrt(input.var([0, 2, 3], unbiased=False) + self.layer.eps)

        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std = batch_std.detach().clone()


        # weighted
        weighted_mean = self.prior * self.running_mean + (1. - self.prior) * batch_mean.detach()
        weighted_std = self.prior * self.running_std + (1. - self.prior) * batch_std.detach()
        
        input = (input - weighted_mean[None,:,None,None]) / weighted_std[None,:,None,None]



        return input * self.layer.weight[None,:,None,None] + self.layer.bias[None,:,None,None]



class OFTTA(nn.Module):
    """
    """
    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.args = args
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        

        priors = get_priors(args)
        # replace BN in the model
        replace_mods = find_bns(self.model, priors)
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)




        self.classifier = get_classifier(self.args, self.model)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


        self.num_classes = get_num_classes(args)

        self.warmup_supports = self.classifier.weight.data 
        warmup_prob = self.classifier(self.warmup_supports) 
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()


        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.filter_K
        self.softmax = torch.nn.Softmax(-1)

        self.flag = 1



    def forward(self, x):
        if self.episodic:
            self.reset()

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



@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(self, x, model, classifier, optimizer):

    output = model(x)

    if isinstance(output, tuple):
        _, feature = output
        p_label = classifier(feature)    # output class-wise possibility
        yhat = torch.nn.functional.one_hot(p_label.argmax(1), num_classes=self.num_classes).float() # possibility to one-hot
        ent = softmax_entropy(p_label)


        self.supports = self.supports.to(feature.device)
        self.labels = self.labels.to(feature.device)
        self.ent = self.ent.to(feature.device)
        self.supports = torch.cat([self.supports, feature]) 

        self.labels = torch.cat([self.labels, yhat]) 
        self.ent = torch.cat([self.ent, ent]) 


        supports, labels = select_supports(self) # ranking and choose low-entropy

        supports = torch.nn.functional.normalize(supports, dim=1)

        weights = (supports.T @ (labels))

        outputs =  feature @ torch.nn.functional.normalize(weights, dim=0)



    else:
        outputs = output

    return outputs


# inspired by https://github.com/matsuolab/T3A
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
            indices.append(indices1[y_hat==i][indices2][:filter_K]) 
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
                if np in ['weight', 'bias']:  
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
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # m.requires_grad_(True)
            m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model


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

def get_priors(args):
    prior_max = 0.99
    prior_min = 0.1
    if args.dataset == "uci":
        num_prior = 6
    elif args.dataset == 'unimib':
        num_prior = 6
    elif args.dataset == 'oppo':
        num_prior = 3
    else:
        print('not this dataset')

    priors = []
    factor = (prior_max/prior_min) ** (1/(num_prior-1))
    for _ in range(num_prior):
        tmp_prior = prior_min
        prior_min = prior_min * factor
        priors.append(tmp_prior)

    return priors