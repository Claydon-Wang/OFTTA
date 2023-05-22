from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


class t3a(nn.Module):
    """
    "Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization"
    """
    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.args = args
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        


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

def get_classifier(args, model):
    # if args.MODEL.ARCH == 'Standard_feature':
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
        _, feature = output
        # print(classifier.weight.data)
        p_label = classifier(feature)    # output class-wise possibility
        yhat = torch.nn.functional.one_hot(p_label.argmax(1), num_classes=self.num_classes).float() # possibility to one-hot
        ent = softmax_entropy(p_label)

        # print(self.supports.shape)
        self.supports = self.supports.to(feature.device)
        # print(self.supports.shape, 'support')
        self.labels = self.labels.to(feature.device)
        # # print(self.labels.shape, 'labels')
        self.ent = self.ent.to(feature.device)

        # if self.flag <10:
        self.supports = torch.cat([self.supports, feature]) # (class, feature_dim) + batchsize(32)
        # print(self.supports.shape, 'support')
        self.labels = torch.cat([self.labels, yhat]) #  7,7 + batchsize(32)
        self.ent = torch.cat([self.ent, ent]) #  7 + batchsize(32) # 全部添加进来
        # self.flag+=1

        supports, labels = select_supports(self) # ranking and choose low-entropy

        supports = torch.nn.functional.normalize(supports, dim=1)
        # print(self.supports.shape, 'support')
        # print(labels.shape, 'label')
        weights = (supports.T @ (labels))

        outputs =  feature @ torch.nn.functional.normalize(weights, dim=0)
        # outputs = classifier(feature)

        # if True:
        #     for nm, m  in self.model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask = (torch.rand(p.shape)<0.4).float().cuda() 
        #                 with torch.no_grad():
        #                     p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)


    else:
        outputs = output

    return outputs

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
    #         m.requires_grad_(True)
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
