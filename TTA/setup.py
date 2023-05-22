import torch.optim as optim
from TTA.adapt_algorithm import norm, oftta, t3a, tent, pl, shot, sar, tast, tast_bn
import torch


def setup_source(args, model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_norm(args, model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.NORM(args, model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    # logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model

def setup_tent(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tent_model = tent.Tent(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tent_model

def setup_pl(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = pl.configure_model(model)
    params, param_names = pl.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tent_model = pl.PL(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tent_model

def setup_shot(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = shot.configure_model(model)
    params, param_names = shot.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tent_model = shot.SHOT(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tent_model


def setup_sar(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sar.SAM(params, base_optimizer, lr= args.lr, momentum=0.9)
    sar_model = sar.SAR(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return sar_model

def setup_t3a(args, model):

    model = t3a.configure_model(model)
    params, param_names = t3a.collect_params(model)
    optimizer = setup_optimizer(args, params)
    t3a_model = t3a.t3a(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return t3a_model

def setup_tast(args, model):

    model = tast.configure_model(model)
    params, param_names = tast.collect_params(model)
    optimizer = setup_optimizer(args, params)
    t3a_model = tast.TAST(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return t3a_model

def setup_tast_bn(args, model):

    model = tast_bn.configure_model(model)
    params, param_names = tast_bn.collect_params(model)
    optimizer = setup_optimizer(args, params)
    t3a_model = tast_bn.TAST_BN(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return t3a_model


def setup_oftta(args, model):

    model = oftta.configure_model(model)
    params, param_names = oftta.collect_params(model)
    optimizer = setup_optimizer(args, params)
    t3a_model = oftta.OFTTA(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return t3a_model

def setup_optimizer(args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if True:
        return optim.Adam(params,
                    lr=args.lr,
                    betas=(0.9, 0.99),
                    weight_decay=5e-4)
    # elif cfg.OPTIM.METHOD == 'SGD':
    #     return optim.SGD(params,
    #                lr=cfg.OPTIM.LR,
    #                momentum=cfg.OPTIM.MOMENTUM,
    #                dampening=cfg.OPTIM.DAMPENING,
    #                weight_decay=cfg.OPTIM.WD,
    #                nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def get_adaptation(args, base_model):
    if args.adaption == 'source':
        model = setup_source(args, base_model)
    elif args.adaption == 'norm':
        model = setup_norm(args, base_model)
    elif args.adaption == "tent":
        model = setup_tent(args, base_model)
    elif args.adaption == "t3a":
        model = setup_t3a(args, base_model)
    elif args.adaption == "tast":
        model = setup_tast(args, base_model)
    elif args.adaption == "tast_bn":
        model = setup_tast_bn(args, base_model)
    elif args.adaption == "oftta":
        model = setup_oftta(args, base_model)
    elif args.adaption == "pl":
        model = setup_pl(args, base_model)
    elif args.adaption == "shot":
        model = setup_shot(args, base_model)
    elif args.adaption == "sar":
        model = setup_sar(args, base_model)
    else:
        print('not exist this adaptation')

    return model