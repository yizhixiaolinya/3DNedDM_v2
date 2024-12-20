import copy

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    # if load_sd:
    #     if model_spec['name'] == 'lccd':
    #         model.load_state_dict(model_spec['sd_G'])
    #     else:
    #         model.load_state_dict(model_spec['sd_D'])

    if 'sd_G' in model_spec:  # 加载生成器权重
        model.load_state_dict(model_spec['sd_G'])
    elif 'sd_D' in model_spec:  # 加载判别器权重
        model.load_state_dict(model_spec['sd_D'])

    return model