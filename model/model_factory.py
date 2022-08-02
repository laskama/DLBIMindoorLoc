from model.ac_model import ACmodel
from model.dlbim_model import DLBIMmodel


def get_model(model_params, base_dir, dp):

    m_params = model_params['params'] if 'params' in model_params else {}
    m_type = model_params['type']
    model_name = model_params['name']

    if m_type == 'DLBIM':
        model = DLBIMmodel(m_params, dp, base_dir, model_name=model_name)
    elif m_type == 'AC':
        model = ACmodel(m_params, dp, base_dir, model_name=model_name)
    else:
        model = None

    model.setup_model(model_params)

    return model
