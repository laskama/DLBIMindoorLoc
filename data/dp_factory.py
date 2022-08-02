from data.ac_data_provider import ACdataProvider
from data.dlbim_data_provider import DLBIMdataProvider
from data.gia_vslam_data_connector import GiaVSLAMdataConnector


def get_data_provider(dataset_params, m_type):

    d_params = dataset_params['params']

    # determine preprocessing scaling method
    powed_scaling = False
    std_scaling = False
    if 'scaling' in d_params:
        if d_params['scaling'] == 'standard':
            std_scaling = True
        elif d_params['scaling'] == 'powed':
            powed_scaling = True

    conn = GiaVSLAMdataConnector(floors=d_params['floors'],
                                 devices=d_params['devices'],
                                 test_devices=d_params['test_devices'] if 'test_devices' in d_params else None,
                                 test_trajectories=d_params['test_trajectories'] if 'test_trajectories' in d_params else None
                                 ).load_dataset().load_walls()

    if m_type == 'AC':
        dp = ACdataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices()
        dp = dp.replace_missing_values().standardize_data(powed_scaling, std_scaling)
        dp = dp.setup_target_vector()
        dp = dp.generate_validation_indices()

    elif m_type == 'DLBIM':
        dp = DLBIMdataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices()
        dp = dp.load_walls()
        dp = dp.replace_missing_values().standardize_data(powed_scaling, std_scaling)
        dp = dp.setup_target_vector()
        dp = dp.generate_validation_indices()

    else:
        dp = None

    return dp
