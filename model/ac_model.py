from data.ac_data_provider import ACdataProvider
import tensorflow as tf
import numpy as np

from model.base_model import BaseModel
from model.model_definition import get_model_from_yaml_definition


class ACmodel(BaseModel):

    def __init__(self, model_params, data_provider: ACdataProvider, base_dir, model=None, model_name='model'):
        super(ACmodel, self).__init__(model_params, data_provider, base_dir, model, model_name)
        self.dp = data_provider

    def setup_model(self, model_params):

        input_dim = self.dp.get_input_dim()
        output_dim = self.dp.get_output_dim()

        # obtain the tensorflow model for the specified parameters
        model = get_model_from_yaml_definition(model_params, input_dim, output_dim)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.pr.get_param('lr'))
        )

        self.model = model

        return self

    def evaluate_model(self, metrics='aggregated', load_weights=True, compute_error_vec=True):
        if load_weights:
            self.model.load_weights(self.base_dir + self.model_name + ".hdf5")

        out = self.model.predict(self.dp.get_x('test'))

        polys, floor_pred, metrics = self.dp.get_polygons_for_predictions(np.argmax(out, axis=1), obtain_metrics=True)

        return metrics, polys, floor_pred
