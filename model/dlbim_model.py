from data.dlbim_data_provider import DLBIMdataProvider
from model.base_model import BaseModel
import tensorflow as tf

from model.model_definition import get_model_from_yaml_definition


class DLBIMmodel(BaseModel):

    def __init__(self, model_params, data_provider: DLBIMdataProvider, base_dir, model=None, model_name='model'):
        super().__init__(model_params, data_provider, base_dir, model, model_name)
        self.dp = data_provider

    def setup_model(self, model_params):

        input_dim = self.dp.get_input_dim()
        output_dim = self.dp.get_output_dim()

        # obtain the tensorflow model for the specified parameters
        model = get_model_from_yaml_definition(model_params, input_dim, output_dim)

        # obtain output of other head for regularization
        output_h_bottom = [layer for layer in model.layers if layer.name == 'output_h_bottom'][0]
        output_v_right = [layer for layer in model.layers if layer.name == 'output_v_right'][0]

        reg_const = self.pr.get_param('dlp_reg_lambda')

        losses = {
            'output_h_top': custom_cat_loss(other_output=output_h_bottom.output, const=reg_const),
            'output_h_bottom': tf.keras.losses.categorical_crossentropy,
            'output_v_left': custom_cat_loss(other_output=output_v_right.output, const=reg_const),
            'output_v_right': tf.keras.losses.categorical_crossentropy
        }

        model.compile(
            loss=losses,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.pr.get_param('lr'))
        )

        self.model = model

        return self

    def evaluate_model(self, metrics='aggregated', load_weights=True, compute_error_vec=False):
        if load_weights:
            self.model.load_weights(self.base_dir + self.model_name + ".hdf5")

        out = self.model.predict(self.dp.get_x('test'))

        polys, floor_pred, metrics = self.dp.get_polygons_for_predictions(*self.dp.decode_prediction(out), obtain_metrics=metrics, return_raw_walls=True)

        return metrics, polys, floor_pred


def custom_cat_loss(other_output, const=2.0):
    """
    Cross entropy loss plus regularization term to avoid collapsing
    polygon predictions.
    :param other_output: other output vector of classification head for opposite wall
    :param const: regularization constant (lambda in paper)
    :return:
    """
    const_val = tf.constant(const)

    def loss(y_true, y_pred):

        l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # regularization term: p^{ht} * p^{hb} to avoid collapsing polygons
        reg = tf.reduce_sum(other_output * y_pred, axis=1)

        return l + const_val * reg

    return loss
