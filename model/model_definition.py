import tensorflow as tf


def get_model_from_yaml_definition(conf, input_dim, output_dim):
    """
    Constructs a tensorflow model from a given yaml config file.
    :param conf: dictionary obtained from .yml config file
    :param input_dim: Input dimension of model
    :param output_dim: Output dimension of model
    :return: tf.keras.models.Model
    """
    input = tf.keras.layers.Input(shape=input_dim, name='input')
    bb = input

    # generate input based on backbone type
    bb_conf = conf['backbone']
    if bb_conf is not None:
        bb_type = bb_conf['type']

        if bb_type == "MLP":
            for l in bb_conf['layers']:
                bb = tf.keras.layers.Dense(l, activation=bb_conf['activation'])(bb)

                if 'dropout' in bb_conf:
                    bb = tf.keras.layers.Dropout(bb_conf['dropout'])(bb)

    # generate HEAD based on model type
    head = bb
    h_conf = conf['head']

    if conf['type'] == "AC":
        # single output branch with tanh activation
        for l in h_conf['layers']:
            head = tf.keras.layers.Dense(l, activation=h_conf['activation'])(head)

            if 'dropout' in h_conf:
                head = tf.keras.layers.Dropout(h_conf['dropout'])(head)

        output = tf.keras.layers.Dense(output_dim, activation='softmax')(head)

        model = tf.keras.models.Model(input, output)

    elif conf['type'] == "DLBIM":
        # 4 output branches for each wall (t, b, l, r)

        top = head
        for l in h_conf['layers']:
            top = tf.keras.layers.Dense(l, activation=h_conf['activation'])(top)

            if 'dropout' in h_conf:
                top = tf.keras.layers.Dropout(h_conf['dropout'])(top)

        top = tf.keras.layers.Dense(output_dim[0], activation='softmax', name="output_h_top")(top)

        bottom = head
        for l in h_conf['layers']:
            bottom = tf.keras.layers.Dense(l, activation=h_conf['activation'])(bottom)

            if 'dropout' in h_conf:
                bottom = tf.keras.layers.Dropout(h_conf['dropout'])(bottom)

        bottom = tf.keras.layers.Dense(output_dim[0], activation='softmax', name="output_h_bottom")(bottom)

        left = head
        for l in h_conf['layers']:
            left = tf.keras.layers.Dense(l, activation=h_conf['activation'])(left)
            if 'dropout' in h_conf:
                left = tf.keras.layers.Dropout(h_conf['dropout'])(left)
        left = tf.keras.layers.Dense(output_dim[1], activation='softmax', name="output_v_left")(left)

        right = head
        for l in h_conf['layers']:
            right = tf.keras.layers.Dense(l, activation=h_conf['activation'])(right)
            if 'dropout' in h_conf:
                right = tf.keras.layers.Dropout(h_conf['dropout'])(right)
        right = tf.keras.layers.Dense(output_dim[1], activation='softmax', name="output_v_right")(right)

        model = tf.keras.models.Model(input, [top, bottom, left, right])

    return model
