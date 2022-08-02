import shapely.geometry
import numpy as np

from data.base_data_provider import BaseDataProvider

WALL_ORDER = ['top', 'bottom', 'left', 'right']


class DLBIMdataProvider(BaseDataProvider):

    def __init__(self, params, dc):
        super().__init__(params, dc)

        self.walls_h = None
        self.walls_v = None

        self.num_h_walls_per_floor = None
        self.num_v_walls_per_floor = None

        # target vectors of model
        # {'top': [], 'bottom': [], 'left': [], 'right': []}
        self.targets = None

    #
    # Getter & Setter via dataset connector
    #

    @property
    def walls_h(self):
        return self.dc.walls_h

    @property
    def walls_v(self):
        return self.dc.walls_v

    @walls_h.setter
    def walls_h(self, val):
        self.dc.walls_h = val

    @walls_v.setter
    def walls_v(self, val):
        self.dc.walls_v = val

    def load_walls(self):
        self.dc.load_walls()
        return self

    def get_num_h_walls(self):
        return np.shape(self.y[0])[1]

    def get_num_v_walls(self):
        return np.shape(self.y[2])[1]

    def get_output_dim(self):
        return self.get_num_h_walls(), self.get_num_v_walls()

    #
    #   Filtered access to data (if train/val data is accessed)
    #   -> do not include those values that do not have a valid
    #   target vector representation (all zeros) to avoid misleading training.
    #   Test data is left unchanged to not artificially change test results.
    #   See discussion of paper for details
    #

    def get_y_mask(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        t, b, r, l = [self.y[idx][subset] for idx in range(4)]
        a = np.any(t > 0.0, axis=1)
        b = np.any(b > 0.0, axis=1)
        c = np.any(r > 0.0, axis=1)
        d = np.any(l > 0.0, axis=1)

        return np.where(np.logical_and(np.logical_and(np.logical_and(a, b), c), d))[0]

    def get_y(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]

        if partition == 'train' or partition == 'val':
            # exclude those without proper target value
            mask = self.get_y_mask(partition)
            print("Excluded fingerprints from training: {}/{}".format(len(subset) - len(mask), len(subset)))
            res = [self.y[idx][subset][mask] for idx in range(4)]
        else:
            mask_test = self.get_y_mask(partition)
            print("Problematic test fingerprints (all-zero): {}/{}".format(
                len(subset) - len(mask_test), len(subset)))
            res = [self.y[idx][subset] for idx in range(4)]

        return res

    def get_x(self, partition='train'):
        x = super().get_x(partition)
        if partition == 'train' or partition == 'val':
            # exclude those without  proper target value
            mask = self.get_y_mask(partition)
            return x[mask]
        else:
            mask_test = self.get_y_mask(partition)
            print("Problematic test fingerprints (all-zero): {}/{}".format(
                len(x) - len(mask_test), len(x)))
            return x

    #
    #   Setup of target vector (see section IV-C)
    #

    def setup_target_vector(self):
        """
        Constructs the target vectors for each of the output branches of the network model
        (top, bottom, left, right) as proposed in section IV-C of the paper.
        Sets self.y which is used subsequently for model training/evaluation
        """
        names = WALL_ORDER
        dist_th = self.pr.get_param('dist_th')
        range_th = self.pr.get_param('range_th')
        close_th = self.pr.get_param('close_th')

        scores_dict = {names[idx]: [] for idx in range(len(names))}

        num_hor_walls = []
        num_ver_walls = []
        indices = []
        rss_test = []

        # compute for each floor individually and merge afterwards
        for idx in range(self.num_floors):
            mask = np.where(self.floor == self.floors[idx])[0]
            indices += [mask]
            rss_test += [self.rss[mask]]

            # get absolute score values
            scores = self.get_scores(floor=idx, dist_th=dist_th, range_th=range_th)

            # filter out those entries where enclosing wall is too close to corresponding
            # ground truth position (since this might mislead the model)
            scores = remove_close_walls(*scores, threshold=close_th)

            # normalize the absolute score values to obtain a valid probability distribution
            # over walls in each entry of the target vector
            scores = normalize_scores(*scores)

            # store amount of walls per floor for concatenating
            num_hor_walls.append(np.shape(scores[0])[1])
            num_ver_walls.append(np.shape(scores[2])[1])

            for s_idx, s in enumerate(scores):
                scores_dict[names[s_idx]].append(s)

        # merge target vectors of all floors
        for f in range(self.num_floors):
            np_hor_walls = np.array(num_hor_walls)
            sum_h = np.sum(np_hor_walls)
            start = np.sum(np_hor_walls[:f])
            end = sum_h - np.sum(np_hor_walls[:(f + 1)])

            scores_dict['top'][f] = np.pad(scores_dict['top'][f], ((0, 0), (start, end)),
                                           'constant', constant_values=0.0)
            scores_dict['bottom'][f] = np.pad(scores_dict['bottom'][f], ((0, 0), (start, end)),
                                              'constant', constant_values=0.0)

            np_ver_walls = np.array(num_ver_walls)
            sum_v = np.sum(np_ver_walls)
            start = np.sum(np_ver_walls[:f])
            end = sum_v - np.sum(np_ver_walls[:(f + 1)])

            scores_dict['left'][f] = np.pad(scores_dict['left'][f], ((0, 0), (start, end)),
                                            'constant', constant_values=0.0)
            scores_dict['right'][f] = np.pad(scores_dict['right'][f], ((0, 0), (start, end)),
                                             'constant', constant_values=0.0)

        # use indices to restore original order
        order = np.argsort(np.concatenate(indices))
        self.targets = {names[idx]: np.concatenate(
            scores_dict[names[idx]],
            axis=0)[order] for idx in range(4)}

        self.y = list(self.targets.values())

        self.num_h_walls_per_floor = num_hor_walls
        self.num_v_walls_per_floor = num_ver_walls

        return self

    def get_scores(self, floor=0, dist_th=25.0, range_th=10.0):
        """
        Computes the absolute score entries of the target vectors based on the projection
        of the positions onto the walls
        :param floor: Floor to access subset of data
        :param dist_th: distance threshold (max_dist in paper)
        :param range_th: range threshold (max_range in paper)
        :return: Tuple of absolute score values for each output branch (top, bottom, left, right)
        """

        walls_h = self.walls_h[floor]
        walls_v = self.walls_v[floor]

        # obtain subset of given floor
        floor_mask = np.where(self.floor == self.floors[floor])[0]
        y = self.pos[floor_mask]

        # setup walls
        walls_base = np.concatenate([walls_h, walls_v])
        walls = np.stack([walls_base] * len(y), axis=0)

        # for each label calc range score to the walls
        # project positions onto walls (see Eq. (1)-(4) of paper)
        proj_h = proj(walls, len(walls_h), y, horizontal_walls=True)
        proj_v = proj(walls, len(walls_h), y, horizontal_walls=False)

        # calc whether projection lie on wall (range_h = True) and compute distance to the closest endpoint of walls
        # this will be used subsequently to also include walls where the projection does not lie on the wall
        # but is not too far off (see section IV-C for details)
        range_h, dist_to_ep_h = get_range_mask_of_walls(proj_h, walls, len(walls_h),
                                                        horizontal_walls=True)
        range_v, dist_to_ep_v = get_range_mask_of_walls(proj_v, walls, len(walls_h),
                                                        horizontal_walls=False)

        # check whether distance to endpoints is below defined threshold
        mask_h = (dist_to_ep_h < range_th).astype(int)
        mask_v = (dist_to_ep_v < range_th).astype(int)

        # if either the projection lies directly on the wall or the projection is not too far off
        # -> set the mask entry to 1
        mask_h = mask_h + range_h.astype(int)
        mask_h[mask_h > 0.0] = 1.0

        mask_v = mask_v + range_v.astype(int)
        mask_v[mask_v > 0.0] = 1.0

        # compute the distance from the projected point to the wall (component-wise) ($d$ in paper)
        # those will be used to determine the probability mass within the target vector
        dist_loss = proj_h[:, :, 1] - y[:, np.newaxis, 1]
        dist_loss_v = proj_v[:, :, 0] - y[:, np.newaxis, 0]

        # filter out the values that do not match the previously computed range constraint)
        scores_h = mask_h * dist_loss
        scores_v = mask_v * dist_loss_v

        # for top horizontal wall => only use positive distances (other should be 0)
        scores_h[np.abs(scores_h) > dist_th] = 0.0
        scores_v[np.abs(scores_v) > dist_th] = 0.0

        # store computed values in the data structure
        scores_h_top = np.copy(scores_h)
        scores_h_bottom = np.copy(scores_h)

        scores_v_left = np.copy(scores_v)
        scores_v_right = np.copy(scores_v)

        scores_h_top[scores_h_top <= 0] = 0.0
        scores_h_bottom = np.negative(scores_h_bottom)
        scores_h_bottom[scores_h_bottom <= 0] = 0.0

        scores_v_left[scores_v_left <= 0] = 0.0
        scores_v_right = np.negative(scores_v_right)
        scores_v_right[scores_v_right <= 0] = 0.0

        return scores_h_top, scores_h_bottom, scores_v_left, scores_v_right

    #
    #   Decoding and information extraction of raw prediction output of the model
    #

    def decode_prediction(self, out):
        """
        Transforms the raw prediction of the model into the corresponding polygon and identifies
        the predicted floor based on the floor location of the chosen wall.
        In case the model does not find a consensus, the majority defines the floor and the walls of the
        branch which do not match the floor chosen by the majority is replaced by the most probable wall of the
        chosen floor.
        This includes identifying the chosen wall per output branch (top, bottom, left, right)
        and computing the intersection of those walls to obtain the polygon exterior.
        :param out: raw prediction output of model
        :return: Tuple of (predicted floors,
                           predicted walls of each branch,
                           indices where ensemble did not reach consensus)
        """

        max_idx = [np.argmax(out[idx], axis=1) for idx in range(4)]
        wall_pred = {WALL_ORDER[idx]: np.argmax(out[idx], axis=1) for idx in range(4)}

        floor_pred = np.zeros(len(out[0]))

        last_nh, last_nv = 0, 0
        sum_nh, sum_nv = 0, 0
        corrected_idx = []
        for f_idx, (nh, nv) in enumerate(
                zip(self.num_h_walls_per_floor, self.num_v_walls_per_floor)):
            sum_nh += nh
            sum_nv += nv
            mh = np.logical_and(last_nh <= max_idx[0], max_idx[0] < sum_nh)
            mt = np.logical_and(last_nh <= max_idx[1], max_idx[1] < sum_nh)
            ml = np.logical_and(last_nv <= max_idx[2], max_idx[2] < sum_nv)
            mr = np.logical_and(last_nv <= max_idx[3], max_idx[3] < sum_nv)

            m = np.stack([mh, mt, ml, mr], axis=1)
            m = m.astype(int)
            vote = np.sum(m, axis=1) > 2
            consensus = np.sum(m, axis=1) == 4

            floor_pred[np.where(vote)[0]] = f_idx

            # compute those indices where with majority on this floor
            # but no consensus was found
            correction_mask = np.where(np.logical_and(vote, ~consensus))[0]
            corrected_idx += correction_mask.tolist()

            for c_idx in correction_mask:
                # find out which wall prediction is on different floor
                w_pos = np.where(m[c_idx] < 1)[0]
                for o in w_pos:
                    cor_o = np.argmax(out[o][c_idx, last_nh:sum_nh])
                    max_idx[o][c_idx] = cor_o

            last_nh += nh
            last_nv += nv

        return floor_pred, wall_pred, corrected_idx

    def get_polygons_for_predictions(self, floor_pred, wall_preds, corrected_idx, obtain_metrics='aggregated', partition='test', return_raw_walls=False):
        """
        Constructs the polygon from the chosen walls and determines all relevant metrics
        :param floor_pred: The predicted floors of the model
        :param wall_preds: The predicted walls of the model ([[t1, t2, ..., tn], [b1, b2, ..., bn], ...])
        :param corrected_idx: Those indices where no consensus on floor prediction could be reached
        :param obtain_metrics: Whether to compute metrics
        :param partition: Dataset partition (train/val/test)
        :param return_raw_walls: Whether to return the raw predicted walls in place of the computed polygons
        :return: Tuple of (predicted polygons/predicted walls, predicted floors, dict that hold computed metrics)
        """
        y_true = self.get_data(self.pos, partition)
        floor_true = self.get_data(self.floor, partition)
        floors = self.floors

        metrics = {}

        a_hit_list = []
        f_hit_list = []
        cor_mask = []
        dev_list = []
        area_list = []

        wo = WALL_ORDER
        polys = []
        bound_walls = []
        for idx in range(len(wall_preds[wo[0]])):
            poly, bound_wall = self.get_polygon_for_prediction(
                [wall_preds[wo[j]][idx] for j in range(4)])

            polys += [poly]
            bound_walls += [bound_wall]

            if obtain_metrics is not None:
                point = shapely.geometry.Point(*y_true[idx, :])
                hit = poly.contains(point)

                f_hit_list += [floors[int(floor_pred[idx])] == floor_true[idx]]
                a_hit_list += [hit]
                area_list += [poly.area]
                cor_mask += [idx in corrected_idx]
                dev_list += [point.distance(poly) if not hit else 0]

        f_hit_list = np.array(f_hit_list)
        a_hit_list = np.array(a_hit_list)
        area_list = np.array(area_list)
        cor_mask = np.array(cor_mask)
        dev_list = np.array(dev_list)

        if obtain_metrics == 'raw':
            metrics['floor_ACC'] = f_hit_list
            metrics['area_ACC'] = a_hit_list
            metrics['area'] = area_list
            metrics['corrected'] = cor_mask
            metrics['deviation'] = dev_list
        elif obtain_metrics == 'aggregated':
            metrics['floor_ACC'] = len(np.where(f_hit_list)[0]) / len(f_hit_list)
            metrics['area_ACC'] = len(np.where(a_hit_list)[0]) / len(a_hit_list)
            metrics['area_mean'] = np.mean(area_list)
            metrics['area_median'] = np.median(area_list)
            metrics['corrected'] = len(np.where(cor_mask)[0]) / len(cor_mask)
            metrics['deviation'] = np.mean(dev_list[cor_mask])

        if return_raw_walls:
            return bound_walls, floor_pred, metrics
        else:
            return polys, floor_pred, metrics

    def get_polygon_for_prediction(self, wall_pred):
        # get walls of floor
        walls_h = np.concatenate(self.walls_h, axis=0)
        walls_v = np.concatenate(self.walls_v, axis=0)

        w_t = walls_h[wall_pred[0]]
        w_b = walls_h[wall_pred[1]]
        w_l = walls_v[wall_pred[2]]
        w_r = walls_v[wall_pred[3]]

        walls = np.concatenate((w_t.reshape(1, -1),
                                w_b.reshape(1, -1),
                                w_l.reshape(1, -1),
                                w_r.reshape(1, -1)))

        return compute_polygon_from_wall_prediction(w_t, w_b, w_l, w_r), walls


#
#   Static function required for target vector construction
#


def proj(walls, num_wh, y_pred_s, horizontal_walls=True):
    """
    Orthogonal projection of points onto walls.
    Implementation of equations (1)-(4) of paper.
    """
    if horizontal_walls:
        walls = walls[:, :num_wh, :]
    else:
        walls = walls[:, num_wh:, :]

    s_h = np.concatenate(
        [(walls[:, :, 2] - walls[:, :, 0])[:, :, np.newaxis],
         (walls[:, :, 3] - walls[:, :, 1])[:, :, np.newaxis]],
        axis=2)

    c_h = y_pred_s[:, np.newaxis, :2] - walls[:, :, :2]

    proj = np.einsum('ij,ijk->ijk',
                     np.einsum('ijk,ijk->ij', c_h, s_h) /
                     np.einsum('ijk,ijk->ij', s_h, s_h), s_h)

    # translate back
    proj = proj + walls[:, :, :2]

    return proj


def get_range_mask_of_walls(proj, walls, num_wh, horizontal_walls=True):
    """
    Compute mask that stores whether projected points lies on walls and computes the
    distance to the closest wall endpoint for all projected points.
    :param proj: Projected points
    :param walls: Walls onto which was projected
    :param num_wh: The number of horizontal walls (required for accessing horizontal or vertical walls)
    :param horizontal_walls: Whether to compute for horizontal (or vertical walls)
    :return: Tuple of (Range mask, distance to the closest endpoints)
    """
    if horizontal_walls:
        walls = walls[:, :num_wh, :]
        wall_idx = [0, 2]
        w_idx = 0
    else:
        walls = walls[:, num_wh:, :]
        wall_idx = [1, 3]
        w_idx = 1

    walls_x = np.concatenate([(walls[:, :, wall_idx[0]])[:, :, np.newaxis],
                           (walls[:, :, wall_idx[1]])[:, :, np.newaxis]], axis=2)
    walls_x_max = np.max(walls_x, axis=2)
    walls_x_min = np.min(walls_x, axis=2)

    wall_x_max_range = np.less_equal(proj[:, :, w_idx], walls_x_max)
    wall_x_min_range = np.greater_equal(proj[:, :, w_idx], walls_x_min)

    wall_range = np.logical_and(wall_x_max_range, wall_x_min_range)

    # dist to endpoints
    first_ep_dist = np.linalg.norm(proj[:, :, :] - walls[:, :, :2], axis=2)
    second_ep_dist = np.linalg.norm(proj - walls[:, :, 2:], axis=2)
    joined_ep_dist = np.stack([first_ep_dist, second_ep_dist], axis=2)
    dist_to_closest_endpoint = np.min(joined_ep_dist, axis=2)

    return wall_range, dist_to_closest_endpoint


def normalize_scores(s_h_top, s_h_bottom, s_v_left, s_v_right):
    """
    Normalize the computes target vector scores to form valid
    probability distribution of walls for each entry of the target vector.
    :param s_h_top: Scores for top output branch
    :param s_h_bottom: Scores for bottom output branch
    :param s_v_left: Scores for left output branch
    :param s_v_right: Scores for right output branch
    :return: Tuple that holds normalized scores for each output branch
    """
    s_h_top = 1.0 / s_h_top
    s_h_top[np.isinf(s_h_top)] = 0.0
    s_h_top = s_h_top / np.sum(s_h_top, axis=1)[:, np.newaxis]
    s_h_top[np.isnan(s_h_top)] = 0.0

    s_h_bottom = 1.0 / s_h_bottom
    s_h_bottom[np.isinf(s_h_bottom)] = 0.0
    s_h_bottom = s_h_bottom / np.sum(s_h_bottom, axis=1)[:, np.newaxis]
    s_h_bottom[np.isnan(s_h_bottom)] = 0.0

    s_v_left = 1.0 / s_v_left
    s_v_left[np.isinf(s_v_left)] = 0.0
    s_v_left = s_v_left / np.sum(s_v_left, axis=1)[:, np.newaxis]
    s_v_left[np.isnan(s_v_left)] = 0.0

    s_v_right = 1.0 / s_v_right
    s_v_right[np.isinf(s_v_right)] = 0.0
    s_v_right = s_v_right / np.sum(s_v_right, axis=1)[:, np.newaxis]
    s_v_right[np.isnan(s_v_right)] = 0.0

    return s_h_top, s_h_bottom, s_v_left, s_v_right


def remove_close_walls(s_h_top, s_h_bottom, s_v_left, s_v_right, threshold=5.0):
    """
    Detects the walls where the projected point is closer than the given threshold
    and removes the score entry from the score
    :param s_h_top: Scores for top output branch
    :param s_h_bottom: Scores for bottom output branch
    :param s_v_left: Scores for left output branch
    :param s_v_right: Scores for right output branch
    :param threshold: The distance threshold for when to discard score entry
    :return: The filtered scores
    """
    res = []
    for score_set in [s_h_top, s_h_bottom, s_v_left, s_v_right]:

        # compute threshold mask
        t_mask = score_set > threshold

        # compute blacklist mask (rows that would result in all 0 entries after
        # deletion of values below threshold
        t_thres = np.sum((t_mask).astype(int), axis=1)
        t_blacklist = np.full((len(score_set)), True)
        t_blacklist[np.where(t_thres < 1)[0]] = False
        t_blacklist = np.stack([t_blacklist] * np.shape(score_set)[1], axis=1)

        score_set[np.logical_and(~t_mask, t_blacklist)] = 0

        res.append(score_set)

    return res


#
#   Construct polygon from predicted walls by intersecting the
#   extended walls and determining their intersection points
#   Those represent the exterior points of the polygon
#   See Fig. 4 of paper for intuition
#

def compute_polygon_from_wall_prediction(w_t, w_b, w_r, w_l):
    return shapely.geometry.Polygon((
        get_intersect_lines(w_t, w_r),
        get_intersect_lines(w_r, w_b),
        get_intersect_lines(w_b, w_l),
        get_intersect_lines(w_l, w_t),
    ))


def get_intersect_lines(l1, l2):
    a1 = (l1[0], l1[1])
    a2 = (l1[2], l1[3])

    b1 = (l2[0], l2[1])
    b2 = (l2[2], l2[3])

    return get_intersect(a1, a2, b1, b2)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])      # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])            # get first line
    l2 = np.cross(h[2], h[3])            # get second line
    x, y, z = np.cross(l1, l2)           # point of intersection
    if z == 0:                           # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z
