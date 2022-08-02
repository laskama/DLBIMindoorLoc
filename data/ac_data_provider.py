import numpy as np
import shapely.geometry

from data.base_data_provider import BaseDataProvider


class ACdataProvider(BaseDataProvider):

    def __init__(self, params, dc):
        super(ACdataProvider, self).__init__(params, dc)

        self.chosen_area_type = None

    def setup_target_vector_from_areas(self, areas):
        num_areas = np.sum(np.array([len(a) for a in areas]))
        targets = np.zeros((len(self.pos), num_areas))

        last_area_idx = 0
        for floor_idx in range(self.num_floors):
            sub_idx = np.where(self.floor == self.floors[floor_idx])[0]

            enc = []
            for l in self.pos[sub_idx, :]:
                added = False
                for p_idx in range(len(areas[floor_idx])):
                    if areas[floor_idx][p_idx].intersects(shapely.geometry.Point(l[0], l[1])):
                        enc += [p_idx]
                        added = True
                        break
                if not added:
                    enc += [-1]

            # find missing entries
            enc = np.array(enc)
            correct_idx = np.where(enc != -1)[0]
            shifted = enc + last_area_idx
            t = np.zeros((len(sub_idx), num_areas))
            t[np.arange(len(t))[correct_idx], shifted[correct_idx]] = 1
            targets[sub_idx, :] = t

            last_area_idx += len(areas[floor_idx])

        failed_idx = np.where(np.sum(targets, axis=1) != 1)[0]
        print("{} of {} position labels could not be assigned to any area".format(len(failed_idx), len(targets)))

        self.y = targets

    def setup_target_vector(self, area_type=None):
        if area_type is None:
            area_type = self.pr.get_param('area_type')

        # obtain classes from polygons
        # those can be constructed from:
        #   - polygon obtained from building structure via shp2geo module
        if area_type == 'fp':
            areas = self.get_polygons_from_building_structure()

        self.setup_target_vector_from_areas(areas)

        self.chosen_area_type = area_type

        return self

    def get_polygons_for_predictions(self, predictions, area_type=None, obtain_metrics=True, areas=None):
        metrics = {}

        y_true = self.get_data(self.pos, 'test')

        if area_type is None:
            area_type = self.chosen_area_type

        if areas is None:
            areas = self.dc.get_areas(area_type)

        areas_con = np.concatenate(areas, axis=0)

        polys = []
        sur_area = []
        hit_cnt = 0
        for idx, pred in enumerate(predictions):
            poly = areas_con[pred]
            point = shapely.geometry.Point(*y_true[idx, :])
            hit = poly.contains(point)
            polys += [poly]
            sur_area += [poly.area]
            if hit:
                hit_cnt += 1

        sur_area = np.array(sur_area)

        floor_pred = self.get_floor_from_predictions(predictions, areas)
        floor_pred_conv = [self.floors[fp] for fp in floor_pred]
        floor_true = self.get_data(self.floor, 'test')

        floor_acc = len(np.where(floor_pred_conv == floor_true)[0]) / len(floor_true)

        metrics['floor_ACC'] = floor_acc
        metrics['area_ACC'] = hit_cnt / len(y_true)
        metrics['area_mean'] = np.mean(sur_area)
        metrics['area_median'] = np.median(sur_area)
        metrics['deviation'] = -1

        return polys, floor_pred, metrics

    def get_floor_from_predictions(self, predictions, areas=None):
        floor_pred = np.full(len(predictions), -1)
        if areas is None:
            areas = self.dc.get_areas(self.chosen_area_type)
        curr_idx = 0
        last_idx = 0
        for floor_idx in range(self.num_floors):
            curr_idx += len(areas[floor_idx])
            mask = np.logical_and(predictions >= last_idx, predictions < curr_idx)
            sub_idx = np.where(mask)[0]

            floor_pred[sub_idx] = floor_idx # + 1

            last_idx = curr_idx

        return floor_pred

    def get_polygons_from_building_structure(self):
        if self.dc.fp_areas is None:
            self.dc.load_areas_from_building_structure()

        return self.dc.fp_areas
