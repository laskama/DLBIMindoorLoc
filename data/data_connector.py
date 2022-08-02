

class DatasetConnector:

    def __init__(self):
        self.rss = None
        self.pos = None
        self.floor = None
        self.floors = None
        self.num_floors = None

        self.walls_h = None
        self.walls_v = None
        self.walls_ls = None

        # polygon obtained from building structure
        # contains list of polygon per floor
        self.fp_areas = None

        self.floorplan_width = None
        self.floorplan_height = None

        self.split_indices = None

    def load_dataset(self):
        pass

    def load_walls(self):
        pass

    def load_areas_from_building_structure(self):
        raise NotImplementedError

    def get_dataset_identifier(self):
        raise NotImplementedError

    def get_areas(self, area_type='fp'):
        if area_type == 'fp':
            return self.fp_areas
        else:
            return None
