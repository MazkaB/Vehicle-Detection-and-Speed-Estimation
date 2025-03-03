# vehicle_counter.py

class VehicleCounter:
    def __init__(self):
        self.total_count = 0
        self.class_count = {}
        self.seen_ids = {}  # tracker_id -> class_name

    def increment_class_count(self, tracker_id, cls_name):
        if tracker_id not in self.seen_ids:
            self.seen_ids[tracker_id] = cls_name
            self.total_count += 1
            if cls_name not in self.class_count:
                self.class_count[cls_name] = 0
            self.class_count[cls_name] += 1
        else:
            old_class = self.seen_ids[tracker_id]
            if old_class != cls_name:
                # Optional: handle reclassification logic
                pass
