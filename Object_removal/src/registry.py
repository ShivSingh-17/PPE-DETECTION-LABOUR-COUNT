


class ObjectRegistry:

    def __init__(self):
        self.data = {}

    def update(self, obj_id, payload):

        if obj_id not in self.data:
            self.data[obj_id] = payload
        else:
            self.data[obj_id].update(payload)

    def mark_missing(self, obj_id):
        if obj_id in self.data:
            self.data[obj_id]["missing"] += 1

    def reset_missing(self, obj_id):
        if obj_id in self.data:
            self.data[obj_id]["missing"] = 0

    def remove(self, obj_id):
        if obj_id in self.data:
            del self.data[obj_id]