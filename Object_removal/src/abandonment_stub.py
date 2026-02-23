


import time

class AbandonmentStub:

    def __init__(self):
        self.static_memory = {}

    def check_abandoned(self, obj_id):

        if obj_id not in self.static_memory:
            self.static_memory[obj_id] = time.time()

        elapsed = time.time() - self.static_memory[obj_id]

        if elapsed > 300:
            return True

        return False