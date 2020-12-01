import numpy as np
import json

class NpEncoder(json.JSONEncoder):

    """
    writes json files with np.int
    use json.dumps(fp,cls = NpEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)