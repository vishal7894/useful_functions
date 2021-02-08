import numpy as np
import json

# copy this code block and paste in 
# the file where we get the error in json.dumps
# where int64 is causing issues


# while using json.dumps(<json_obj>, indent=<4>, cls=NpEncoder)

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
