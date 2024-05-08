import pickle
from ewc_utils import EWC, ewc_train, evaluate
import heapq
from tqdm import tqdm

with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_translate_instance.txt", "rb") as fp:
    ewc_race_tran = pickle.load(fp)
class Parameter:
    def __init__(self, value, layer, index):
        self.value = value
        self.index = index
        self.layer = layer
    def __lt__(self, other):
        return self.value < other.value

def count_parameters(model):
    c = 0
    for k, v in model.items():
        if len(v.shape) == 1:
            c += len(v)
        else:
            c += len(v) * len(v[0])
    return c

no_parameters = count_parameters(ewc_race_tran._precision_matrices)

heap_len = 5000
top_n_members_cp = []
print(top_n_members_cp)
progress_bar = tqdm(range(no_parameters))
for k,v  in ewc_race_tran._precision_matrices.items():
    for r, r_value in enumerate(v):
        p = None
        if len(v.shape) == 1:
            p = Parameter(r_value, k, [r])
            if len(top_n_members_cp) < heap_len:
                heapq.heappush(top_n_members_cp, p)
            else:
                if top_n_members_cp[0] < p:
                    heapq.heappushpop(top_n_members_cp, p)
            progress_bar.update(1)
        elif len(v.shape) == 2:
            for c, c_value in enumerate(r_value):
                p = Parameter(c_value, k, [r, c])
                if len(top_n_members_cp) < heap_len:
                    heapq.heappush(top_n_members_cp, p)
                    # print(p.value)
                else:
                    if top_n_members_cp[0] < p:
                        heapq.heappushpop(top_n_members_cp, p)
                progress_bar.update(1)
                
        else:
            raise ValueError

with open("/scratches/dialfs/alta/hln35/distillation/top_n_ewc_after_tran_5000.txt", "wb") as fp:
    pickle.dump(top_n_members_cp, fp)
