import torch, sys
path = sys.argv[1]
ck = torch.load(path, map_location='cpu', weights_only=False)
print('TOP KEYS:', list(ck.keys()))
state = ck.get('model_state_dict', ck.get('state_dict', None))
if state is None and isinstance(ck, dict):
    # might be flat state dict
    state = ck
for k, v in list(state.items()):
    print(k, tuple(v.shape))
sys.stdout.flush()
