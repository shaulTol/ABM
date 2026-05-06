import numpy as np, torch, json
from transformers import CLIPModel, CLIPTokenizer
from scipy.stats import spearmanr

pe = np.load('/Users/stolk/github/ssa-paper/experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz', allow_pickle=True)
saved_da1 = np.asarray(pe['da1_eigvec']).astype(np.float64)
js = json.load(open('/Users/stolk/github/ssa-paper/experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.json'))
print('Target da1.gender =', js['da1_cosines']['gender'])

zt = np.load('/Users/stolk/github/ssa-paper/experiments/elderly_da_bootstrap/clip_da_elderly_gs7p5_fc_z_tau.npz')
z_tau = np.asarray(zt['z_tau']).astype(np.float64)
norms = np.linalg.norm(z_tau, axis=1)
print(f'z_tau norms: min={norms.min():.4f} mean={norms.mean():.4f} max={norms.max():.4f}')

tok = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
m = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').eval()

def make(fn, name):
    woman = fn('a photograph of a woman')
    man   = fn('a photograph of a man')
    a = woman - man; a /= np.linalg.norm(a) + 1e-12
    cos = float(saved_da1 @ a)
    proj_da1 = z_tau @ saved_da1
    proj_g = z_tau @ a
    r, _ = spearmanr(proj_da1, proj_g)
    print(f'  [{name}]  cos(da1, axis)={cos:+.4f}  ρ(da1·zτ, axis·zτ)={r:+.4f}')

@torch.no_grad()
def fn1(t):
    inp = tok([t], return_tensors='pt', padding=True, truncation=True)
    out = m.text_model(**inp)
    e = m.text_projection(out.pooler_output)
    return (e / e.norm(dim=-1, keepdim=True)).numpy().squeeze(0).astype(np.float64)

@torch.no_grad()
def fn2(t):
    inp = tok([t], return_tensors='pt', padding=True, truncation=True)
    out = m.text_model(**inp)
    h = out.last_hidden_state
    eos = inp['input_ids'].argmax(dim=-1)
    p = h[torch.arange(h.size(0)), eos]
    e = m.text_projection(p)
    return (e / e.norm(dim=-1, keepdim=True)).numpy().squeeze(0).astype(np.float64)

@torch.no_grad()
def fn3(t):
    inp = tok([t], return_tensors='pt', padding=True, truncation=True)
    raw = m.get_text_features(**inp)
    if torch.is_tensor(raw):
        e = raw
    else:
        e = m.text_projection(raw.pooler_output)
    return (e / e.norm(dim=-1, keepdim=True)).numpy().squeeze(0).astype(np.float64)

make(fn1, 'text_model.pooler→proj')
make(fn2, 'text_model.eos→proj')
make(fn3, 'get_text_features')
