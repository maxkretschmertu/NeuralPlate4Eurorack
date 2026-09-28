from pathlib import Path
import numpy as np
import torch

from model import PlateNet

N_MODES, STEPS, BATCH, PAIRS, N_FREQS = 256, 30000, 16, 32, 256
DAMPING, FREQ_WEIGHT = 0.03, 10.0
ROOT = Path(__file__).parent
DATA = np.load(ROOT / "plate_dataset.npz")
MODEL_PATH = ROOT.parent / "models" / "plate_nn.pt"
MODEL_PATH.parent.mkdir(exist_ok=True)

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = lambda x: torch.as_tensor(x, device=device)

geometry = T(np.column_stack((DATA["morphs"], DATA["aspects"])))
factors, points, gains = T(DATA["factors"]), T(DATA["points"]), T(DATA["point_gains"])
train, val = T(DATA["train_indices"]), T(DATA["val_indices"])

model = PlateNet(N_MODES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
omega = torch.logspace(
    np.log10(float(factors[train].min()) * 0.9),
    np.log10(float(factors[train].max()) * 1.05),
    N_FREQS, device=device,
)


def response(mu, residues):
    mu, w = mu[:, None, :, None], omega[None, None, None, :]
    return (residues[..., None] / (mu**2 - w**2 + 1j * 2 * DAMPING * mu * w)).sum(2)


def relative_mse(pred, target):
    return (pred - target).abs().square().sum() / target.abs().square().sum().clamp_min(1e-8)


def pair_prediction(idx, i, j):
    b = torch.arange(len(idx), device=device)[:, None]
    p, phi = points[idx], gains[idx]
    z = model.encode_geometry(geometry[idx])
    log_mu = model.predict_log_factors(z)
    pred = model.predict_gains(z, p[b, i]) * model.predict_gains(z, p[b, j])
    true = phi[b, i] * phi[b, j]
    return log_mu, log_mu.exp(), pred, true


torch.manual_seed(1234)
val_i = torch.randint(points.shape[1], (len(val), PAIRS), device=device)
val_j = torch.randint(points.shape[1], (len(val), PAIRS), device=device)
torch.manual_seed(0)


@torch.no_grad()
def validate():
    _, mu, pred, true = pair_prediction(val, val_i, val_j)
    modal = ((mu - factors[val]).abs() / factors[val]).mean() * 100
    transfer = relative_mse(response(mu, pred), response(factors[val], true)).sqrt() * 100
    return modal.item(), transfer.item()


print(f"Device: {device}\nTrain geometries: {len(train)}\nValidation geometries: {len(val)}")
best = float("inf")

for step in range(1, STEPS + 1):
    idx = train[torch.randint(len(train), (BATCH,), device=device)]
    i = torch.randint(points.shape[1], (BATCH, PAIRS), device=device)
    j = torch.randint(points.shape[1], (BATCH, PAIRS), device=device)
    j[:, :PAIRS // 4] = i[:, :PAIRS // 4]

    log_mu, mu, pred, true = pair_prediction(idx, i, j)
    loss_f = (log_mu - factors[idx].log()).square().mean()
    loss_h = relative_mse(response(mu, pred), response(factors[idx], true))
    loss = FREQ_WEIGHT * loss_f + min(1.0, step / 1250.0) * loss_h

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 250 == 0:
        modal, transfer = validate()
        print(f"{step:5d}/{STEPS} | loss={loss.item():.4f} | modal={modal:.2f}% | transfer={transfer:.2f}%")
        if transfer < best:
            best = transfer
            torch.save(model.state_dict(), MODEL_PATH)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
modal, transfer = validate()
print(f"\nModal mean error: {modal:.3f}%\nTransfer RMS:     {transfer:.3f}%\nSaved: {MODEL_PATH}")
