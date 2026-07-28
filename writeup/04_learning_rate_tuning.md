# Problem (learning_rate_tuning): Tuning the learning rate (1 point)

## Setup

We ran the decaying-SGD example from the handout (update rule: `θ_{t+1} = θ_t − (α/√(t+1)) ∇L(θ_t)`)
on the toy problem `loss = (weights**2).mean()` with `weights = torch.nn.Parameter(5 * torch.randn((10, 10)))`,
for 10 training iterations at learning rates `1e1`, `1e2`, and `1e3`. The seed is fixed
(`torch.manual_seed(0)`) so all three runs start from the same initialization. Run on CPU with
Python 3 and torch 2.1.1.

## Code

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def run(lr: float, num_steps: int = 10, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for t in range(num_steps):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        losses.append(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
    return losses


for lr in [1e1, 1e2, 1e3]:
    print(lr, run(lr))
```

## Results

Loss at each of the 10 steps (seed 0, identical initialization for all runs):

| Step | lr = 1e1     | lr = 1e2     | lr = 1e3     |
|-----:|-------------:|-------------:|-------------:|
| 0    | 2.627140e+01 | 2.627140e+01 | 2.627140e+01 |
| 1    | 1.681370e+01 | 2.627140e+01 | 9.483977e+03 |
| 2    | 1.239434e+01 | 4.507460e+00 | 1.638032e+06 |
| 3    | 9.697248e+00 | 1.078737e-01 | 1.822136e+08 |
| 4    | 7.854772e+00 | 1.102677e-16 | 1.475930e+10 |
| 5    | 6.512506e+00 | 1.229002e-18 | 9.314807e+11 |
| 6    | 5.492434e+00 | 4.138482e-20 | 4.781918e+13 |
| 7    | 4.693441e+00 | 2.465322e-21 | 2.057386e+15 |
| 8    | 4.053156e+00 | 2.114913e-22 | 7.583084e+16 |
| 9    | 3.530749e+00 | 2.349903e-23 | 2.435013e+18 |

## Answer (deliverable)

With lr = 1e1 the loss decays steadily but slowly (26.3 → 3.5 over 10 steps), with lr = 1e2 it converges much faster — after one overshooting step it collapses to essentially zero (~1e-23) by step 9 — and with lr = 1e3 training diverges, the loss exploding monotonically to ~2.4e18.
