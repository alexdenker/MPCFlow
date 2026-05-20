
from tqdm import tqdm 

import torch
from torch.optim import Adam

import matplotlib.pyplot as plt 

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper


from model import MLP 
from dataset import Hexagon

device = "cuda"

vf = MLP(input_dim=2, time_dim=1, hidden_dim=128)
vf = vf.to(device)

print("Number of Parameters: ", sum([p.numel() for p in vf.parameters()]))

num_iters = 100000
batch_size = 512
lr=1e-4

hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())


optimizer = Adam(vf.parameters(), lr=lr)

running_mean_loss = 0.0
momentum = 0.9
for i in tqdm(range(num_iters)):
    optimizer.zero_grad()
    x_1 = hexagon.sample(batch_size).to(device)    
    x_0 = torch.randn_like(x_1).to(device)

    t = torch.rand(batch_size).to(device) 
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1) 
    loss = torch.pow( vf(path_sample.x_t,path_sample.t) - path_sample.dx_t, 2).mean()
    loss.backward()
    optimizer.step()

    running_mean_loss = momentum * running_mean_loss + (1 - momentum) * loss.item()
    if (i+1) % 500 == 0:
        print(f"Iter {i+1}, Loss: {running_mean_loss/(1-momentum**(i+1))}")

        torch.save(vf.state_dict(), 'mlp_hexagon.pt')

        with torch.no_grad():
            class WrappedModel(ModelWrapper):
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                    return self.model(x, t)

            wrapped_vf = WrappedModel(vf)

            # step size for ode solver
            step_size = 0.05

            batch_size = 512  
            times = torch.linspace(0,1,10) 
            times = times.to(device=device)

            x_init = torch.randn((batch_size, *x_1.shape[1:]), device=device)
            solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
            sol = solver.sample(time_grid=times, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False)  # sample from the model

            plt.figure(figsize=(6,6))
            plt.scatter(sol[:,0].cpu(), sol[:,1].cpu(), alpha=0.5, label='Flow Samples')
            corners = hexagon.calculate_corners()
            plt.scatter(corners[:, 0], corners[:, 1], c='r', s=100, label='Target Corners')
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Flow Matching Samples after Training')
            plt.legend()
            plt.savefig('flow_matching_hexagon_samples.png')
            plt.close()