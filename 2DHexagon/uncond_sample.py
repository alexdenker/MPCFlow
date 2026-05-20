

import torch
import matplotlib.pyplot as plt 

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from model import MLP 
from dataset import Hexagon

device = "cuda"

vf = MLP(input_dim=2, time_dim=1, hidden_dim=128)
vf.load_state_dict(torch.load('mlp_hexagon.pt', weights_only=True))
vf = vf.to(device)

hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))
path = AffineProbPath(scheduler=CondOTScheduler())

x_0 = torch.randn(1024, 2).to(device)  

with torch.no_grad():
    class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return self.model(x, t)

    wrapped_vf = WrappedModel(vf)

    times = torch.linspace(0,1,100) 
    step_size = times[1] - times[0]
    times = times.to(device=device)

    solver = ODESolver(velocity_model=wrapped_vf)
    sol = solver.sample(time_grid=times, 
                        x_init=x_0, 
                        method='euler', 
                        step_size=step_size, 
                        return_intermediates=True)  

    corners = hexagon.calculate_corners()
    boundary_points = hexagon.get_boundary_points(num_points_per_edge=200)

    plt.figure(figsize=(6,6))
    plt.plot(boundary_points[:,0], boundary_points[:,1], c='grey', linewidth=3)
    plt.scatter(sol[-1,:,0].cpu(), sol[-1,:,1].cpu(), alpha=0.5, label='Flow Samples')
    plt.scatter(corners[:, 0], corners[:, 1], c='r', s=100, zorder=2)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Flow Matching Samples after Training')
    plt.legend()
    plt.show()