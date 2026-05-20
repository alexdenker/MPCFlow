"""
Solving the optimal control problem for the 2D hexagon 
using a simple Euler integration scheme and Adam optimization.
"""

from tqdm import tqdm 
import torch
import matplotlib.pyplot as plt 
import os 

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from model import MLP 
from dataset import Hexagon

device = "cuda"

vf = MLP(input_dim=2, time_dim=1, hidden_dim=128)
vf.load_state_dict(torch.load('mlp_hexagon.pt', weights_only=True))
vf = vf.to(device)

hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))

path = AffineProbPath(scheduler=CondOTScheduler())

x_0 = torch.tensor([[-0.75, -0.5]], device=device)
num_timesteps = 100 

u = torch.zeros((num_timesteps, x_0.shape[0], x_0.shape[1]), device=device)
u = torch.nn.Parameter(u)

def euler(vf, u, x_0, dt):
    x = x_0
    x_ = [x.detach()]
    control_loss = 0.0
    for t in range(num_timesteps):
        x = x + dt * (vf(x, torch.tensor(t * dt, device=device)) + u[t])
        x_.append(x.detach())
        control_loss += torch.sum(u[t]**2) * dt
    x_ = torch.stack(x_, dim=0)
    return x, x_, control_loss


optim = torch.optim.Adam([u], lr=1e-2)
num_iters = 3000
dt = 1.0 / num_timesteps

corners = hexagon.calculate_corners()
boundary_points = hexagon.get_boundary_points(num_points_per_edge=200)

target_corner = corners[-1].to(device)

control_scaling = 0.001
save_dir = "global_control_adam/control_scaling={}/".format(control_scaling)
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    _, xt, _ = euler(vf, torch.zeros_like(u), x_0, dt)
    print("Uncond trajectory xt:", xt.shape)

    plt.figure(figsize=(6,6))
    plt.plot(boundary_points[:,0], boundary_points[:,1], c='grey', linewidth=3)
    plt.scatter(corners[:, 0], corners[:, 1], c='black', s=60, zorder=2)
    plt.scatter(target_corner[0].cpu(), target_corner[1].cpu(), c='red', s=90, zorder=3, label='Target Corner')
    plt.plot(xt[:,0, 0].cpu(), xt[:,0, 1].cpu(), label='Uncond Trajectory')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Initial Trajectory before Control')
    plt.legend()
    plt.show()

    torch.save(xt.cpu(), 'uncontrolled_traj.pt')

for it in tqdm(range(num_iters)):
    optim.zero_grad()
    x_T, x_traj, control_loss = euler(vf, u, x_0, dt)
    loss = 0.075 * 0.5 * torch.sum((x_T - target_corner)**2) + control_scaling * control_loss
    loss.backward()
    optim.step()

    if (it + 1) % 500 == 0:
        print(f"Iter {it+1}, Loss: {loss.item():.4f}, Control Loss: {control_loss.item():.4f}")

        plt.figure(figsize=(6,6))
        plt.plot(boundary_points[:,0], boundary_points[:,1], c='grey', linewidth=3)
        plt.scatter(corners[:, 0], corners[:, 1], c='black', s=60, zorder=2)
        plt.scatter(target_corner[0].cpu(), target_corner[1].cpu(), c='red', s=90, zorder=3, label='Target Corner')
        plt.plot(xt[:,0, 0].cpu(), xt[:,0, 1].cpu(), label='Uncond Trajectory', c='grey', linestyle='--')
        plt.plot(x_traj[:,0, 0].cpu(), x_traj[:,0, 1].cpu(), label='Controlled Trajectory')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Controlled Trajectory after Training')
        plt.legend()
        plt.savefig(f'{save_dir}/controlled_trajectory_iter_{it+1}.png')
        plt.close()


torch.save(u, f'{save_dir}/optimal_control_inputs.pt')
torch.save(x_traj.cpu(), f'{save_dir}/global_controlled_trajectory.pt')