
from tqdm import tqdm 

import torch
import os 
import matplotlib.pyplot as plt 

from model import MLP 
from dataset import Hexagon

device = "cuda"
control_scaling = 0.001

for K in range(1,20):
    save_dir = "mpc_receding_horizon/K={}/control_scaling={}/".format(K, control_scaling)
    os.makedirs(save_dir, exist_ok=True)

    vf = MLP(input_dim=2, time_dim=1, hidden_dim=128)
    vf.load_state_dict(torch.load('mlp_hexagon.pt', weights_only=True))
    vf = vf.to(device)

    hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))

    x_0 = torch.tensor([[-0.75, -0.5]], device=device)

    def euler(vf, u, x_0, T):
        x = x_0.detach().clone()
        x_ = [x]
        control_loss = 0.0

        for i in range(T.shape[0]-1):
            dt = T[i+1] - T[i]
            x = x + dt * (vf(x, T[i]) + u[i])
            x_.append(x)
            control_loss += torch.sum(u[i]**2) * dt
        x_ = torch.stack(x_, dim=0)
        return x, x_, control_loss

    corners = hexagon.calculate_corners()
    boundary_points = hexagon.get_boundary_points(num_points_per_edge=200)

    target_corner = corners[-1].to(device)

    num_timesteps = 100
    T = torch.linspace(0, 1, num_timesteps+1, device=device)

    with torch.no_grad():
        u = torch.zeros((num_timesteps, x_0.shape[0], x_0.shape[1]), device=device)
        _, xt, _ = euler(vf, u, x_0, T)
        print("Uncond trajectory xt:", xt.shape)

    control_iters = 200

    xk = x_0.clone()
    x_traj = [xk.detach()]

    control = torch.zeros((K, x_0.shape[0], x_0.shape[1]), device=device)

    all_controls = []
    for i in tqdm(range(len(T)-1)):

        # warm start, remove first control input and append zero at the end
        if i > 0:
            control = torch.cat([control[1:], torch.zeros((1, x_0.shape[0], x_0.shape[1]), device=device)], dim=0)
        if K == 1:
            control = torch.zeros((K, x_0.shape[0], x_0.shape[1]), device=device)

        control = torch.nn.Parameter(control)
        optimiser = torch.optim.Adam([control], lr=0.1)

        T_opt = torch.linspace(T[i], 1, K+1, device=device)
        for k in range(control_iters):
            optimiser.zero_grad()
            
            # do a single euler step 
            xT, _, control_loss = euler(vf, control, xk, T_opt)

            data_loss = 0.075 * 0.5 * torch.sum((xT - target_corner)**2)
            loss = control_scaling * control_loss + data_loss
            loss.backward()

            optimiser.step()
        
        # one step with the updated control 
        with torch.no_grad():
            dt = T[i+1] - T[i]
            t_in = torch.tensor(T[i], device=device)
            xk = xk + dt *(vf(xk, t_in) + control[0])

            x_traj.append(xk.detach())  
            all_controls.append(control[0].detach().cpu())

    x_traj = torch.stack(x_traj, dim=0)
    all_controls = torch.stack(all_controls, dim=0)
    torch.save(x_traj.cpu(), f"{save_dir}/mpc_traj_K={K}_controliters={control_iters}.pt")
    torch.save(all_controls.cpu(), f"{save_dir}/mpc_controls_K={K}_controliters={control_iters}.pt")

    try:
        # 2DHexagon/global_control_adam/control_scaling=0.05
        x_traj_global = torch.load(f"global_control_adam/control_scaling={control_scaling}/global_controlled_trajectory.pt")

        print("Global trajectory shape: ", x_traj_global.shape, " MPC trajectory shape: ", x_traj.shape)

        global_terminal_loss = torch.sum((x_traj_global[-1,:,:] - target_corner.cpu())**2)
        print("Global Terminal Loss: ", global_terminal_loss.item())

        mpc_terminal_loss = torch.sum((x_traj[-1,:,:] - target_corner)**2)
        print("MPC Terminal Loss: ", mpc_terminal_loss.item())

        print("Difference of trajectories: ", torch.sum((x_traj.cpu() - x_traj_global)**2).item())
    except FileNotFoundError:
        print("Global trajectory file not found.")

        

    plt.figure(figsize=(6,6))
    plt.plot(boundary_points[:,0], boundary_points[:,1], c='grey', linewidth=3)
    plt.scatter(corners[:, 0], corners[:, 1], c='black', s=60, zorder=2)
    plt.scatter(target_corner[0].cpu(), target_corner[1].cpu(), c='red', s=90, zorder=3, label='Target Corner')
    plt.plot(xt[:,0, 0].cpu(), xt[:,0, 1].cpu(), label='Uncond Trajectory', c='grey', linestyle='--')
    plt.plot(x_traj[:,0, 0].cpu(), x_traj[:,0, 1].cpu(), label='Controlled Trajectory')
    if 'x_traj_global' in locals():
        plt.plot(x_traj_global[:,0, 0].cpu(), x_traj_global[:,0, 1].cpu(), label='Global Controlled Trajectory', linestyle=':')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('MPC Trajectory')
    plt.legend()
    plt.savefig(f'{save_dir}/mpc_trajectory_K={K}_controliters={control_iters}.png')
    plt.close()