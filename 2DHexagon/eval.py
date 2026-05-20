import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os 

from dataset import Hexagon
from model import MLP

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'text.usetex': False,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 1.8,
    'axes.linewidth': 1.0,
})

colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
global_color = '#CC79A7'
boundary_color = '#333333'
delta_t_color = '#8c564b'


control_scaling = 0.01

x_traj_global = torch.load(f"global_control_adam/control_scaling={control_scaling}/global_controlled_trajectory.pt")

K = 19
x_traj = []
for k in range(1, K+1):
    x_traj.append(torch.load(f"mpc_receding_horizon/K={k}/control_scaling={control_scaling}/mpc_traj_K={k}_controliters=100.pt"))

traj_diff = []
for k in range(K):
    diff = torch.sum((x_traj[k].cpu() - x_traj_global)**2).item()
    traj_diff.append(diff)


save_dir = "results/"
os.makedirs(save_dir, exist_ok=True)

hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))

corners = hexagon.calculate_corners()
boundary_points = hexagon.get_boundary_points(num_points_per_edge=200)

target_corner = corners[-1]

terminal_loss_list = []
global_terminal_loss = torch.sum((x_traj_global[-1,:,:] - target_corner)**2).item()
print("Global terminal point: ", x_traj_global[-1,:,:], " target corner: ", target_corner, " terminal loss: ", global_terminal_loss)
for k in range(K):
    terminal_loss = torch.sum((x_traj[k][-1,:,:] - target_corner)**2).item()
    print("K=", k+1, " terminal point: ", x_traj[k][-1,:,:], " target corner: ", target_corner, " terminal loss: ", terminal_loss)
    terminal_loss_list.append(terminal_loss)
terminal_error = [abs(loss - global_terminal_loss) for loss in terminal_loss_list]


vf = MLP(input_dim=2, time_dim=1, hidden_dim=128)
vf.load_state_dict(torch.load("mlp_hexagon.pt", map_location="cpu", weights_only=True))
vf.eval()
grid = torch.linspace(-2.2, 2.2, 21)
gx, gy = torch.meshgrid(grid, grid, indexing="xy")
pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
with torch.no_grad():
    t = torch.full((pts.shape[0], 1), 0.5)
    vel = vf(pts, t)
vx = vel[:, 0].reshape(gx.shape).cpu().numpy()
vy = vel[:, 1].reshape(gy.shape).cpu().numpy()


def plot_scene(ax, show_legend=True, show_frame=False):
    ax.quiver(
        gx.cpu().numpy(),
        gy.cpu().numpy(),
        vx,
        vy,
        color=(0.5, 0.5, 0.5, 0.18),
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.0022,
        zorder=0,
    )
    ax.plot(boundary_points[:,0], boundary_points[:,1], c=boundary_color, linewidth=2.5, alpha=0.8, zorder=1)

    ax.plot(x_traj_global[:,0,0].cpu(), x_traj_global[:,0,1].cpu(), label='Global optimum', 
            linewidth=2.5, c=global_color, linestyle='-', zorder=4)
    
    k_values = [0,1,2,3, 9]
    for idx, k in enumerate(k_values):
        ax.plot(x_traj[k][:,0,0].cpu(), x_traj[k][:,0,1].cpu(), 
                label=f'MPC-RHC $K={k+1}$', linewidth=2.0, c=colors[idx], alpha=0.85, zorder=3)

    start_pt = x_traj[0][0].reshape(-1).cpu()
    ax.scatter(start_pt[0], start_pt[1], c="grey", s=25, linewidths=1.0, zorder=7)

    ax.scatter(corners[:, 0], corners[:, 1], c='black', s=60, marker='o', 
               edgecolors='white', linewidths=0.8, zorder=5)
    ax.scatter(target_corner[0].cpu(), target_corner[1].cpu(), c='red', s=150, 
               marker='*', edgecolors='darkred', linewidths=1.2, zorder=6, label='Target')

    if show_legend:
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', 
                  framealpha=0.95, shadow=False, ncol=2, fontsize=8)
    if show_frame:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")
    ax.set_aspect('equal', adjustable='box')


# Figure 1: Trajectory Comparison
fig1 = plt.figure(figsize=(3.5, 3.5))
ax1 = fig1.add_subplot(111)
plot_scene(ax1, show_legend=True)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(f'{save_dir}/mpc_trajectories.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/mpc_trajectories.png', format='png', bbox_inches='tight', dpi=300)

# Figure 1b:  Trajectory Comparison (Zoomed)
fig1b = plt.figure(figsize=(6, 4))
ax1b = fig1b.add_subplot(111)
plot_scene(ax1b, show_legend=False)
zoom_center = [-0.5, -1.0]
zoom_radius = 1.8
y_ratio = 0.5
ax1b.set_xlim(zoom_center[0] - zoom_radius, zoom_center[0] + zoom_radius)
ax1b.set_ylim(zoom_center[1] - zoom_radius*y_ratio, zoom_center[1] + zoom_radius*y_ratio)
for spine in ax1b.spines.values():
    spine.set_edgecolor("#d62728")
    spine.set_linewidth(1.2)

inset_size = "60%"
inset_ax = inset_axes(ax1b, width=inset_size, height=inset_size, loc="lower left",
                      borderpad=0.0, bbox_to_anchor=(-0.0, 0.1, 1.0, 1.0),
                      bbox_transform=ax1b.transAxes)
plot_scene(inset_ax, show_legend=False, show_frame=True)
inset_ax.set_xlim(-2.5, 2.5)
inset_ax.set_ylim(-2.5, 2.5)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.patch.set_facecolor("white")
inset_ax.patch.set_alpha(1.0)
for spine in inset_ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.6)

zoom_x0 = zoom_center[0] - zoom_radius
zoom_y0 = zoom_center[1] - zoom_radius

plt.tight_layout()
handles, labels = ax1b.get_legend_handles_labels()
fig1b.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3,
             frameon=True, fancybox=False, edgecolor='black', framealpha=0.95, shadow=False, fontsize=10)
fig1b.subplots_adjust(top=0.82)
plt.savefig(f'{save_dir}/mpc_trajectories_zoom.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/mpc_trajectories_zoom.png', format='png', bbox_inches='tight', dpi=300)

# Figure 2: Convergence Analysis
fig2 = plt.figure(figsize=(3.5, 2.8))
ax2 = fig2.add_subplot(111)

ax2.semilogy(range(1, K+1), traj_diff, marker='o', markersize=6, 
             linewidth=2.0, color='#0072B2', markerfacecolor='#0072B2', 
             markeredgecolor='white', markeredgewidth=1.0)
ax2.set_xlabel('Discretisation Steps $K$', fontsize=9)
ax2.set_ylabel('Distance to Global Trajectory', fontsize=9)
ax2.set_xticks(range(1, K+1, 3))
ax2.grid(True, which="both", ls="--", linewidth=0.6, alpha=0.5, color='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{save_dir}/mpc_convergence_log.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/mpc_convergence_log.png', format='png', bbox_inches='tight', dpi=300)

# Figure 2b: Convergence analysis (inset)
fig2 = plt.figure(figsize=(3.5, 2.8))
ax2 = fig2.add_subplot(111)

ax2.plot(range(1, K+1), traj_diff, marker='o', markersize=6, 
             linewidth=2.0, color='#0072B2', markerfacecolor='#0072B2', 
             markeredgecolor='white', markeredgewidth=1.0)
ax2.set_xlabel('Discretisation Steps $K$', fontsize=9)
ax2.set_ylabel('Distance to Global Trajectory', fontsize=9)
ax2.set_xticks(range(1, K+1, 3))
ax2.grid(True, which="both", ls="--", linewidth=0.6, alpha=0.5, color='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{save_dir}/mpc_convergence.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/mpc_convergence.png', format='png', bbox_inches='tight', dpi=300)


# Figure 3: Terminal Objective Error
fig3 = plt.figure(figsize=(3.5, 2.8))
ax3 = fig3.add_subplot(111)

ax3.semilogy(range(1, K+1), terminal_error, marker='s', markersize=5,
             linewidth=1.8, color='#D55E00', markerfacecolor='#D55E00',
             markeredgecolor='white', markeredgewidth=0.8)
ax3.set_xlabel('Discretisation Steps $K$', fontsize=9)
ax3.set_ylabel('Terminal Objective Error', fontsize=9)
ax3.set_xticks(range(1, K+1, 3))
ax3.grid(True, which="both", ls="--", linewidth=0.6, alpha=0.5, color='gray')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{save_dir}/mpc_terminal_error.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/mpc_terminal_error.png', format='png', bbox_inches='tight', dpi=300)
plt.show()
