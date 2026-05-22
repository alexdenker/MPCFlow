# 2D Hexagon

Minimal 2D flow-matching and MPC examples used for lightweight MPC-Flow experiments.

Typical order:

```powershell
python train_flow.py
python optimal_control.py
python mpc_receding_horizon.py
python eval.py
```

`mlp_hexagon.pt` is the saved velocity-field checkpoint. The scripts write local experiment outputs such as trajectories, figures, and `results/` under this folder.

