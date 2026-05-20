
import torch 

class Hexagon():
    def __init__(self, radius=1.0, center=(0.0, 0.0)):
        self.radius = radius
        self.center = center

    def sample(self, num_samples):
        angles = torch.linspace(0, 2 * torch.pi, steps=7)[:-1]
        points = torch.stack([self.radius * torch.cos(angles), self.radius * torch.sin(angles)], dim=1)
        points += torch.tensor(self.center)

        indices = torch.randint(0, 6, (num_samples,))
        samples = []
        for idx in indices:
            p1 = points[idx]
            p2 = points[(idx + 1) % 6]
            t = torch.rand(1)
            sample = (1 - t) * p1 + t * p2
            samples.append(sample)
        return torch.stack(samples)

    def calculate_corners(self):
        angles = torch.linspace(0, 2 * torch.pi, steps=7)[:-1]
        points = torch.stack([self.radius * torch.cos(angles), self.radius * torch.sin(angles)], dim=1)
        points += torch.tensor(self.center)
        return points
    
    def get_boundary_points(self, num_points_per_edge=100):
        angles = torch.linspace(0, 2 * torch.pi, steps=7)[:-1]
        points = torch.stack([self.radius * torch.cos(angles), self.radius * torch.sin(angles)], dim=1)
        points += torch.tensor(self.center)

        boundary_points = []
        for i in range(6):
            p1 = points[i]
            p2 = points[(i + 1) % 6]
            edge_points = torch.stack([ (1 - t) * p1 + t * p2 for t in torch.linspace(0, 1, steps=num_points_per_edge)], dim=0)
            boundary_points.append(edge_points)
        return torch.cat(boundary_points, dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hexagon = Hexagon(radius=2.0, center=(0.0, 0.0))
    samples = hexagon.sample(1000)
    corners = hexagon.calculate_corners()

    x = torch.randn(1000, 2) 

    boundary_points = hexagon.get_boundary_points(num_points_per_edge=200)

    plt.figure(figsize=(6,6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.scatter(corners[:, 0], corners[:, 1], c='r', s=100)
    plt.scatter(x[:, 0], x[:, 1], alpha=0.2, label='Base Samples (N(0,I))')
    plt.plot(boundary_points[:,0], boundary_points[:,1], 'black', label='Hexagon Boundary')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Samples from Hexagon Distribution')
    plt.legend()
    plt.show()
