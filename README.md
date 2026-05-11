# Solving Inverse Problems with Flow-based Models via Model Predictive Control

Official implementation of the paper  
📄 *Solving Inverse Problems with Flow-based Models via Model Predictive Control*  
by [**George Webber**](https://george-webber.com/), [**Alexander Denker**](https://alexdenker.github.io/), **Riccardo Barbano**, **Andrew J. Reader**


> Flow-based generative models provide strong unconditional priors for inverse problems, but guiding their dynamics for conditional generation remains challenging. Recent work casts training-free conditional generation in flow models as an optimal control problem; however, solving the resulting trajectory optimisation is computationally and memory intensive, requiring differentiation through the flow dynamics or adjoint solves.
>
>We propose **MPC-Flow**, a model predictive control framework that formulates inverse problem solving with flow-based generative models as a sequence of control sub-problems, enabling practical optimal control-based guidance at inference time. We provide theoretical guarantees linking MPC-Flow to the underlying optimal control objective and show how different algorithmic choices yield a spectrum of guidance algorithms, including regimes that avoid backpropagation through the generative model trajectory.
>
>We evaluate MPC-Flow on benchmark image restoration tasks, spanning linear and non-linear settings such as in-painting, deblurring, and super-resolution, and demonstrate strong performance and scalability to massive state-of-the-art architectures via training-free guidance of FLUX.2 (32B) in a quantised setting on consumer hardware.

---


## Paper

**arXiv:** https://arxiv.org/abs/2601.23231


If you find this work useful, please cite:

```bibtex
@article{webber2026solving,
  title={Solving Inverse Problems with Flow-based Models via Model Predictive Control},
  author={Webber, George and Denker, Alexander and Barbano, Riccardo and Reader, Andrew J},
  journal={arXiv preprint arXiv:2601.23231},
  year={2026}
}
```