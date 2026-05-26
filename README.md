# MPC-Flow

Research code for **Solving Inverse Problems with Flow-based Models via Model Predictive Control** (accepted to **ICML 2026**).

This repository is organized by experiment family:

- [`FLUX2/`](FLUX2/): image inverse-problem demos with the large vision-language model FLUX.2 [dev].
- [`2DHexagon/`](2DHexagon/): lightweight 2D flow-matching and MPC examples on a hexagon distribution.

The large-scale image experiments from the paper use the FLUX.2 code path. Start there if you want to run style transfer, luminance-constrained colorization or super-resolution:

```powershell
cd FLUX2
conda env create -f environment.yml
conda activate mpcflow
python -c "import flux2.mpc; print('FLUX2 imports ok')"
```
See [`FLUX2/README.md`](FLUX2/README.md) for FLUX.2-specific setup, Hugging Face model access, CLI commands, and notebook instructions.

## Paper

- arXiv: https://arxiv.org/abs/2601.23231

Authors:
George Webber
*<sup>1</sup>, Alexander Denker\*<sup>2</sup>, Riccardo Barbano<sup>3</sup>, and Andrew J. Reader<sup>1</sup>

1. School of Biomedical Engineering and Imaging Sciences, King's College London
2. Helmholtz Imaging, Deutsches Elektronen-Synchrotron DESY, Germany
3. Department of Computer Science, University College London

\* Equal contribution.

## Citation

```bibtex
@article{webber2026solving,
  title={Solving Inverse Problems with Flow-based Models via Model Predictive Control},
  author={Webber, George and Denker, Alexander and Barbano, Riccardo and Reader, Andrew J},
  journal={arXiv preprint arXiv:2601.23231},
  year={2026}
}
```

## License

Repository code license: [`LICENSE.md`](LICENSE.md).

FLUX.2 model weights are not redistributed in this repository. FLUX.2 model use is governed separately by the model license in [`FLUX2/model_licenses/LICENSE-FLUX-DEV`](FLUX2/model_licenses/LICENSE-FLUX-DEV).
