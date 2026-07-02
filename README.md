# SDHP

## Required Packages

| Package                     | Tested Version |
| --------------------------- | -------------: |
| Python                      |         3.8.19 |
| PyTorch (`torch`)           |          2.2.2 |
| TorchVision (`torchvision`) |         0.17.2 |
| NumPy (`numpy`)             |         1.24.3 |
| SciPy (`scipy`)             |          1.9.3 |
| Matplotlib (`matplotlib`)   |          3.7.5 |
| Pillow (`Pillow`)           |         10.2.0 |

## Running Experiments
Run the three experiments from the project root:
```bash
python hyper_cleaning/hyper_cleaning_SDHP.py
python PID/PID_SDHP.py
python quantile_Huber/bilevel_qh_mlp_in_domain.py
