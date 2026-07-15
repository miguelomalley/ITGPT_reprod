# ITGPT_reprod
Messy repo for ITGPT paper reproducibility code.

This repo is meant to produce the exact tables from the paper ITGPT. If you want to train models using the same dataset we used, clone the main ITGPT repo, download the following packs from one of the many resources (e.g. itgpacks.com):

```
Fraxtil's Arrow Arrangements
Fraxtil's Beast Beats
Fraxtil's Cute Charts
Sweet Arrows and Hella Steps Vols. 1-4
Tsunamix III
```
and follow the training instructions. Optionally use the text documents we provide here for the exact paper splits by substituting them in for the identically named documents which will be created by smfiler from the main ITGPT repo.

## Usage

To reproduce the exact tables from the ITGPT paper, download the models found [here](https://drive.google.com/drive/folders/1z6W92_4uzDGZFikUOHLJK1i3V5H8Agwk?usp=drive_link) and run the appropriate eval notebooks. We provide .pkl results from DDCL/DDC/GOCT, but if you would like to run those yourself follow the instructions below.

### DDC/DDCL reprod

The appropriately named notebooks in the DDCL folder in this repo can be used to reproduce the paper table results for DDCL. Install the appropriate dependencies from [the DDCL repo](https://github.com/miguelomalley/DDCL) and run the notebooks.

### GOCT reprod

This process is extremely involved. Follow the reproducibility instructions for [the GOCT repo](https://github.com/stet-stet/goct_ismir2023/tree/main), but instead of the splits and packs they list for the fraxtil dataset, use the ones we provide above and the split from this repo. Then, save the results to a table as we do above.


