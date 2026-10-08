# Data provenance

## Included files

- `raw/Food_Waste_national_2018_m1_v1.3.2_9b1bb41.csv` - EPA Flow-By-Sector method 1.
- `raw/Food_Waste_national_2018_m2_v1.3.2_9b1bb41.csv` - EPA Flow-By-Sector method 2; used by `analysis/train_model.py`.

The EPA catalog describes these as national 2018 food-waste flows attributed to waste-generating and waste-management sectors. Method 1 attributes estimates from the EPA Wasted Food Report; method 2 attributes estimates from the National Commercial Non-Hazardous Waste Flow-By-Sector dataset. The files were generated with FLOWSA v1.3.2. These are modeled estimates, not donation transactions or current local supply.

## Source and reuse

- EPA dataset catalog: <https://catalog.data.gov/dataset/u-s-food-waste-flows-between-sectors-2018-v1-3-2>
- Dataset DOI: <https://doi.org/10.23719/1529936>
- EPA data license and disclaimer: <https://pasteur.epa.gov/license/sciencehub-license.html>
- FLOWSA method repository: <https://github.com/USEPA/flowsa/tree/v1.3.2>

The EPA license page states that, unless otherwise specified, EPA-produced data are public domain in the United States, and it cautions users to consult the dataset metadata and assess accuracy and utility. This repository includes the source CSVs for reproducibility and credits EPA and FLOWSA.

## Citation

U.S. Environmental Protection Agency, Office of Research and Development. *U.S. Food Waste Flows Between Sectors, 2018 v1.3.2*. DOI: 10.23719/1529936.
