# Reactor-Water-Age
Annular Reactor (AR) project on water age during the COVID-19 pandemic

This repository contains scripts used to analyze all project data, which were run using Jupyter Lab, listed below: 

1. ARProject_clsm_analysis.py -- Python script for the import of .czi files and parameter calculation.

2. Water Quality Analysis for AR Water Age Project -- R script for final analysis and figure generation for data from flow cytometry, ATP, qPCR, general water quality (temp, pH, chlorine), and CLSM images. Must run the CLSM script first to generate the needed csv files.

3. ARProject_metagenomics_scripts_notebook.ipynb -- This notebook contains or generates code for metagenomics commands at the command line, beginning with read processing and ending with MAG gneration through Anvi'o. The notebook is a rough documentation of what was run for preliminary processing.

4. MASH -- coming soon

5. SCG -- coming soon

6. ARProject_MAG_analysis.ipynb -- R script for final analysis and figure generation for MAG coverage and KEGG functional pathways

This repository also contains much of the necessary raw data to run the scripts. Raw reads were deposited in the NCBI SRA. Some of the code will need to be modified to import csv files from local directories instead of from Google Sheets. 

1. clsm data?
2. water quality data
3. qPCR data
4. 
