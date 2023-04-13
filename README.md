# Reactor-Water-Age
Annular Reactor (AR) project on water age during the COVID-19 pandemic

This repository contains scripts used to analyze all project data, which were run using Jupyter Lab, listed below: 

1. ARProject_clsm_analysis.py -- Python script for the import of .czi files and parameter calculation.

2. ARProject_water_quality_analysis -- R script for final analysis and figure generation for data from flow cytometry, ATP, qPCR, general water quality (temp, pH, chlorine), and CLSM images. Must run the CLSM script first to generate the needed csv files.

3. ARProject_metagenomics_scripts_notebook.ipynb -- This notebook contains or generates code for metagenomics commands at the command line, beginning with read processing and ending with MAG gneration through Anvi'o. The notebook is a rough documentation of what was run for preliminary processing.

4. ARProject_MASH_analysis.ipynb -- R code to process and plot MASH distances of metagenomic reads.

5. ARProject_scg_taxonomy.ipynb -- Python script for SCG, must be run before the corresponding scg R script.

5. ARProject_RPS2_scg_analysis.ipynb -- R code to analyze taxonomy using the single copy gene RPS2. Must run the python script for SCG analysis first to generate the data csv. 

6. ARProject_MAG_analysis.ipynb -- R script for final analysis and figure generation for MAG coverage and KEGG functional pathways

This repository also contains much of the necessary raw data to run the scripts. Raw reads were deposited in the NCBI SRA. Some of the code will need to be modified to import csv files from local directories instead of from Google Sheets. 

1. ARProject_arbf_data.csv -- Main data spreadsheet for water quality data
2. ARProject_qPCR_data -- Data spreadsheet with raw qPCR data
3. ARProject_clsm_file_key -- List of files used for CLSM analysis including metadata columns to help read in raw data. The raw .czi files were too large to be provided but can be provided upon request: hannahghealy@gmail.com.
4. ARProject_id2code -- Key to translate sample names across versions
5. ARProject_binnames -- Key to translate bin names across versions
