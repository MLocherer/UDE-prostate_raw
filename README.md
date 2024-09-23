# mpUDE-prostate
A new multiparametric MRI (mp-MRI) prostate dataset called UDE-prostate (UDEp) whose raw data was collected and segmented by our partner at the university Duisburg-Essen. It contains exams in the Digital Imaging and Communications in Medicine (DICOM) format and segmentation labels in the Medical Imaging Interaction Toolkit (MITK) format for apparent diffusion coeffient (ADC), T2 WI and dynamic contrast-enhanced (DCE) with 37, 29 and 30 exams respectively. For 22 of the exams all the aforementioned modalities are available.

The UDEp dataset is segmented into WG (whole gland), PZ (peripheral zone) and the PLES (prostate lesion classes PLES1, PLES2 and PLES3) which depict different lesions within the prostate. Since all PLES classes encode the same information, namely lesion, they are unified to the single prostate cancer class PCa.

## Download the archive
If you just wish to extract the dataset please download it from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12817071.svg)](https://doi.org/10.5281/zenodo.12817071).

Watch out version 3 is the complete raw dataset. Version 1 and 2 are already modified datasets.

## Download MITK
Download [The Medical Imaging Interaction Toolkit (MITK)](https://www.mitk.org/) for your operating system.

## Dataset overview
An extensive dataset description can be found in the spread-sheet `mpUDE-prostate.ods`. For each patient the modalities and image shapes are given.

## Extract the archive
### Using a shell script 
If you just want to look at the image files using [MITK](https://www.mitk.org/) you can use the shell script `unzip_UDEp.sh`. This will just extract all the archives. The script expects you to have a folder called `mpUDE-prostate` in which you store the archive files. Moreover, you have to also download the file `mpUDE-prostate.txt` that contains a list of all archives.

### Using the `ds_ude_mitk_process_a.py` file
This file was orgininally created to extract all archives based on the MRI modality and shape information, hence, you can exactly specify which candidates you would like to retrieve based on the dataset description in `mpUDE-prostate.ods`. Follow the function descriptions alongside this file or just use the example usage from line 1047 onwards. 
To extract the dataset you need to use the `mitk_process` function. This function will store the dataset in a pickle file.
