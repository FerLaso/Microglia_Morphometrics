# Microglia\_Morphometrics



This notebook presents a fully reproducible and universally applicable pipeline for quantitative microglial morphometric analysis. Although the workflow is technically compatible with any single-cell dataset, it has been specifically optimized and documented for microglia, the resident immune cells of the central nervous system.



Microglial morphology provides relevant information about functional state, neuroimmune activity, and tissue homeostasis. However, quantitative morphometric analysis remains inconsistent across laboratories and often relies on partially manual workflows, limited descriptor sets, and computational approaches with insufficient validation. Here, this work presents a fully automated, modular, and reproducible Python-based pipeline for multiscale microglial morphometry. The framework integrates dataset loading, preprocessing, segmentation, skeletonization, fractal dimension, lacunarity, GLCM texture metrics, multiscale fractal spectrum analysis, graph‑based descriptors, Sholl profiling, adaptive statistics, PCA/UMAP, Random Forest classification, and SHAP interpretability.





# **Main Modules**



0\. Synthetic microglia generator.



1.Universal dataset loader.



2\. Core image‑processing functions.



3\. Preprocessing, segmentation, skeletonization, and fractal dimension.



4\. Lacunarity analysis.



5\. GLCM texture extraction.



6\. Multiscale Fractal Spectrum.



7\. Graph‑based morphometric analysis.



8\. Sholl analysis.



9\. Adaptive statistical testing.



10\. Dimensionality reduction, classification, and interpretability.





# **Inputs**



Image folders organized either flat or by subfolders/groups.



Common formats: PNG, JPG/JPEG, TIF/TIFF.



Preprocessing parameters defined within the script.





# **Outputs**



CSV and XLSX tables.



PNG figures for quality control and results.



Integrated morphometric matrix.



Statistical and post‑hoc results.



PCA, UMAP, Random Forest feature importance, and SHAP values.





# **Key Features**



Modular end‑to‑end integration of multiple morphometric domains.



Synthetic microglia generator for benchmarking.



Automated dual CSV/XLSX export.



AI‑ready descriptor matrix.



Adaptive statistics with FDR correction.



SHAP‑based interpretability.



Functional documentation oriented toward biomedical users.





# **Quick Start — Download and Run**



**Microglia Morphometrics v1.0.0** is available as a pre-packaged desktop application for Windows, Linux and macOS. Users can download the corresponding release without compiling the Python source code.



### Step 1. Download the application



* Visit the official **releases section**.



* Select the package corresponding to your operating system:



Operating system	             Download

Windows 64-bit	             -->     Microglia-Morphometrics-v1.0.0-windows-x64.zip

Ubuntu Linux 22.04 x86-64    -->     Microglia-Morphometrics-v1.0.0-ubuntu-22.04-x86\_64.tar.gz

macOS (Apple Silicon)	     -->     Microglia-Morphometrics-v1.0.0-macos-arm64.zip

macOS (Intel)	             -->     Microglia-Morphometrics-v1.0.0-macos-x86\_64.zip



### Step 2. Extract and launch



1. Download and extract the appropriate archive into a folder on your computer.



2\. Open the extracted directory and launch the packaged application using the executable provided for your operating system.



The repository also includes the Python script and Jupyter notebook for users who prefer working directly with the source code.



### Step 3. Prepare the images



The pipeline is designed for 2D images containing individual, sufficiently isolated cellular profiles.



Supported formats include PNG, JPG/JPEG and TIF/TIFF.



Images can be organized in a single directory or in subfolders representing experimental groups.



### Step 3. Run the analysis



Follow the application's interface to select your dataset and configure the available analysis options.



The pipeline includes segmentation, skeletonization, fractal dimension, lacunarity, texture analysis, multiscale descriptors, graph-based morphometrics, Sholl analysis and downstream statistical modules.



### Step 5. Inspect the outputs



Depending on the analyses performed, the pipeline generates:



* Morphometric tables in CSV and XLSX formats.



* Quality-control images and analytical figures.



* Integrated morphometric feature matrices.



* Statistical comparisons and dimensionality-reduction outputs.



**Important:** The current implementation analyzes two-dimensional cellular profiles and does not perform three-dimensional reconstruction. Reliable measurements depend on suitable image quality and adequate segmentation of individual cells.



# **Core Dependencies**



Python, numpy, pandas, matplotlib, seaborn, scipy, scikit‑image, OpenCV, Pillow, networkx, scikit‑learn, umap‑learn, pingouin, statsmodels, scikit‑posthocs, shap, openpyxl, and tkinter.



# **Authors**



Fernando Laso García - ORCID ID: 0000-0002-5481-0514

Translational Stroke Laboratory (TREAT), Clinical Neurosciences Research Laboratory (LINC), Health Research Institute of Santiago de Compostela (IDIS), Santiago de Compostela, 15706, Spain. email: fernilaso.9@gmail.com



Jorge Pascual Guerra - ORCID ID: 0000-0001-6667-3310

Neurooncology Unit, Chronic Disease Department (UFIEC), Instituto de Salud Carlos III, Majadahonda, Madrid, Spain. email: jorge.pascualguerra@gmail.com

