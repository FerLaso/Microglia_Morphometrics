# Microglia\_Morphometrics



This notebook presents a fully reproducible and universally applicable pipeline for quantitative microglial morphometric analysis. Although the workflow is technically compatible with any single-cell dataset, it has been specifically optimized and documented for microglia, the resident immune cells of the central nervous system.



Microglial morphology provides relevant information about functional state, neuroimmune activity, and tissue homeostasis. However, quantitative morphometric analysis remains inconsistent across laboratories and often relies on partially manual workflows, limited descriptor sets, and computational approaches with insufficient validation. Here, this work presents a fully automated, modular, and reproducible Python-based pipeline for multiscale microglial morphometry. The framework integrates dataset loading, preprocessing, segmentation, skeletonization, fractal dimension, lacunarity, GLCM texture metrics, multiscale fractal spectrum analysis, graph‑based descriptors, Sholl profiling, adaptive statistics, PCA/UMAP, Random Forest classification, and SHAP interpretability.





# **Main Modules**



0\. Synthetic microglia generator.



1. Universal dataset loader.



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





# **Download**



Downloads are available in the **Releases** section:



\-- Windows x64



\-- Ubuntu Linux x86‑64



\-- macOS (Apple Silicon, arm64)



\-- macOS (Intel, x86\_64)





# **Core Dependencies**



Python, numpy, pandas, matplotlib, seaborn, scipy, scikit‑image, OpenCV, Pillow, networkx, scikit‑learn, umap‑learn, pingouin, statsmodels, scikit‑posthocs, shap, openpyxl, and tkinter.



# **Authors**



Fernando Laso García - ORCID ID: 0000-0002-5481-0514

Translational Stroke Laboratory (TREAT), Clinical Neurosciences Research Laboratory (LINC), Health Research Institute of Santiago de Compostela (IDIS), Santiago de Compostela, 15706, Spain. email: fernilaso.9@gmail.com



Jorge Pascual Guerra - ORCID ID: 0000-0001-6667-3310

Neurooncology Unit, Chronic Disease Department (UFIEC), Instituto de Salud Carlos III, Majadahonda, Madrid, Spain. email: jorge.pascualguerra@gmail.com





