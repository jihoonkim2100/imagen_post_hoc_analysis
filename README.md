# imagen_post_hoc_analysis
Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, last modified 11th January 2022

Those are required to run this module: <br>
    - h5py                               2.10.0 <br>
    - shap                               0.39.0 <br>
    - pickle                             0.7.5 <br>
    - nibabel                            3.2.1 <br>
    - nilearn                            0.8.1 <br>
    - numpy                              1.20.1 <br>
    - scikit-learn                       0.24.1 <br>
    - scipy                              1.6.2 <br>
    - pandas                             1.2.4 <br>
    - matplotlib                         3.3.4 <br>
    - seaborn                            0.11.1 <br>
    - statannot                          0.2.3 <br>
    - statsmodels                        0.12.2 <br>

The preliminary results in our IMAGEN paper adcovates for a more in-depth understanding of what contributes to the significant performance of the ML models for the three time points:
- Baseline (BL), Age 14 <br>
- Follow 1 year (FU1), Age 16 <br>
- Follow 2 year (FU2), Age 19 <br>
- Follow 3 year (FU3), Age 22 <br>

Such in-depth understaning can be acheived by performing follow-up analysis such as:
<br>
### Summary Statistics
<b>Compare</b> the different instruments suchh as personality traits, socio-demographic, life history, and so on between the two groups = Aldolescents Alcohol Misuse(AAM) and Healthy Control (HC).
<br>
<br>
### Senstivity Analysis
Test the robustness of the result: Did we overfit? Was the influence of confound 'site' really removed?
<br>
<br>
### Error Analysis
Error analysis of the model prediction to understand which and why the model performs well on some subjects and bad on some others with Instrument
<br>
<br>
### Visualization
Visualize the relative importance of the different structural features and their contribution to the model performance using methods such as SHAP values (lundberg 2017)
<br>
<br>
### Preprocessing
Preprocess the Instrument https://imagen-europe.com/resources/imagen-dataset/documentation/ and post hoc dataset; HDF5, RUN file, and generate the SHAP derivatives dataset.

The further information can be found in https://github.com/RoshanRane/imagen_ml
