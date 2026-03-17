
## **Using Spectral Analysis and Passive Linguistic Monitoring for Detecting Cognitive, Behavioral, and Risk-related Fluctuations in Cyclical Health Timelines**

### **Project Overview**

This project validates a computational method for detecting **menstrual cycle patterns** directly from social media text. **By applying spectral analysis to passive linguistic monitoring, we identified a high-confidence cohort of 1,059 users exhibiting a robust periodic signal in alignment with biological phase.** **Unlike active self-reporting, which is susceptible to expectancy bias, this passive approach captures naturalistic behavior and implicit changes in user state**.

### **Methodology & Data Pipeline**

* **Anchor Detection** **:** We utilized a multi-stage regex pipeline to identify self-reports of the beginning of the menstrual cycle (Cycle Day 1) from the Pushshift Reddit Archive.
* **Uncertainty Filtering** **:** An uncertainty detection layer excludes speculative language (e.g., "should have," "maybe") to ensure high precision in cycle anchoring.
* **Phase Mapping** : User timelines are stratified into four standardized phases: **Menstrual** (days 1–4), **Follicular** (days 5–13), **Ovulation** (days 14–16), and **Luteal** (days 17–28).
* **Spectral Analysis**: We applied Fast Fourier Transform (FFT) with linear interpolation, filtering for signals with a Signal-to-Noise Ratio (SNR) > 3.0 to identify biological rhythms

### **Key Results**

* **Cycle Distribution** : Detected a unimodal cycle length distribution with a mean of **29.1 days**, closely matching large-scale physiological studies of active trackers.
* **Cognitive "Brain Fog"**: Observed a distinct **Luteal phase** change characterized by an increase in spelling errors, suggesting a deficit in attentional resources required for error-checking rather than a loss of verbal fluency.
* **Clinical Risk Analysis**: Identified that users in the **Premenstrual Dysphoric Disorder (PMDD)** subgroup exhibited significantly higher rates of psychiatric risk markers, including depression and suicidality.

### **Repo Structure**

* /scr: Core logic functions.
* /scripts: The main pipeline.
* /configs: Hyperparameters for SNR thresholds and phase window durations.
* run_submission.sh: Executable script to reproduce the trajectories and statistics mentioned in the report.
