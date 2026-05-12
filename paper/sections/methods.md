**2. Methods**

**2.1 Data Collection and** **preprocessing**** **

**Data Source:** **Pushshift** **Reddit Archive**

We extracted all postings between January 2015 and October 2025 that matched expressions indicating the beginning of the menstrual cycle (cycle day 1) using a set of regular expressions designed to capture explicit self-reports of menstruation (e.g., 'started my period', see Appendix for the full list of patterns). To minimize noise, we implemented an uncertainty detection layer, flagging and excluding posts containing **speculative language (**e.g., 'it seems', 'should have'). We then extracted all the posts (across Reddit) made by women who matched these regular expressions at least once. **The raw dataset underwent a multi-stage preprocessing pipeline: deleted accounts and submissions were removed, and submissions shorter than 150 characters were excluded to ensure sufficient linguistic signal. For each remaining post, we calculated the temporal offset from the reported cycle start date. **

To validate the automated extraction pipeline, a random sample of up to 30 posts for each of the 6 targeted patterns (N=176) was manually annotated by an independent annotator. We calculated both observed agreement and Cohen's Kappa to measure human-computer inter-rater agreement. The pipeline demonstrated high overall accuracy. For five of the six patterns, this translated to strong Cohen’s Kappa scores ranging from 0.86 to 1.00 (mean 𝜅 **= 0.94). For one pattern, despite achieving an 83.3% observed agreement, the highly imbalanced nature of the target class resulted in a high expected chance agreement, artificially reducing the Kappa score (**𝜅 = 0) due to the prevalence paradox. Overall, the validation confirms the high precision of the regex pipeline in replicating human annotations. ** **

**2.2 Timeline construction and feature extraction**

For each user with an anchor post, we constructed a chronological timeline of posting activity centred on their anchor posts. The analysis window was restricted to 3 months prior and 3 months subsequent to each anchor post, since menstruation cycles are known to vary ([Urteaga, 2020](https://www.researchgate.net/publication/341647447_Characterizing_physiological_and_symptomatic_variation_in_menstrual_cycles_using_self-tracked_mobile-health_data)). To ensure sufficient density for periodicity detection, we required at least 10 days with posting activity within a range of 45 days around the anchor post. To capture fluctuations in mental state, we computed a set of linguistic and cognitive features for every post.

[table here] 



**2.3 Periodicity and signal detection**

To identify periodicity, posts from the same day were aggregated by calculating the mean value for each linguistic feature, resulting in a single time series per feature per user. Then, we applied the Fast Fourier Transform (FFT) with linear interpolation to the user timelines. To ensure robustness, we filtered out signals with a Signal-to-Noise Ratio (SNR) lower than 3.0. SNR was defined as the ratio of the spectral power at the detected peak frequency to the median power of the background spectrum. We also tested the use of **Lomb-**Scargle periodograms, which are often used for unevenly spaced data. However, these periodograms frequently identified **spurious peaks at search window boundaries (e.g., 21 or 35 days) rather than biological rhythms. 


**2.4 Phase mapping**

We stratified user posts into four standardized phases ([Fehring, 2006](https://www.researchgate.net/publication/7080221_Variability_in_the_Phases_of_the_Menstrual_Cycle)): menstrual (days 1–4), follicular (days 5–13), ovulation (days 14–16), and luteal (days 17–28). For cycles diverging from the standard 28-day model, we adjusted the duration of the follicular phase (expanding or compressing the window), while maintaining a fixed duration for remaining phases ([Fehring, 2006](https://www.researchgate.net/publication/7080221_Variability_in_the_Phases_of_the_Menstrual_Cycle)) For each user, the follicular phase was defined as days 5 through (cycle length − 12), such that a 28-day cycle yields days 5–16, a 30-day cycle yields days 5–18, and a 26-day cycle yields days 5–14. While adjusting the follicular phase duration for individual cycle variability improves accuracy, it is important to acknowledge that maintaining fixed durations for the menstrual, ovulation, and luteal phases is a simplification and future research could explore more dynamic modelling of all phases to capture a broader spectrum of physiological patterns.
