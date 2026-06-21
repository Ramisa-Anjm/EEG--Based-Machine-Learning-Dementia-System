Overview:
This project investigates whether neural oscillatory interactions, quantified through Phase-Amplitude Coupling (PAC), can be used to decode cognitive states from electroencephalography (EEG) recordings.

PAC features are extracted from event-related EEG epochs and used to train a Random Forest classifier capable of distinguishing between different cognitive conditions based on underlying patterns of neural activity. The project combines neuroscience signal processing techniques with machine learning to explore how brain activity can be translated into predictive models.

Research Question:
Can Phase-Amplitude Coupling (PAC) features extracted from EEG recordings be used to accurately classify cognitive states associated with attentional performance?

Data Description: 
The dataset consists of EEG recordings collected during the Sustained Attention to Response Task (SART), a widely used paradigm for measuring sustained attention and response inhibition. During the task, participants are presented with a sequence of numbers (1–9) and are instructed to respond to every stimulus except a designated target number (3). Behavioural markers such as errors on target trials are used to identify differences in attentional state and cognitive control.

Methodology:
1. EEG Data Import and Event Extraction: Event latencies and event labels were extracted and converted into an MNE-compatible event structure, where each event contained the sample index, event identifier and event type label, which was later used for epoch segmentation and trial classification.
2. EEG Data Conversion and Channel Configuration: Raw EEG signals were converted from microvolts to volts and imported into the MNE-Python framework as a continuous EEG recording. Channel names were extracted from the dataset metadata and used to create an MNE information object containing EEG channel labels, sampling frequency, and channel type information. Electrode locations were assigned using the International 10–20 electrode placement system through the standard_1020 montage.
3. Signal Filtering: To reduce high-frequency noise while preserving task-relevant neural activity, a low-pass filter with a cutoff frequency of 40 Hz was applied to the continuous EEG recording. Because the dataset had already undergone high-pass filtering at 1 Hz during acquisition or previous preprocessing, no additional low-frequency filtering was performed. To remove electrical interference from power-line noise, a notch filter centred at 60 Hz was applied to all channels.
4. Independent Component Analysis (ICA): Independent Component Analysis (ICA) was used to identify and remove physiological artifacts from the EEG recordings. ICA was performed using a variance retention threshold of 99%, a random seed of 97 and an automatic convergence criteria. The resulting independent components were inspected using an automated electrooculogram (EOG) detection procedure based on the Fp1 electrode. Components exhibiting strong correlations with ocular activity were identified as eye-blink or eye-movement artifacts and marked for removal.
5.  
   

Results & Applications:
The Machine Learning model was found to be 86% accurate, demonstrating that EEG-derived Phase-Amplitude Coupling features contain meaningful information that can be used to predict cognitive states. Statistical testing and machine learning evaluation suggest that oscillatory coupling patterns may serve as useful neural biomarkers for cognitive decoding applications. Such approaches may contribute to the development of non-invasive tools for monitoring neurological conditions, detecting cognitive decline, and supporting earlier intervention in disorders such as ADHD and dementia.
