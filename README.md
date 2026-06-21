Overview:
This project investigates whether neural oscillatory interactions, quantified through Phase-Amplitude Coupling (PAC), can be used to decode cognitive states from electroencephalography (EEG) recordings.

PAC features are extracted from event-related EEG epochs and used to train a Random Forest classifier capable of distinguishing between different cognitive conditions based on underlying patterns of neural activity. The project combines neuroscience signal processing techniques with machine learning to explore how brain activity can be translated into predictive models.

Research Question:
Can Phase-Amplitude Coupling (PAC) features extracted from EEG recordings be used to accurately classify cognitive states associated with attentional performance?

Data Description: 
The dataset consists of EEG recordings collected during the Sustained Attention to Response Task (SART), a widely used paradigm for measuring sustained attention and response inhibition. During the task, participants are presented with a sequence of numbers (1–9) and are instructed to respond to every stimulus except a designated target number (3). Behavioural markers such as errors on target trials are used to identify differences in attentional state and cognitive control.

Methodology:
1. EEG Preprocessing: The raw EEG data underwent several preprocessing steps: Bandpass filtering, removal of powerline noise, Independent Component Analysis (ICA) for artifact removal, identification and removal of noisy channels and signal normalization.
2. Event Extraction and Epoching: Experimental event markers are extracted, and EEG recordings are segmented into epochs surrounding each event. The epoch is 500 milliseconds before the event marker and 1500 milliseconds after the event marker to collect relevant neural signals. 
3. Feature (PAC) Extraction: PAC measures how the phase of a low-frequency oscillation modulates the amplitude of a higher-frequency oscillation. For each epoch, the low-frequency phase is extracted from Theta (4-8 Hz) and Alpha (8-12 Hz) while the high-frequency phase is extracted from Beta (13-30 Hz) and Gamma (30-100 Hz).
4. Feature Matrix Construction: Each epoch is represented by PAC metrics, channel information, and frequency-pair coupling strengths.
5. Machine Learning: A Random Forest classifier is trained on the feature matrix. The performance of the Machine Learning model was assessed on Area Under the Curve (AUC) and p-value tests.

Results & Applications:
The Machine Learning model was found to be 86% accurate, demonstrating that EEG-derived Phase-Amplitude Coupling features contain meaningful information that can be used to predict cognitive states. Statistical testing and machine learning evaluation suggest that oscillatory coupling patterns may serve as useful neural biomarkers for cognitive decoding applications. Such approaches may contribute to the development of non-invasive tools for monitoring neurological conditions, detecting cognitive decline, and supporting earlier intervention in disorders such as ADHD and dementia.
