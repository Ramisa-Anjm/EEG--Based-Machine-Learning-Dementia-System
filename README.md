Overview:
This project investigates whether neural oscillatory interactions, measured through Phase-Amplitude Coupling (PAC), 
can be used to decode cognitive states from EEG recordings. By extracting PAC features from event-related EEG epochs 
and training a Random Forest classifier, the model attempts to distinguish between different cognitive conditions 
based on underlying neural activity.

The project combines neuroscience signal processing techniques with machine learning to explore how patterns of 
brain activity can be translated into predictive models.

Data Description: 
The data consists of EEG recordings collected from participants performing the Sustained Attention to Response Task (SART). SART is a cognitive task used to measure attention and response inhibition. Participants are shown a sequence of numbers between 1 and 9 and must respond to every stimulus except for a designated target (in this case, the number 3). Markers identifying errors on target trials and reaction time patterns are used to assess lapses in attention and cognitive control. 

Methodology:
1. EEG Preprocessing: The raw EEG data underwent several preprocessing steps: Bandpass filtering, removal of powerline noise, Independent Component Analysis (ICA) for artifact removal, identification and removal of noisy channels and signal normalization.
2. Event Extraction and Epoching: Experimental event markers are extracted, and EEG recordings are segmented into epochs surrounding each event. The epoch is 500 milliseconds before the event marker and 1500 milliseconds after the event marker to collect relevant neural signals. 
3. Feature (PAC) Extraction: PAC measures how the phase of a low-frequency oscillation modulates the amplitude of a higher-frequency oscillation. For each epoch, the low-frequency phase is extracted from Theta (4-8 Hz) and Alpha (8-12 Hz) while the high-frequency phase is extracted from Beta (13-30 Hz) and Gamma (30-100 Hz).
4. Feature Matrix Construction: Each epoch is represented by PAC metrics, channel information, and frequency-pair coupling strengths.
5. Machine Learning: A Random Forest classifier is trained on the feature matrix. The performance of the Machine Learning model was assessed on Area Under the Curve (AUC) and p-value tests.

Results & Applications:
The Machine Learning model was found to be 86% accurate, demonstrating that EEG-derived Phase-Amplitude Coupling features contain meaningful information that can be used to predict cognitive states. Statistical testing and machine learning evaluation suggest that oscillatory coupling patterns may serve as useful neural biomarkers for cognitive decoding applications. Such approaches may contribute to the development of non-invasive tools for monitoring neurological conditions, detecting cognitive decline, and supporting earlier intervention in disorders such as ADHD and dementia.
