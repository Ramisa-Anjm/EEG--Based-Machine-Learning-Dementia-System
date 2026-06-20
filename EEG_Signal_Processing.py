import matplotlib
matplotlib.use("Qt5Agg")
 
import scipy.io as spio
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# LOAD DATA
fname = "sub-007_ses-pre_task-SART_eeg.set"
mat = spio.loadmat(fname, struct_as_record=False, squeeze_me=True)

#manually extracting events
events_eeglab = mat['event']

# Make sure it's iterable
if not isinstance(events_eeglab, (list, np.ndarray)):
    events_eeglab = [events_eeglab]

latencies = []
event_types = []

for ev in events_eeglab:
    lat = ev.latency
    if np.size(lat) == 1:
        latencies.append(int(lat) - 1)
        event_types.append(str(ev.type[0]))

unique_types = sorted(set(event_types))
event_id = {k: i+1 for i, k in enumerate(unique_types)}

events = np.zeros((len(latencies), 3), dtype=int)
events[:, 0] = latencies
events[:, 2] = [event_id[t] for t in event_types]

print(event_id)
print(events.shape)

data = np.asarray(mat['data'], dtype=float) * 1e-6  # volts
srate = float(mat['srate'])

chanlocs = mat['chanlocs']
chans = [cl.labels for cl in chanlocs]

info = mne.create_info(chans, srate, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# SET MONTAGE 
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, match_case=False, on_missing="ignore")

# FILTERING (DATA ALREADY HP=1 Hz, SO WE DON'T TOUCH LOW FREQ)
raw.filter(l_freq=None, h_freq=40)
raw.notch_filter(freqs=60)

# ICA PREP DATA 
raw_for_ica = raw.copy().filter(l_freq=1., h_freq=None)

ica = ICA(
    n_components=0.99,
    random_state=97,
    max_iter="auto"
)

ica.fit(raw)

eog_inds, scores = ica.find_bads_eog(raw, ch_name='Fp1')
ica.exclude = eog_inds

ica.apply(raw)
mne.export.export_raw("sub07_clean.set", raw, fmt="eeglab")
