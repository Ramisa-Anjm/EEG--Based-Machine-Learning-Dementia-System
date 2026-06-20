from mne.io.eeglab import eeglab
import numpy as np
import pandas as pd
import mne
from pactools import Comodulogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.model_selection import permutation_test_score
from scipy.signal import welch

# --------- ignore broken EEGLAB events ----------
def _dummy_annotations_eeglab(fname, uint16_codec=None):
    from mne import Annotations
    return Annotations([], [], [])

eeglab._read_annotations_eeglab = _dummy_annotations_eeglab

def process_single_subject(events_file, eeg_file, subject_id, task_name):
    """
    Process a single subject and return features, labels, and group ID.
    """
    print(f"\n{'='*50}")
    print(f"Processing subject {subject_id}...")
    print(f"EEG file: {eeg_file}")
    print(f"Events file: {events_file}")

    # Load events
    events = pd.read_csv(events_file, sep="\t")
    
    # Load EEG
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    sfreq = raw.info['sfreq']
    print(f"Loaded {len(raw.ch_names)} channels at {sfreq} Hz")

    events_df = events.copy()

    # Get target trials (correct inhibition - no response)
    targets = events_df[
        (events_df["value"] == "nontarget") & 
        (events_df["response_type"] == "correct")
    ]["onset"].values

    # Get non-target trials (correct response - they pressed)
    nontargets = events_df[
        (events_df["value"] == "nontarget") & 
        (events_df["response_type"] == "incorrect")
    ]["onset"].values

    print(f"  Raw targets (correct inhibition): {len(targets)}")
    print(f"  Raw non-targets (correct response): {len(nontargets)}")

    # Balance classes
    n = min(len(targets), len(nontargets))
    if n == 0:
        print(f"WARNING: No valid trials for subject {subject_id}, skipping...")
        return None, None, None
    
    print(f"  Balanced to: {n} per class")

    # Random sample if unequal (set seed for reproducibility)
    np.random.seed(42)
    if len(targets) > n:
        targets = np.random.choice(targets, n, replace=False)
    if len(nontargets) > n:
        nontargets = np.random.choice(nontargets, n, replace=False)

    # Create dataframes for the selected onsets
    focused_df = events_df[events_df["onset"].isin(targets)]
    unfocused_df = events_df[events_df["onset"].isin(nontargets)]

    # Extract onsets
    target_onsets = focused_df['onset'].values
    nontarget_onsets = unfocused_df['onset'].values

    # Combine
    all_onsets = np.concatenate([target_onsets, nontarget_onsets])
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    # Convert to samples for MNE
    events_array = np.column_stack([
        (all_onsets * sfreq).astype(int),
        np.zeros(len(all_onsets), dtype=int),
        np.ones(len(all_onsets), dtype=int)
    ]).astype(int)

    # Create epochs
    epochs = mne.Epochs(
        raw, events_array, {'trial': 1},
        tmin=-0.2,    # 200ms before stimulus
        tmax=1.0,     # 1 second after
        baseline=None,
        preload=True,
        verbose=False
    )

    # ========== EXTRACT PAC ==========
    # Find Fz channel (or alternative)
    ch_name = 'Fz' if 'Fz' in epochs.ch_names else 'FCz' if 'FCz' in epochs.ch_names else epochs.ch_names[0]
    ch_idx = epochs.ch_names.index(ch_name)
    sfreq = epochs.info['sfreq']

# Get epoch data and normalize
    epoch_data = epochs.get_data()[:, ch_idx, :]  
    subject_mean = np.mean(epoch_data)
    subject_std = np.std(epoch_data)
    normalized_data = (epoch_data - subject_mean) / (subject_std + 1e-8)  # NOW DEFINED

# Compute PAC
    comod = Comodulogram(
        fs=sfreq,
        low_fq_range=(4, 6),      # Narrower Fmθ: 4-6 Hz
        high_fq_range=(30, 50),   # Low gamma only: 30-50 Hz
        method='tort'
    )

    pac_features = []
    for trial in normalized_data:
        comod.fit(trial)
        pac_features.append(np.mean(comod.comod_))

# Theta power features 
    theta_powers = []
    for trial in epoch_data:  # Use original non-normalized data
        freqs, psd = welch(trial, fs=sfreq, nperseg=256)
        theta_mask = (freqs >= 4) & (freqs <= 7)
        theta_powers.append(np.mean(psd[theta_mask]))

    X = np.column_stack([pac_features, theta_powers])
    y = labels
    groups = [f"{task_name}_{subject_id}"] * len(y)

    print(f"  ✓ Extracted {len(y)} trials (PAC: {X.mean():.4f} ± {X.std():.4f})")
    print(f"    Target PAC: {X[y==1].mean():.4f}")
    print(f"    Non-target PAC: {X[y==0].mean():.4f}")

    return X, y, groups


def load_all_subjects(subjects, task_name="SART"):
    """
    Load and combine data from all subjects.
    """
    all_X = []
    all_y = []
    all_groups = []

    for sub in subjects:
        events_file = f"sub-{sub}_ses-pre_task-SART_events.txt"
        eeg_file = f"sub{sub}_clean.set"

        X, y, groups = process_single_subject(events_file, eeg_file, sub, task_name)
        
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_groups.extend(groups)

    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"\n{'='*50}")
    print(f"COMBINED DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(y_combined)}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Class distribution: {np.bincount(y_combined.astype(int))}")
    print(f"Target (1): {np.sum(y_combined == 1)}")
    print(f"Non-target (0): {np.sum(y_combined == 0)}")
    
    return X_combined, y_combined, all_groups

# Define all subjects
subjects = ["001", "002", "003", "004", "005", "006", "007"]

# Load all data
X, y, groups = load_all_subjects(subjects, task_name="SART")

print(f"\n{'='*50}")
print("TRAINING OPTIONS")
print(f"{'='*50}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"AUC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-target', 'Target']))

#Leave-One-Subject-Out Cross-Validation
print(f"\n--- Leave-One-Subject-Out Cross-Validation---")
logo = LeaveOneGroupOut()
clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

# Convert groups to numeric for LeaveOneGroupOut
unique_groups = list(set(groups))
group_ids = [unique_groups.index(g) for g in groups]

cv_scores = cross_val_score(clf_cv, X, y, cv=logo.split(X, y, group_ids), scoring='roc_auc')

print(f"Subject-wise AUC scores: {cv_scores}")
print(f"Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# ========== INTERPRETATION ==========
print(f"\n{'='*50}")
print("INTERPRETATION")
print(f"{'='*50}")

mean_auc = cv_scores.mean()
if mean_auc > 0.7:
    print("EXCELLENT: Strong PAC difference with good generalization!")
elif mean_auc > 0.6:
    print("GOOD: Moderate PAC difference, reasonable generalization")
elif mean_auc > 0.5:
    print(" WEAK: Slight difference, may not generalize well")
else:
    print("RANDOM: No detectable difference or poor generalization")
    
target_pac = X[y == 1, 0]  # First feature, target trials
nontarget_pac = X[y == 0, 0]  # First feature, non-target trials

t_stat, p_value = stats.ttest_ind(target_pac, nontarget_pac)

print(f"\n1. INDEPENDENT T-TEST (Target vs Non-target PAC):")
print(f"   Target PAC:     M={target_pac.mean():.4f}, SD={target_pac.std():.4f}, n={len(target_pac)}")
print(f"   Non-target PAC: M={nontarget_pac.mean():.4f}, SD={nontarget_pac.std():.4f}, n={len(nontarget_pac)}")
print(f"   t({len(target_pac)+len(nontarget_pac)-2}) = {t_stat:.3f}")
print(f"   p-value = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(ns)'}")

# 2. Effect size
pooled_std = np.sqrt(((len(target_pac)-1)*target_pac.var() + (len(nontarget_pac)-1)*nontarget_pac.var()) / 
                     (len(target_pac) + len(nontarget_pac) - 2))
cohens_d = (target_pac.mean() - nontarget_pac.mean()) / pooled_std
print(f"   Cohen's d = {cohens_d:.3f}")

# 3. Mann-Whitney U 
u_stat, p_mw = stats.mannwhitneyu(target_pac, nontarget_pac, alternative='two-sided')
print(f"\n2. MANN-WHITNEY U TEST:")
print(f"   U = {u_stat:.1f}, p = {p_mw:.4f}")
