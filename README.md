# BCI_Ker_Challenge
This is my solution to the P-300 BCI_KER challenge on Kaggle
# Brain-Computer Interface Error Detection (P300 Speller)

## Problem Description
As humans think, we produce brain waves. These brain waves can be mapped to actual intentions.
In this project, we work on **detecting errors during spelling tasks from EEG brain wave data**.
The setting comes from the **P300-Speller**, a well-known Brain-Computer Interface (BCI) paradigm. Participants focus on visual stimuli to spell words. Each trial involves flashing groups of letters/numbers on a screen, and the algorithm identifies the intended symbol based on the brain’s P300 response.
The challenge: **determine when the feedback (selected item) does not match the intended target** using EEG signals recorded after feedback.

---

## Experimental Setup

* **P300 Speller**:
  * Uses EEG signals and the P300 event-related potential.
  * 36 possible items (letters/numbers) displayed in a matrix.
  * Items are flashed in random groups.

* **Calibration**:
  * A short calibration session was used to learn a prototypical target response for each participant.

* **Test Sessions**:
  * Subjects spelled words by focusing on target items.
  * After flashes, feedback was displayed on screen (may be correct or incorrect).
  * Task: analyze brain waves during feedback to detect **errors**.

* **Modes**:
  1. **Fast mode**: 4 flashes per item (higher error rate).
  2. **Slow mode**: 8 flashes per item (lower error rate).

---
## Approach

1. **Data Loading & Merging**
   * EEG recordings provided in multiple `.csv` files were extracted and combined into a single DataFrame.
   * Each row corresponds to a timestamp with EEG channels, an EOG (eye movement) channel, and feedback labels.

2. **Signal Preprocessing**
   * Applied a **notch filter (50 Hz)** to each EEG channel to remove powerline noise.
   * Cleaned data by addressing artifacts:

     * **ICA decomposition**: separated EEG into independent components.
     * Correlated each component with the **EOG signal** and suppressed those strongly correlated (>0.3 threshold).
     * Reconstructed EEG signals after artifact suppression.
     * Further applied **linear regression** to regress out residual EOG influence from EEG channels.

3. **Feature Extraction**
   * Computed simple statistical features per timestamp:
     * Variance
     * Mean
     * Standard deviation
   * These features form the input representation for classification.

4. **Modeling**
   * Split dataset into **train (80%) / validation (20%)**.
   * Trained a **Support Vector Classifier (SVC)** with probability estimates.
   * Evaluated using **ROC-AUC** score on validation set.

5. **Testing & Predictions**
   * Loaded and processed test EEG files similarly.
   * Extracted the same statistical features (variance, mean, std).
   * Used the trained SVM to predict whether feedback events were **correct or incorrect**.

---


---


