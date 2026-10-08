# Scaletta — Presentation

**Project:** Audio Anomaly Detection on Water Pumps — Learning a ConvAE Latent Space for Unsupervised Fault Detection
**Course:** Digital Forensics and Biometrics — A.A. 2025/2026
**Format:** English · ~20 minutes · 25 slides · 16:9
**Visual style:** imitates `presentation-template.pdf` (cream background, bold headings, lavender section chips, big stat tiles, 3-card rows, two-up "winner" stats, blank centered dividers).
**Numbers source:** `presentation_assets/` (metrics/tables/figures; figures mirrored in `images/`, which the deck uses).

---

## Design system (locked)

| Element | Value |
|---|---|
| Background | `#FFFAFA` (warm off-white) |
| Heading | `#1A1A1A`, heavy geometric sans (Poppins), left-aligned (centered on dividers) |
| Section chip | pill, fill `#D5DCF6`, dark uppercase label |
| Body / caption | `#444444` |
| Accent blue | `#6094EA` · Accent red | `#E23343` |

---

## Reference numbers (final test, file-level; mean aggregation; balanced-accuracy threshold)

| Method | Bal Acc | AUC-ROC | AUC-PR | Uses labels? |
|---|---|---|---|---|
| ConvAE Reconstruction Error | 0.595 | 0.646 | 0.722 | No |
| ConvAE + Isolation Forest | 0.560 | 0.601 | 0.702 | No |
| **ConvAE + LOF** ⭐ | **0.733** | **0.788** | **0.852** | No |
| One-Class SVM (AE latent) | 0.545 | 0.615 | 0.716 | No |
| PANNs + LOF | 0.702 | 0.761 | 0.830 | No (external features) |
| PANNs + Isolation Forest | 0.596 | 0.609 | 0.713 | No (external features) |
| SVM-sup (AE latent) | 0.827 | 0.953 | 0.972 | **Yes** |
| XGBoost (AE latent) | 0.827 | 0.941 | 0.965 | **Yes** |
| **SVM-sup (MFCC stats)** | **0.854** | 0.934 | 0.959 | **Yes** |

- **Splits (files):** ConvAE train 1792 / val 449 (normal only, IDs 00/02/04); tuning 200 normal + 136 anomaly; **test 200 normal + 320 anomaly**. **ID 06 unseen in training.**
- **Preprocessing:** 16 kHz mono · 2 s window / 1 s hop → 9 windows per 10 s clip · mel `n_fft=1024`, `hop=512`, `n_mels=128` · log dB `ref=1.0` · clip 1st/99th pct (≈ −46.1 / −10.0 dB) → [0,1] · pad to **128 × 64**.
- **ConvAE:** encoder 32→64→128→256 (first layer stride (1,2)) → **64-d latent** · **8192 → 64 = 128× compression** · sigmoid decoder · MSE, Adam (1e-3, wd 1e-5), ReduceLROnPlateau, early stopping · checkpoint **52 epochs, best val MSE 0.0073**.
- **Per-machine LOF false-positive rate:** ID06 **51.1%** vs ID00 14.5% / ID02 10.2% / ID04 16.3%.
- **Protocol 2 (add unseen ID06 normals to training), LOF:** Bal Acc 0.733→**0.737** · AUC 0.788→**0.810** · ID06 FPR 51.1%→**40.4%** · ID06 FNR 31.9%→34.8%.

---

## Speaker Script

**Deck:** `presentation.pptx` (25 slides) · **Target:** 20 minutes · English.


### Slide 1 — Title  *(~30 s)*
Good morning. This project is **audio anomaly detection on water pumps**. The goal is to tell whether a pump *sounds* healthy or faulty, using only its recorded sound. The key constraint — and the assignment requirement — is a two-stage design: first **learn a representation** of normal audio with a convolutional autoencoder, then **detect anomalies** in that learned feature space.

### Slide 2 — What is Acoustic Anomaly Detection?  *(~50 s)*
Three ideas frame the project. First, **condition monitoring**: rotating machinery like pumps degrades in ways you can often *hear* before it fails, so audio is a cheap early-warning signal. Second, the setting is **unsupervised, or one-class**: in the real world you have lots of healthy recordings but almost no labeled faults, because faults are rare and diverse. So the model is trained only on normal sound, and anything that deviates is treated as an anomaly. Third — and this is the assignment's structure — the approach is to **represent, then detect**: a feature space is learned first, and only afterwards are outlier detectors applied on top of it.

### Slide 3 — Dataset  *(~55 s)*
The data are recordings from **four pump machine IDs** — 00, 02, 04, and 06 — as mono audio at **16 kHz**. Each 10-second clip is cut into overlapping windows, giving **9 windows per clip**, and the held-out test set has **200 normal and 320 anomalous** files. One detail matters a lot later: **machine ID 06 is never seen by the autoencoder or the detectors — it appears only in the tuning and test data.** So the model has to cope not just with faults, but with a *normal machine it has never heard*. All splits are at the file level with a fixed seed, so nothing leaks between training, tuning, and test.

### Slide 4 — Divider: Data & Preprocessing  *(~5 s)*
First, how the raw audio becomes model input.

### Slide 5 — Preprocessing pipeline  *(~50 s)*
The pipeline has four steps. Audio is loaded as mono at 16 kHz; it is sliced into **2-second windows with a 1-second hop**, which gives 9 windows per clip; each window becomes a **log-mel spectrogram** of size 128 by 64; and that is normalized to the range zero to one. The windowing is deliberate: it keeps the view **local**, so a short fault isn't averaged away across ten seconds. Because labels are per file, the window-level scores are later recombined into one score per file.

### Slide 6 — A normalization choice that matters  *(~55 s)*
One preprocessing decision is worth highlighting. When the spectrogram is converted to decibels, it uses a **fixed reference of 1.0**, not each spectrogram's own maximum. Normalizing by the per-image max would throw away *absolute loudness* — but a quieter or louder pump is exactly the kind of difference a one-class model should see. The values are then clipped to the 1st and 99th percentiles of the training data — about minus 46 to minus 10 dB — and mapped to zero–one. This histogram shows those percentile bounds on the training distribution.

### Slide 7 — Divider: The ConvAE Representation  *(~5 s)*
Now the model that learns the feature space.

### Slide 8 — Convolutional Autoencoder  *(~60 s)*
The representation model is a **convolutional autoencoder**. The encoder has four convolutional stages — 32, 64, 128, 256 channels — that compress the spectrogram down to a **64-dimensional latent vector**; the decoder mirrors that back to a reconstruction and ends in a sigmoid, because the target is normalized to zero–one. That's a **128-times compression**, from 8192 spectrogram values to 64, which forces the bottleneck to keep only the essential structure of *normal* sound. One design touch: the first convolution uses stride (1,2) so the frequency axis isn't crushed too early, which would lose harmonic detail. Crucially, the autoencoder is **trained on normal audio only**, then frozen — and it provides two anomaly signals: the reconstruction error and the latent vector.

Si è fermato a 52 epochs per via dell'early stopping.

### Slide 9 — Training converges cleanly  *(~40 s)*
A quick sanity check on training. The model trained for **52 epochs** with early stopping, best validation MSE around **0.0073**, and — importantly — the training and validation curves track each other closely. So if separation turns out weak later, it is *not* because the autoencoder failed to train or overfit; the representation is learned properly. 

(The autoencoder trained cleanly — it reconstructs normal audio well (MSE 0.0073), it stopped and tuned itself automatically, and training and validation stayed glued together, so any weak results later are not a training failure.)

### Slide 10 — Reconstruction quality  *(~45 s)*
Here are input spectrograms and their reconstructions. On normal windows the autoencoder rebuilds them faithfully, which is the desired behavior. But notice it *also* reconstructs many **anomalous** windows fairly well. That already hints at a problem the results will confirm: if the decoder can rebuild a fault almost as well as a normal sample, then plain reconstruction error won't separate them cleanly.

### Slide 11 — Divider: Detecting Anomalies  *(~5 s)*
So, given this representation, how are anomalies actually flagged?

### Slide 12 — Two families of detectors  *(~55 s)*
The detectors are split into two families, to be explicit about fairness. On the left, the **unsupervised** methods — reconstruction error, Isolation Forest, LOF, and One-Class SVM — use **no anomaly labels**; these are the true anomaly-detection claim of the project. On the right, **supervised** methods — an SVM, XGBoost, and an SVM on MFCC statistics — *do* use anomaly labels during training. They're useful as **diagnostic baselines**, to check how much discriminative signal exists, but they solve an easier problem, so they are not treated as the headline result.

### Slide 13 — Detectors on the latent space  *(~45 s)*
The three unsupervised latent-space detectors work on different principles. **One-Class SVM** learns a single rigid boundary around the normal data. **LOF** — Local Outlier Factor — is density-based: it flags points sitting in locally sparse regions. And **Isolation Forest** isolates outliers with random splits. All three consume the *same* frozen 64-d latent, and all follow one convention — higher score means more anomalous — so they plug into a single shared thresholding and evaluation pipeline.

### Slide 14 — Thresholding & evaluation  *(~55 s)*
This slide is about turning scores into an honest decision — three steps. **One: combine.** Each detector scores every window, but the label is per file, so the windows are **averaged** into one score per file — a single noisy window can't trigger a false alarm on its own. **Two: draw the line.** The decision threshold is picked on a **separate tuning split**, never on the test set, optimizing **balanced accuracy** — because the test set is lopsided, 200 normal versus 320 anomalies, and balanced accuracy scores both sides fairly. **Three: seal the exam.** The test set stays untouched until the very end, and tuning and test files never overlap, so there's no leakage. And two of the reported numbers — AUC-ROC and AUC-PR — don't even need a threshold: they just measure whether anomalies score higher than normals, so those results hold wherever you draw the line.

* First, the window scores are combined: the windows are **averaged** into one score per file, so one bad window does not cause a false alarm. 

* Then, the decision threshold is chosen using a separate tuning set with balanced accuracy, while keeping the test set untouched to make sure the results are fair. 

* Finally, AUC-ROC and AUC-PR measure how well the model ranks abnormal files above normal ones, so they do not depend on the chosen threshold.

* Tell that the LOF is the only one that does a good separation compared to the other ones.

### Slide 15 — Divider: Results & Analysis  *(~5 s)*
I'll now take you through the results and what they show.

### Slide 16 — Winner: ConvAE + LOF  *(~45 s)*
Among the unsupervised methods, the clear winner is **ConvAE plus LOF**, with **73.3% balanced
accuracy** and an **AUC-ROC of 0.788** on the held-out test set. The takeaway is that **local
density** in the latent space captures the anomaly structure far better than methods that try to
draw a single global boundary.

### Slide 17 — All methods compared  *(~65 s)*
Here's the full picture. Isolation Forest is essentially at chance — 0.56 balanced accuracy —
and reconstruction error is only marginally better at 0.60. LOF, highlighted in blue, is the
best unsupervised method at 0.733. The grey rows are the supervised baselines: they score higher — up to **0.854** for the
MFCC-statistics SVM — but remember they **use anomaly labels**, and their threshold is tuned on
the same labeled split, so those numbers are optimistic. They confirm the audio *does* contain
discriminative signal, but they're diagnostics, not the main claim. Also included is **PANNs plus
LOF**, using external pretrained audio embeddings, at 0.702 — a sanity check that the ConvAE
features are competitive with an off-the-shelf representation.

### Slide 18 — Why reconstruction error is weak  *(~50 s)*
This slide explains the weak reconstruction result. These are the score distributions for normal
versus anomaly on the test set. For reconstruction error and Isolation Forest, the two
distributions **overlap heavily** — there's no clean cut. LOF separates them best. The reason is
that **average pixelwise MSE is a coarse score**: a pump fault often changes local texture,
harmonic stability, or periodicity without moving the *mean* reconstruction error much — so it
hides inside normal-looking error values.

### Slide 19 — The ID 06 problem  *(~55 s)*
Now the most interesting analysis. The LOF errors were broken down **by machine ID**, after the
thresholds were already fixed. Machine **ID 06** — the one never seen in training — has a
false-positive rate of **51%**, while the three training machines sit at 14, 10, and 16%. In
other words, most of LOF's false alarms come from a *normal machine it had never heard*: the
model confuses "new normal machine" with "anomaly". This is a diagnostic decomposition — it
doesn't change the reported metrics — but it points at a concrete cause. Let me show you the
evidence.

### Slide 20 — Per-machine score distributions  *(~45 s)*
These are the actual score distributions, machine by machine. Look at the bottom-right panel:
for LOF, the **ID 06 normal scores straddle the decision threshold** — that is the 51%
false-positive rate, visually. Now compare the top-right panel: for **reconstruction error**,
the ID 06 normals sit comfortably *below* the threshold — ID 06 actually has the *lowest*
false-positive rate there. So the two signals disagree in an informative way: the problem is
not that ID 06 is badly reconstructed; it is that its latents fall in **low-density regions**
of the training distribution. The ID 06 problem lives in the latent-density model.

### Slide 21 — Ablation: add ID 06 normals  *(~55 s)*
To test that hypothesis, an ablation was run — Protocol 2 — adding some **normal ID 06** audio
into the autoencoder's training set, **taken from the tuning split**, keeping the test set
identical. The ID 06 false-positive rate drops from **51% to 40%**, and overall LOF AUC ticks up
from **0.79 to 0.81**. So the unseen-machine effect is real — but the false-positive rate is
still high, and the ID 06 false-*negative* rate even ticks up slightly, from **31.9% to 34.8%**.
So it's only a **partial** explanation; there's genuine overlap in the representation too. A single
global threshold was deliberately kept instead of per-machine calibration, because there
are too few files per machine to calibrate without overfitting.

### Slide 22 — The latent space is only partly separable  *(~40 s)*  *(optional slide)*
This t-SNE view of the latent space makes the ceiling visual: normal and anomaly points are
**partially** separated but still overlap. That's exactly consistent with the moderate
unsupervised scores — the representation carries real but incomplete anomaly information.

### Slide 23 — A diagnostics recap  *(~50 s)*
Before concluding, it's fair to ask: could these moderate numbers just mean something is
wrong in the pipeline? Each stage was checked. The **normalization** uses a fixed dB reference and percentile
clipping verified directly on the training distribution, so absolute loudness is preserved. For
**tuning**, four window-aggregation strategies were swept against three tuning metrics — mean is
the most stable — and thresholds were always chosen on a separate split, with no leakage. As a
**sanity check on the features themselves**, a supervised SVM on hand-crafted MFCC statistics
reaches 0.854 balanced accuracy and 0.93 AUC *when anomaly labels are available* — so the
discriminative signal is clearly in the audio; the unsupervised gap is not missing information.
And the **ID 06 ablation** shows that fixing the unseen-machine issue only recovers part of the
gap. Every stage checks out. The conclusion is that the residual errors come from the **one-class
setting itself**: without labels, the faults are subtle enough to overlap with normal variation in
the learned representation — and one normal machine is entirely unseen in training.

### Slide 24 — Conclusion & Future Work  *(~55 s)*
To conclude. The project meets the assignment objective: it builds a **ConvAE latent space** for water-pump
audio and evaluates several anomaly detectors on it, with **ConvAE plus LOF** as the best
unsupervised method at 0.733 balanced accuracy and 0.788 AUC. Honestly, **reconstruction error
alone is insufficient** for these subtle faults, and the unseen-machine false-positive
effect was diagnosed and quantified with an ablation. For future work: **richer temporal aggregation** than a
simple mean, learning to **disentangle machine identity from fault information**, and
**contrastive or alternative reconstruction objectives** to sharpen the latent space.

### Slide 25 — Thank you  *(~10 s)*
Thank you — I'm happy to take questions.

---

## Anticipated questions & suggested answers

**Q: Why an autoencoder trained only on normal data?**
It's a one-class setup — anomalies are rare and unlabeled in practice. The AE learns to represent normal sound; deviations, in reconstruction error or latent density, signal anomalies without ever training on faults.

**Q: Why is reconstruction error so weak?**
Average pixelwise MSE on log-mel images is coarse. A fault can change local texture or periodicity without raising the mean error much, and the decoder reconstructs many anomalies well enough that their error overlaps the normal range.

**Q: Why does LOF beat Isolation Forest and One-Class SVM?**
LOF uses *local* density, so it adapts to a latent space that's only partially structured. IF and One-Class SVM rely on a more global boundary, which this overlapping space doesn't provide cleanly.

**Q: The supervised models score higher — isn't that your real result?**
They score higher but use anomaly labels in training and tuning, so they solve an easier problem and their numbers are optimistic. They are reported as diagnostics that confirm signal exists; the assignment's object is the unsupervised latent-space detection.

**Q: How do you prevent data leakage?**
File-level splits so windows from one recording never cross splits; the threshold is tuned only on a separate tuning split; supervised cross-validation is grouped by file; and scaling is fit inside each CV fold. The final test set is only used once, at the end.

**Q: Why balanced accuracy instead of plain accuracy?**
The test set is imbalanced — 200 normal, 320 anomaly — so plain accuracy could hide poor performance on one class. Balanced accuracy averages recall and specificity; AUC-ROC and AUC-PR, which are threshold-independent, are also reported.

**Q: What is AUC-ROC / AUC-PR? (Theory)**
Both are single 0-to-1 scores for how well the detector **ranks faults above normals**, with no threshold needed — higher is better. **ROC** is the balanced overall view: it's the probability a random fault scores higher than a random normal (here ≈ 0.79). **PR** zooms in on the rare anomaly class, so it's the more honest number under imbalance (here ≈ 0.85). They are reported because, unlike balanced accuracy, they don't depend on where the threshold is drawn.

**Q: Why are the Isolation Forest / LOF anomaly scores negative on slide 14?**
It's a convention, not a bug. Those scores come from scikit-learn's `decision_function`, which is centered on zero by design — the outlier boundary sits at 0, normals just above, outliers just below. The sign is then flipped so that higher means more anomalous, which pushes the big normal cluster into negative territory. Unlike reconstruction error — a real MSE that is always ≥ 0 — these scores have **no natural "zero means no anomaly"** point, so the sign and magnitude are arbitrary; only the ranking and where the threshold falls actually matter.

**Q: What is PANNs and why is it optional?**
PANNs are large pretrained audio neural networks; their embeddings serve as an external-feature baseline. It's optional because the assignment's object is the project's own ConvAE representation, and PANNs plus LOF (0.702) is only a sanity check, slightly below ConvAE plus LOF. (If asked why PANNs is not in the written report: the report's canonical run had the optional PANNs baseline disabled; the deck uses a later run of the same deterministic pipeline with PANNs enabled — all shared result metrics are identical between the two runs.)

**Q: Where does the data come from, and what does a fault sound like?**
The recordings follow the MIMII water-pump benchmark setup: four pump machine IDs recorded as 10-second clips at 16 kHz, organized into normal-training, normal-test, and anomaly folders. Anomalies are real pump faults — things like leakage, clogging, and imbalance — which typically change the harmonic texture or periodicity of the pump sound rather than its overall loudness.

**Q: Why mean aggregation over windows, and not max?**
A sensitivity check was run over four strategies — max, mean, 90th percentile, and median — combined with different threshold metrics. Mean was the most stable choice: max is very sensitive to a single noisy window, which inflates false positives, while mean pools evidence across all nine windows of a file. Since faults in this data are persistent rather than transient, averaging does not wash them out.

**Q: Why a 64-dimensional latent?**
It's a balance: small enough to force the bottleneck to abstract (128× compression, so the AE cannot just copy the input), large enough that validation MSE stays low and reconstructions are faithful. It also keeps the sample-to-dimension ratio comfortable for the classical detectors fitted on the latent space.

**Q: Why does the validation loss have that little peak early on (slide 9)?**
Two words: **early** and **harmless**. It's at epoch 3, while the learning rate is still high, so the weights move in big steps — and the model is only trained to lower the *training* loss, so a big early step can help training but briefly nudge the (merely observed) validation loss up. It's harmless because it's tiny — a blip on a loss ten times larger — and it never comes back: after epoch 5 both curves fall together and stay locked. If it were overfitting, validation would keep drifting up; it doesn't.

**Q: What exactly is an epoch? (theory)**
An **epoch is one complete pass of the training over the entire training set**. A neural network learns by gradient descent: it doesn't see all the data at once, but in small **batches** — after each batch it nudges its weights a little to lower the loss. When every batch has been seen once, i.e. the whole dataset has passed through the network, that is one epoch. One pass is never enough to converge, so training repeats for many epochs, each one moving the weights closer to a minimum of the loss. Too few epochs → underfitting (the model hasn't learned enough); too many → risk of overfitting, which is why **early stopping** is used. Training ran for 52 epochs.

**Q: What do "early stopping" and "reduce-on-plateau scheduler" mean? (slide 9)**
They're two automatic training controllers that both watch the **validation loss**. **Early stopping** ends training once the validation loss stops improving for a set number of epochs (its "patience"), instead of running a fixed count — this prevents overfitting and wasted time, and it's why training stopped at 52 epochs rather than some round number. **Reduce-on-plateau** manages the **learning rate** — the size of each weight-update step. It starts large for fast early progress, and whenever the validation loss flattens out (a "plateau"), the scheduler cuts the learning rate — here it halves it (factor 0.5) after 5 flat epochs, going 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 in this run — so the model takes smaller, finer steps to settle into a better minimum. Think coarse tuning first, then fine tuning, then stop.

**Q: What is a mel spectrogram, exactly? (theory)**
Start from the **waveform** — amplitude over time. A **spectrogram** is produced by the Short-Time Fourier Transform (STFT): slice the signal into short overlapping frames and run an FFT on each, giving how much energy sits at each frequency at each moment — a 2D **time × frequency** image. A **mel** spectrogram remaps the linear frequency axis onto the **mel scale**, a perceptual scale (roughly linear at low frequencies, logarithmic at high ones) that matches how human hearing resolves pitch; concretely, the FFT bins are grouped by a bank of overlapping triangular filters — here **128 mel bands**. Finally **log-mel** takes the logarithm of those energies (dB) to compress the huge dynamic range. The result is a compact 128 × 64 image where the low-frequency structure — where a pump's harmonics live — is well resolved.

(Simpler words:
Start with the **waveform**, which is just the sound shown as how loud it is over time.
To better understand the sound, it is split into many small, overlapping pieces. For each piece, a mathematical tool called the **Fast Fourier Transform (FFT)** finds out **which frequencies (low and high pitches) are present and how strong they are**.
Putting all of these results together creates a **spectrogram**. You can think of it as a picture where:
* **left to right** = time,
* **bottom to top** = frequency (pitch),
* **color or brightness** = how strong each frequency is.
A **mel spectrogram** changes the frequency scale so it matches how humans hear sound. People hear small differences in **low pitches** better than in **high pitches**, so the mel scale gives more detail to low frequencies and less to high ones. Instead of using every FFT frequency directly, nearby frequencies are grouped into **128 mel bands**.
Finally, the **logarithm** of the energy values is taken (called a **log-mel spectrogram**). This reduces the difference between very loud and very quiet sounds, making the important patterns easier to see and work with.
The final result is a **128 × 64 image** that summarizes the sound. It clearly shows important low-frequency patterns, such as the repeating harmonics produced by a pump, while keeping the data compact and easy for a machine learning model to use.)
---

## Other Possible question
One likely question per content slide (dividers 4, 7, 11 and the title are skipped).

**Slide 2 — Why detect faults from *sound* rather than vibration or temperature sensors?**
Audio is cheap and non-invasive: a single microphone captures the whole machine with no contact sensors to mount per point. Rotating faults change the sound before catastrophic failure, so it's a practical early-warning signal — and the assignment's dataset is audio.

**Slide 3 — Why is machine ID 06 missing from training? Isn't that unfair to the model?**
It comes from the dataset itself: the `train-normal` folder contains only IDs 00, 02, and 04, while ID 06 appears only in the tuning and test folders. Rather than hiding that, the evaluation keeps it, because it is a *realistic* stress test: in deployment a new but healthy machine can appear, so this checks whether the model generalizes "normal" instead of memorizing the training machines. It turned out to be the main source of false positives, analyzed later — and the Protocol 2 ablation tests what happens when some ID 06 normals are added to training.

**Slide 5 — Why a mel spectrogram and not the raw FFT (or waveform)?**
The FFT *is* used — the mel spectrogram is built on top of it. The mel step just regroups the 513 linear FFT bins into 128 perceptual bands: it cuts the input size 4×, spends resolution on the low-frequency harmonics where a pump's signature lives, and is robust to small speed shifts. It's also the standard MIMII/DCASE representation, so results stay comparable.

**Slide 6 — Why a fixed dB reference of 1.0 instead of per-image max normalization?**
Per-image max rescales every spectrogram to its own peak, erasing *absolute loudness* — but a quieter or louder pump is exactly the difference a one-class model should see. Fixed ref = 1.0 keeps absolute intensity; values are then clipped to the 1st/99th percentile and mapped to [0, 1].

**Slide 6 — Why clip to the 1st/99th percentile?**
The sound is measured in decibels, then those values are rescaled into a fixed 0-to-1 range before feeding them to the network. The issue is that a few samples are extreme — some very low, some very high — and sit far from where most of the data is. If the 0-to-1 range is built from the true min and max, those rare extremes become the endpoints and all the normal values get compressed into a narrow band in the middle, killing the contrast between sounds. So instead of min/max, the **1st and 99th percentile** serve as endpoints: 98% of the data spreads across the full range, and anything beyond those points is simply clipped — pinned to 0 or 1 — so it can't stretch the scale. The result is that normal sound keeps good contrast and no single rare value distorts the normalization. *One line: the top and bottom 1% are cut so a few extreme values can't compress all the normal signal into the middle.*

**Slide 8 — Why a convolutional autoencoder and not a classifier?**
A classifier needs fault labels, which the one-class setting doesn't have. The AE learns to compress and rebuild *normal* sound only; the 128× bottleneck forces it to keep essential normal structure, and deviations show up as high reconstruction error or off-manifold latents. The convs exploit the 2D time–frequency structure of the spectrogram.

**Slide 9 — How do you know the model isn't over- or under-fitting?**
Training and validation MSE track each other closely and both plateau around 0.0073. Early stopping halts when validation stops improving, and reduce-on-plateau lowers the learning rate to fine-tune. If it were overfitting, validation would rise while training fell — it never does.

**Slide 10 — If the AE reconstructs anomalies well too, isn't it failing?**
No — reconstructing well is what it's trained to do, and it generalizes. The point is the opposite: because the decoder rebuilds many faults acceptably, reconstruction *error alone* is a weak anomaly score. That's exactly why the **latent space** is also used with density detectors like LOF.

**Slide 12 — The supervised models score higher — why not just use them?**
They use anomaly labels in training and tune their threshold on labeled data — which the one-class assignment forbids and real deployments don't have. They solve an easier problem, so they're reported only as **diagnostic baselines** that prove signal exists; the headline is the unsupervised methods.

**Slide 13 — Why these three detectors, and how do they differ?**
They capture three notions of "outlier": **One-Class SVM** draws one global boundary around normal; **LOF** uses *local* density, flagging points in locally sparse regions; **Isolation Forest** scores how easily random splits isolate a point. All share the frozen 64-d latent and "higher = more anomalous," so one eval pipeline fits all — and LOF's local view wins.

**Slide 14 — How do you pick the threshold without peeking at the test set?**
It's tuned only on a separate **tuning split**, optimizing balanced accuracy; the test set is untouched until the final measurement, and tuning and test files never overlap (file-level split). On top of that, AUC-ROC and AUC-PR are threshold-free, so those numbers hold wherever the line is drawn.
