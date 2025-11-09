# EEG Seizure Detection Using CNN and Autoencoder

This repository provides an end-to-end deep learning pipeline for classifying epileptic seizures from EEG recordings using spectrogram-based Convolutional Neural Networks (CNNs) and an unsupervised convolutional autoencoder for feature pretraining.

The implementation supports open EEG datasets such as CHB-MIT and TUH Seizure Corpus and is designed for research, experimentation, and reproducibility.

## Features

* Complete Python pipeline for EEG seizure detection
* EDF loader using MNE with optional annotation parsing
* Windowing and label assignment based on seizure intervals
* Mel-spectrogram generation using Librosa
* Convolutional Autoencoder for unsupervised representation learning
* CNN classifier trained on encoder embeddings
* Evaluation metrics including classification report and ROC-AUC
* Modular structure allowing extension and customization

## Requirements

* Python 3.8+
* Required packages:

  * mne
  * librosa
  * numpy
  * scipy
  * matplotlib
  * torch
  * torchvision
  * tqdm
  * scikit-learn

Install dependencies with:

```
pip install mne librosa numpy scipy matplotlib torch torchvision tqdm scikit-learn
```

## Dataset Setup

### CHB-MIT

Download the CHB-MIT dataset and place EDF files under:

```
data/chb-mit/
```

Subfolders are supported.

### TUH Seizure Corpus

To use TUH, adapt the EDF file path and annotation reader if needed.

### Optional Seizure Annotation CSV

You may provide a CSV file with seizure intervals:

```
filename,start_sec,end_sec
```

Pass this file using the `--ann_csv` argument.

## Usage

Run the full training pipeline:

```
python EEG_Seizure_Classification_Pipeline.py \
    --data_dir data/chb-mit \
    --out_dir runs/run1 \
    --ae_epochs 10 \
    --clf_epochs 20
```

Key arguments:

* `--data_dir`: directory containing EDF files
* `--ann_csv`: optional annotation file
* `--max_files`: limit the number of EDFs processed
* `--batch_size`: batch size for training
* `--ae_epochs`: autoencoder training epochs
* `--clf_epochs`: classifier training epochs
* `--ae_out` / `--clf_out`: model checkpoint paths

## Repository Structure

```
├── EEG_Seizure_Classification_Pipeline.py   # Main training script
├── runs/                                    # Output directory for models and logs
├── data/                                    # EEG datasets (CHB-MIT/TUH)
└── README.md                                # Project documentation
```

## Model Architecture

### Autoencoder

* Convolutional encoder
* Latent embedding layer
* Transposed convolution decoder

### Classifier

* Encoder reused from autoencoder
* Fully connected layers for binary seizure classification

## Evaluation

The script computes:

* ROC-AUC
* Precision, Recall, F1-score
* Confusion matrix

Evaluation is performed on a held-out test set created via stratified splitting.

## Notes

* The autoencoder step is optional but improves representation learning.
* Spectrogram parameters such as window size, mel bins, and hop length can be tuned.
* Multi-channel spectrograms are supported; channels are treated as CNN input channels.

## License

This project is released for research and educational purposes. Adjust and extend freely as needed.

## Contact

For questions or improvements, feel free to open an issue or pull request.
