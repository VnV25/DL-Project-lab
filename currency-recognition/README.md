# Note Reader: Indian Currency Recognition with Deep Learning

A complete deep learning mini project that recognises Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) from a phone or webcam photo and **speaks the answer in English, Hindi or Marathi**. It uses transfer learning (MobileNetV3, pretrained on ImageNet) and is built to work on **new photos it has never seen**, not just on its test set.

Features of the web app:
- Webcam capture, file upload or paste from clipboard
- Large, colour-coded result and a spoken answer (English / हिन्दी / मराठी)
- Says **"not sure, please retake"** instead of guessing when confidence is low
- Says **"no note found"** when the photo has no note (if you add a background class)
- **Running total**: identify notes one by one and add them up
- Works on your phone through a share link

---

## 1. Setup (one time)

You need **Python 3.10, 3.11 or 3.12**. A GPU is optional; everything runs on a normal laptop CPU.

```bash
cd currency-recognition
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU and want to use it, install PyTorch first with the command from https://pytorch.org/get-started/locally/ and then run `pip install -r requirements.txt`.

## 2. Get the dataset

**Already included in this repository:** the cleaned, resized dataset (`data/processed/`, 1,746 images and the train/val/test split in `splits.csv`) and the trained model (`models/currency_model.pt`). You can run `python app.py` or `python evaluate.py` straight after setup, or retrain with `python train.py`. The original full-size photos (6.9 GB) are too large for GitHub; download them as below only if you want to run `prepare_data.py` again.

Recommended: **Indian Currency Dataset** on Mendeley Data: https://data.mendeley.com/datasets/8ckhkssyn3
It has 1,786 phone photos of notes against various backgrounds, lighting conditions and angles, with no artificial augmentation (which matters, see section 7).

1. Click **Download All** on that page and unzip it.
2. Move the extracted folders into `data/raw/`.

Folder names don't need to follow a fixed format. Anything containing the value works (`500`, `Rs_500_new`, `100 old`, `Tennote`, `5Hundrednote` ...). Old and new designs of the same value are merged into one class.

**Strongly recommended, adds the "no note" answer:** create `data/raw/background/` and put 100–200 photos in it that contain no note: your table, hand, keyboard, wallet, bag, a sheet of paper, the floor.

**Optional, for even better generalisation:** combining a second dataset from a different source (another camera, other people) helps a lot. Just put its folders in `data/raw/` as well. Avoid datasets that are already augmented (rotated/flipped copies of the same photo), because the copies leak between train and test.

## 3. Run the project

```bash
python prepare_data.py    # resize images, remove duplicates, split train/val/test   (~2-5 min)
python train.py           # transfer learning                                         (~20-40 min on CPU, a few min on GPU)
python evaluate.py        # accuracy, confusion matrix, robustness tests, charts      (~5-10 min on CPU)
python app.py             # opens the web app in your browser
```

Use the app on your **phone camera**: run `python app.py --share` and open the `https://....gradio.live` link it prints on your phone.

Classify photos from the command line:

```bash
python predict.py photo1.jpg photo2.jpg
python predict.py photo.jpg --speak --lang hi
```

## 4. Test on truly unseen data (do this!)

Take 5–10 photos of each note **yourself**, in different rooms and lighting, on a table and in your hand, once folded, once at an angle. Put them in `my_photos/10/`, `my_photos/20/` ... `my_photos/500/` and run `python evaluate.py` again. These photos are never used for training, so this number is the honest answer to "does it work on new data?".

If accuracy on your photos is clearly lower than on the test set, the usual fix is more variety in training: add ~20 of *different* own photos per note to `data/raw/` (not the same ones as in `my_photos/`), then run `prepare_data.py` and `train.py` again.

## 5. Results for your report

After `evaluate.py`, the `reports/` folder contains:

| File | What it shows |
|---|---|
| `results.md` | Summary of all numbers, ready to copy into your report |
| `training_curves.png` | Loss and accuracy per epoch (stage 1 vs fine-tuning) |
| `confusion_matrix.png` | Which notes get confused with which |
| `mistakes.png` | The test photos the model got wrong |
| `robustness.png` | Accuracy in dim light, blur, rotation, partly covered, low quality ... |
| `my_photos_results.png` | Predictions on your own photos (green right, red wrong, orange "not sure") |

## 6. Project structure

```
currency-recognition/
├── config.py              all settings (paths, model, epochs, confidence threshold)
├── prepare_data.py        step 1: read any dataset layout, resize, de-duplicate, split
├── train.py               step 2: two-stage transfer learning, saves the best model
├── evaluate.py            step 3: test accuracy, robustness tests, own-photo test, charts
├── app.py                 step 4: Gradio web app with camera, voice and running total
├── predict.py             command-line prediction
├── currency/
│   ├── labels.py          turns folder names like "5Hundrednote" into labels
│   ├── data.py            dataset class and the real-world augmentations
│   ├── model.py           MobileNetV3 / EfficientNet with a new classification layer
│   ├── predictor.py       loads the model, test-time augmentation, confidence check
│   └── speech.py          spoken answers in English, Hindi and Marathi
├── data/raw/              your downloaded dataset goes here
├── my_photos/             your own photos for the unseen-data test
├── models/                trained model (created by train.py)
└── reports/               charts and results (created by train.py and evaluate.py)
```

## 7. How the project makes the model work on unseen photos

1. **Transfer learning.** The network starts from features learned on 1.2 million ImageNet photos, so it already understands edges, textures and colours in real-world pictures. Only a small part is learned from our smaller dataset.
2. **Real-world augmentation** (`currency/data.py`). During training every photo is randomly rotated (notes can be held any way up), cropped (note partly out of frame), viewed at an angle, made darker or brighter, blurred, and partly covered (fingers). The model learns to ignore these, so they don't confuse it later. Hue is barely changed, because colour is one of the main ways to tell notes apart.
3. **Leak-free split** (`prepare_data.py`). Exact duplicates are removed, and near-identical photos (burst shots, re-saved copies) are detected with a perceptual hash and kept in the same split. Otherwise the test score would be inflated by photos the model effectively already saw.
4. **Test-time augmentation.** At prediction time the photo is classified in 4 orientations and the results are averaged, which makes the answer more stable.
5. **Confidence check and background class.** If the model isn't confident, or sees no note, it says so instead of guessing. For a tool meant for visually impaired users, a wrong answer is worse than "please retake".
6. **Honest evaluation.** Besides the normal test set, the project measures accuracy under difficult conditions and on your own photos.

## 8. Ideas to extend it

- **Count several notes in one photo** with an object detector (YOLOv8), trained on a dataset with bounding boxes such as the "Dataset of Indian (Rupees) and Thai (Baht) Banknotes with Annotations" on Mendeley Data.
- **Coins**: add classes for ₹1, ₹2, ₹5, ₹10 and ₹20 coins.
- **Mobile app**: export the model to TensorFlow Lite or ONNX and run it fully offline on a phone.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `prepare_data.py` skips a folder | Rename the folder so it contains the note value, e.g. `500`. |
| Training is very slow | Lower `FINETUNE_EPOCHS` in `config.py`, or close other programs. MobileNetV3 is already the fastest option. |
| `DataLoader` errors on Windows | `NUM_WORKERS` is already 0 on Windows; make sure you run the scripts with `python train.py`, not inside another tool. |
| No voice | gTTS needs internet. Offline, it falls back to your computer's built-in voice (usually English only). The text result is always shown. |
| Webcam not working | Allow camera access in the browser. On a phone, use the `--share` link (phones need https for the camera). |
| App says "not sure" too often | Lower `CONFIDENCE_THRESHOLD` in `config.py` (e.g. 0.6), or add more varied training photos. |
| Good test accuracy but bad on own photos | See section 4: add more varied photos of your own to `data/raw/` and retrain. |
