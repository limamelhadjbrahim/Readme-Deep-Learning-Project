# Readme-Deep-Learning-Project

````markdown
# Emotion Text Classifier – Fine-tuning with Hugging Face 

This project demonstrates how to fine-tune a transformer model for **emotion classification** using the **dair-ai/emotion** dataset.  
The training is performed using the Hugging Face `Trainer` API, with evaluation metrics (accuracy + F1 macro) and deployment via Hugging Face Hub.

---

## Features

- Uses the **dair-ai/emotion** dataset (train/validation/test)
- Fine-tunes the pretrained model: `michellejieli/emotion_text_classifier`
- Uses `Trainer` with:
  - Accuracy and Macro-F1 evaluation
  - FP16 automatic mixed precision (on GPU)
- Saves and reloads the model locally
- Pushes the trained model to the Hugging Face Hub
- Provides an inference pipeline demo

---

## Dataset

Dataset: **dair-ai/emotion**  
Labels:
- 0: sadness  
- 1: joy  
- 2: love  
- 3: anger  
- 4: fear  
- 5: surprise  

---

##  Installation

Clone the project and install dependencies:

```bash
pip install -r requirements.txt
````

---

##  Training the model

Simply run the Python script or the Colab notebook:

```bash
python train.py
```

Or open the Colab version:

 **Colab Notebook:**
[https://colab.research.google.com/drive/1MuR6pqptqGWTckL0z3Dxk-ODHqBS7jyb?usp=sharing](https://colab.research.google.com/drive/1MuR6pqptqGWTckL0z3Dxk-ODHqBS7jyb?usp=sharing)

---

##  Evaluation

After training, the script evaluates the model on the test set:

* **Accuracy**
* **F1-macro**

---

##  Saving & Loading the Model

The fine-tuned model is saved in:

```
hf_emotion_finetuned/
```

You can reload it with:

```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="hf_emotion_finetuned",
    tokenizer="michellejieli/emotion_text_classifier"
)

print(pipe("I love this!"))
```

---

## ☁️ Push to Hugging Face Hub

Use your HF token:

```python
from huggingface_hub import login
login()
```

Then push:

```python
trainer.push_to_hub("emotion-classifier-dair")
```

---

## Live Demo

You can test the deployed model here:

 **Hugging Face Space Demo:**
*Emotion Text Classifier - Space by Limamhb*

---

## Requirements

See `requirements.txt` in this repository.

---

## License

This project is for educational use.
Model and dataset follow their original licenses on Hugging Face.

````

---


Si tu veux, je peux aussi te générer :
✅ un `train.py` propre
✅ un `inference.py`
✅ une structure de repo complète
Souhaites-tu que je te génère les fichiers Python aussi ?
