import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

labels = learn.dls.vocab
print(labels)

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(source='webcam', shape=(192,192)),
    outputs=gr.Label(num_top_classes=3)
)
iface.launch(share=True)
