import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from mlp_model import MLP, activation_map
import cv2

CANVAS_SIZE = 280 
MODEL_PATH = "mlp_mnist_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    mlp = MLP.__new__(MLP)
    mlp.params = data["params"]
    mlp.activation = tuple(activation_map["identity"] if name == "<lambda>" else activation_map[name] for name in data["activation_names"])
    mlp.key = jax.random.key(0)
    return mlp

def preprocess_img(image):
    img = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    st.image(img, caption="Processed (28x28)", width=50)
    flat = (img / 255).reshape(-1)
    return jnp.array(flat, dtype=jnp.float32)

def predict_digit(mlp, img_array):
    logits =  mlp.predict(x=img_array[None, :])
    pred = int(jnp.argmax(logits, axis=1)[0])
    st.write(logits)
    probs = jax.nn.softmax(logits)[0]
    st.write(f"**Prediction:** {pred} (Confidence: {probs[pred]:.2%})")
    return pred

def main():
    st.title("MNIST Digit Recognizer")
    st.write("Draw a digit (0-9) below and click **Predict**.")

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",  
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    mlp = load_model()
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            gray = np.mean(img_data[:, :, :3], axis=2).astype(np.uint8)
            st.image(gray, caption="Original", width=100)
            x = preprocess_img(gray)
            pred = predict_digit(mlp, x)
            st.write(f"**Prediction:** {pred}")
        else:
            st.write("Please draw a digit first.")

if __name__ == "__main__":
    main()
