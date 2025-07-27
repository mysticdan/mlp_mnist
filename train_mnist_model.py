import jax
import jax.numpy as jnp
from jax import random
import pickle
from datasets import load_dataset
from mlp_model import MLP, compute_loss

def save_model(model, path="mlp_mnist_model.pkl"):
    data = {
        "params": model.params,
        "activation_names": [a.__name__ for a in model.activation]
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def preprocess(dataset):
    images = jnp.array(dataset["image"]).astype(jnp.float32).reshape(len(dataset), -1) / 255.0
    labels = jnp.array(dataset["label"]).astype(jnp.int32)
    return images, labels

def evaluate_model(model, x, y, batch_size=256, loss_name="cross_entropy"):
    total_loss = 0.0
    total_correct = 0
    num_samples = x.shape[0]
    steps = num_samples // batch_size

    for i in range(steps):
        xb = x[i * batch_size:(i + 1) * batch_size]
        yb = y[i * batch_size:(i + 1) * batch_size]

        model.key, subkey = random.split(model.key)

        logits = model.predict(xb)
        loss = compute_loss(model.params, model.activation, xb, yb, loss_name=loss_name, num_classes=10, regularization="", lambda_=0.0, dropout_rate=0.0, key=subkey)
        preds = jnp.argmax(logits, axis=1)
        targets = jnp.squeeze(yb).astype(int)
        correct = jnp.sum(preds == targets)

        total_loss += float(loss) * xb.shape[0]
        total_correct += int(correct)

    avg_loss = total_loss / num_samples
    accuracy = total_correct / num_samples
    return avg_loss, accuracy

def train_model(model, x_train, y_train, x_test=None, y_test=None, batch_size=32, epochs=200, lr=5e-3, loss_name="cross_entropy", regularization="l1_l2", lambda_reg=1e-4, patience=20, min_delta=0.0001):
    num_train = x_train.shape[0]
    steps = num_train // batch_size

    key = random.key(0)

    lr_schedule = lr * jnp.exp(-0.015 * jnp.arange(epochs))

    best_test_loss = float('inf')
    best_epoch = -1
    best_params = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        current_lr = lr_schedule[epoch]
        key, subkey = random.split(key)
        idx = random.permutation(subkey, num_train)
        x_shuffled = x_train[idx]
        y_shuffled = y_train[idx]

        for i in range(steps):
            x_batch = x_shuffled[i * batch_size:(i + 1) * batch_size]
            y_batch = y_shuffled[i * batch_size:(i + 1) * batch_size]

            model.train(xs=x_batch, ys=y_batch, loss_name=loss_name, regularization=regularization, lambda_reg=lambda_reg, lr=current_lr)

        train_loss, train_acc = evaluate_model(model, x_train, y_train, loss_name=loss_name)

        if x_test is not None and y_test is not None:
            test_loss, test_acc = evaluate_model(model, x_test, y_test, loss_name=loss_name)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

            if test_loss < (best_test_loss - min_delta):
                best_test_loss = test_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_params = model.params
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}!")
                    print(f"No improvement for {patience} consecutive epochs")
                    print(f"Best test loss: {best_test_loss:.4f} at epoch {best_epoch}")
                    if best_params is not None:
                        model.params = best_params
                    return
        else:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    if best_params is not None:
        model.params = best_params
        print(f"\nTraining completed. Best test loss: {best_test_loss:.4f} at epoch {best_epoch}")


def main():

    ds = load_dataset("mnist")

    train_data = ds['train']
    test_data = ds['test']

    x_train, y_train = preprocess(train_data)
    x_test, y_test = preprocess(test_data)

    mlp = MLP(layer_sizes=[784, 512, 256, 128, 64, 10], activation=["relu", "relu", "relu", "relu", "identity"], dropout_rate=0.3)
    train_model(mlp, x_train, y_train, x_test, y_test)

    save_model(model=mlp)

if __name__ == "__main__":
    main()