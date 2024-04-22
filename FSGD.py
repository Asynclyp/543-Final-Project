import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.applications import mobilenet_v2
from keras.src.applications.mobilenet_v2 import MobileNetV2
import time
import json
import numpy as np

# Configure plotting aesthetics
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['axes.grid'] = False

# Initialize the pre-trained MobileNetV2 model
pretrained_model = MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# Define loss object for the adversarial pattern generation
loss_object = tf.keras.losses.CategoricalCrossentropy()

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = mobilenet_v2.preprocess_input(image)
    return image[None, ...]

def get_imagenet_label(probs):
    """Return the ImageNet label and confidence from prediction probabilities."""
    decoded = mobilenet_v2.decode_predictions(probs, top=1)[0][0]
    return decoded[1], decoded[2]

def decode_predictions(prediction):
    label_map = tf.keras.utils.get_file('imagenet_class_index.json', 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
    with open(label_map) as f:
        class_labels = json.load(f)
    predicted_class_index = np.argmax(prediction)
    return class_labels[str(predicted_class_index)], predicted_class_index

def create_adversarial_pattern(input_image, input_label):
    """Create adversarial perturbation based on model predictions."""
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

def display_image(image, description):
    """Display an image with a description."""
    label, confidence = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)  # Normalize to [0,1]
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                     label, confidence * 100))
    plt.show()

def main():
    images = [
        'panda.jpg', 'bookcase.jpg', 'corn.jpg', 'cowboyhat.jpg', 'dog.jpg',
        'forklift.jpg', 'limo.jpg', 'pajamas.jpg', 'rugbyball.jpg', 'shovel.jpg',
        'submarine.jpg', 'tennisracket.jpg', 'vacuum.jpg', 'waterbottle.jpg'
    ]

    for image_path in images:
        image = preprocess_image(image_path)
        image_probs = pretrained_model.predict(image)
        image_class, class_confidence = get_imagenet_label(image_probs)
        display_image(image, f'{image_class} : {class_confidence * 100:.2f}% Confidence')

        # Prepare label for adversarial pattern generation

        label_index = np.argmax(image_probs)
        label = tf.one_hot(label_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        perturbations = create_adversarial_pattern(image, label)
        plt.imshow(perturbations[0] * 0.5 + 0.5);
        epsilons = [0, 0.01, 0.1, 0.15]
        descriptions = ['Input'] + [f'Epsilon = {eps:.3f}' for eps in epsilons[1:]]

        for eps, desc in zip(epsilons, descriptions):
            adv_x = image + eps * perturbations
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            display_image(adv_x, desc)
            time.sleep(3)

if __name__ == '__main__':
    main()
