import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Load pre-trained model
pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
images = ['panda3.jpg', 'bookcase.jpg', 'corn.jpg', 'cowboyhat.jpg', 'dog.jpg', 'forklift.jpg', 'limo.jpg', 'pajamas.jpg', 'rugbyball.jpg', 'shovel.jpg', 'submarine.jpg', 'tennisracket.jpg', 'vacuum.jpg', 'waterbottle.jpg']
# Load image
def decode_predictions(prediction):
    label_map = tf.keras.utils.get_file('imagenet_class_index.json', 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
    with open(label_map) as f:
        class_labels = json.load(f)
    predicted_class_index = np.argmax(prediction)
    return class_labels[str(predicted_class_index)], predicted_class_index
def pgd_attack_untargeted(image, epsilon, alpha, num_steps, original):
    adv_image = tf.identity(image)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = pretrained_model(adv_image)
            original_class_prob = tf.nn.softmax(prediction)[0, original]
            loss = -original_class_prob  # Minimize the probability of the original class
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)
        adv_image = adv_image + alpha * signed_grad
        perturbation = tf.clip_by_value(adv_image - image, -epsilon, epsilon)
        adv_image = image + perturbation
        adv_image = tf.clip_by_value(adv_image, -1, 1)
    return adv_image


for i in range(len(images)):
    image_path = images[i]
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image[tf.newaxis])
    # Parameters for PGD attack
    epsilon = 0.1
    alpha = 0.01
    num_steps = 20
    origin_predict, origin_class = decode_predictions(pretrained_model.predict(image))

    # Generate adversarial image
    adversarial_image = pgd_attack_untargeted(image, epsilon, alpha, num_steps, origin_class)

    # Get the predicted class for the adversarial image
    adversarial_prediction = pretrained_model.predict(adversarial_image)
    adversarial_class = np.argmax(adversarial_prediction[0])

    # Display results
    print("Original Image:")
    print("Predicted Class:", origin_predict)
    print()

    print("Adversarial Image:")
    print("Predicted Class:", decode_predictions(adversarial_prediction))
    print()

    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Adversarial Image")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(adversarial_image[0]))
    plt.axis('off')

    plt.show()









