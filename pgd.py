import tensorflow as tf
import matplotlib.pyplot as plt
import json
from skopt import gp_minimize
from skopt.space import Real
import numpy as np


# Load pre-trained model
pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
images = ['panda.jpg', 'panda3.jpg', 'bookcase.jpg', 'corn.jpg', 'cowboyhat.jpg', 'dog.jpg', 'forklift.jpg', 'limo.jpg',
          'pajamas.jpg', 'rugbyball.jpg', 'shovel.jpg', 'submarine.jpg', 'tennisracket.jpg', 'vacuum.jpg',
          'waterbottle.jpg']
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
    # CHANGE THESE AS YOU WISH
    epsilon = 0.01
    alpha = 0.05
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




def decode_predictions(prediction):
    label_map = tf.keras.utils.get_file('imagenet_class_index.json',
                                        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
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
            loss = -original_class_prob
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)
        perturbation = alpha * signed_grad
        adv_image = tf.clip_by_value(adv_image + perturbation, image - epsilon, image + epsilon)
        adv_image = tf.clip_by_value(adv_image, -1, 1)
    return adv_image


#Define the objective function
def objective_function(params):
    epsilon, alpha = params
    perturbation_sum = 0
    misclassification_count = 0

    for i in range(len(images)):
        image_path = images[i]
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image[tf.newaxis])

        origin_predict, origin_class = decode_predictions(pretrained_model.predict(image))

        # Generate adversarial image
        adversarial_image = pgd_attack_untargeted(image, epsilon, alpha, 20, origin_class)

        # Get the predicted class for the adversarial image
        adversarial_prediction = pretrained_model.predict(adversarial_image)
        adversarial_class = np.argmax(adversarial_prediction[0])

        # Calculate perturbation
        perturbation_sum += np.sum(np.abs(adversarial_image - image))

        # Check if misclassification occurred
        if adversarial_class != origin_class:
            misclassification_count += 1

    # Average perturbation across all images
    avg_perturbation = perturbation_sum / len(images)
    misclassification_rate = misclassification_count / len(images)

    # Define a weight for the perturbation term
    # Adjust this weight to control the trade-off between misclassification and perturbation minimization

    # Calculate the combined objective function value
    combined_objective = -1000 * misclassification_rate + 5 * avg_perturbation
    print("avg_perturbation: ", avg_perturbation)
    print("misclassification_rate: ", misclassification_rate)
    print("current epsilon: ", epsilon)
    print("current alpha: ", alpha)

    # Return the combined objective function value
    #if -combined_objective >= 0:
    #    return combined_objective
    # punish bad misclassification rates
    if misclassification_rate <= 0.7:
        combined_objective += 1000000
    return combined_objective  # We maximize the negative combined objective function

# Define the search space for Bayesian optimization
param_space = [
    Real(0.001, 0.01, name='epsilon'),
    Real(0.001, 0.05, name='alpha')
]

# Perform Bayesian optimization
result = gp_minimize(
    objective_function,
    param_space,
    n_calls=10,  # Number of calls to objective function
    random_state=42
)

#Get the optimal parameters
best_epsilon, best_alpha = result.x

print("Optimal epsilon:", best_epsilon)
print("Optimal alpha:", best_alpha)

for i in range(len(images)):
    image_path = images[i]
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image[tf.newaxis])
    # Parameters for PGD attack
    epsilon = 0.005
    alpha = 0.005
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
