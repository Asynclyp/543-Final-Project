import tensorflow as tf
import numpy as np
import json
import random

# Load pre-trained model
pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
images = ['panda.jpg', 'panda3.jpg', 'bookcase.jpg', 'corn.jpg', 'cowboyhat.jpg', 'dog.jpg', 'forklift.jpg', 'limo.jpg',
          'pajamas.jpg', 'rugbyball.jpg', 'shovel.jpg', 'submarine.jpg', 'tennisracket.jpg', 'vacuum.jpg',
          'waterbottle.jpg']
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

def decode_predictions(prediction):
    label_map = tf.keras.utils.get_file('imagenet_class_index.json',
                                        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
    with open(label_map) as f:
        class_labels = json.load(f)
    predicted_class_index = np.argmax(prediction)
    return class_labels[str(predicted_class_index)], predicted_class_index


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


# Hill Climbing Parameters
step_size_alpha = 0.005
step_size_epsilon = 0.005
num_steps_per_update = 10


# Generate random values for epsilon and alpha
epsilon = random.uniform(0.001, 0.01)
alpha = random.uniform(0.001, 0.05)

best_alpha = None
best_epsilon = None
best_score = float('inf')
# Perform hill climbing

for _ in range(num_steps_per_update):

    new_alpha = alpha + np.random.uniform(-step_size_alpha, step_size_alpha)
    new_epsilon = epsilon + np.random.uniform(-step_size_epsilon, step_size_epsilon)

    # Evaluate the PGD attack with the perturbed alpha and epsilon
    score = objective_function((new_epsilon, new_alpha))

    # If the new parameters are better, accept the new parameters
    if score < best_score:
        best_alpha = new_alpha
        best_epsilon = new_epsilon
        best_score = score
        alpha = new_alpha
        epsilon = new_epsilon
        print("Best Score: ", best_score)

print("Optimal alpha:", best_alpha)
print("Optimal epsilon:", best_epsilon)