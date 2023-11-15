import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Charger le jeu de données MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Créer un modèle séquentiel
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=3)

# Sauvegarder le modèle
model.save('ecriture.model')

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('ecriture.model')

# Évaluer le modèle sur le jeu de test
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Perte = {loss}")
print(f"Précision = {accuracy}")

# Charger les images à partir du dossier asset_src et faire des prédictions
img_num = 0
while os.path.isfile(f"asset_src/{img_num}_hand.jpeg"):
    img = cv2.imread(f"asset_src/{img_num}_hand.jpeg")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"Cette image asset_src/{img_num}_hand.jpeg est probablement un {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    #plt.show()
    img_num += 1
