from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np

"""
Détails des imports :
    mnist : La DB d'images
    Input, Dense, Reshape, Flattent, Dropout, batchNormalization, Activation
    leakyRelu :  fonction d'activation ~ (x |-> max(0,x))
    Sequential, Model : Modèle de réseaux
    Adam : Optimizer
    pyplot : graphes
    numpy : fonctions mathématiques, matrices...
"""

class GAN():

    def __init__(self):

        #format des imgs et du bruit
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        
        optimizer = Adam(0.0002, 0.5)
        
        """
        On génére les modèles du discriminateur et du générateur.
        
        discriminateur : 
                fonction de perte : cross entropy
                optimizer : adam
                entrée : image (28x28x1)
                sortie : 

        generateur:
                fonction de perte : cross entropy
                optimizer : adam
                entrée : 100-dim bruit
                sortie : image (28x28x1)
        """

        #on génére et stock le modèle du discriminateur dans une variable
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        #on génére et stock le modèle du générateur dans une variable
        self.generator = self.build_generator()

        # ------ Construction du modèle combiné -------
        z = Input(shape=(self.latent_dim,)) # z input bruit (100)
        img = self.generator(z) # z --> img (28x28)
        self.discriminator.trainable = False # on entraine pas le disc. pour le modèle combiné
        validity = self.discriminator(img) #img -> validity (1) = 0/1
        
        self.combined = Model(z, validity) #definition du modèle combiné
        #L'évaluation du bruit (input du modèle) se fait sur la validité (output) de l'image généré
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        """
        Construction du générateur :
                - Couche dense 256 neurones, activation : ReLU
                - Couche dense 512 neurones, activation : ReLU
                - Couche dense 1024 neurones, activation : ReLU
                - Couche dense 784 neurones (28x28), activation : tanh 
        
        entrée : bruit (100-dim)
        sortie : reShape -> dimension de l'image (28x28x1)
        """

        model = Sequential()

        #Couches
        #-------- conv2D ? --------
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # print("\n\nGenerator : \n\n")
        # model.summary()#On affiche le résumé
        
        #input & output
        noise = Input(shape=(self.latent_dim,)) #(100, ) 100 par "convention" , on veut juste k >= 28
        img = model(noise)
        
        return Model(noise, img) #On renvoit notre model

    def build_discriminator(self):
        
        """
        Construction du discriminateur :
                - Couche flatten 784 neurones (28x28)
                - Couche dense 512 neurones, activation : ReLU
                - Couche dense 256 neurones, activation : ReLU
                - Couche dense 1 neurone, activation : sigmoid 
        
        entrée : Image (28x28x1)
        sortie : 0 l'image est fake, 1 l'image est réelle
        """

        model = Sequential()
        
        #Couches
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # print("\n\nDiscriminator : \n\n")
        # model.summary()#On affiche le résumé
        
        #input & output
        img = Input(shape=self.img_shape)
        validity = model(img)
        
        return Model(img, validity) #On renvoit notre model

    def train(self, epochs, batch_size=128, sample_interval=50):

        """
        On récupère les 60 000 images de la DB et on redimensionne tout ça
        On crée également 2 matrices resp. de 1 et de 0 pour comparer avec les résultats obtenus par le discriminateur.
        """
        
        (X_train, Y_train), (_, _) = mnist.load_data() 
        #matrice X := (60 000 x 28 x 28) Les images de chiffres 
        #matrice Y := (60 000, ) Les chiffres correspondants aux images

        max = 2 #On traite les chiffres de 0 à max
        sample = 5000 #le nb d'image pour chaque chiffre
        X_tab = np.empty(([(max+1)*sample, X_train.shape[1], X_train.shape[2]]), dtype=int)
        Y_tab = np.empty(([(max+1)*sample,]), dtype=int)
        
        for i in range(max+1):
            X_tab[sample*i:sample*(i+1)] = X_train[Y_train == i][0:sample,:]
            Y_tab[sample*i:sample*(i+1)] = Y_train[Y_train == i][0:sample]
        
        X_train = X_tab
        Y_train = Y_tab

        X_train = X_train / 127.5 - 1. #redimensionne [0,255] --> [-1, 1]
        X_train = np.expand_dims(X_train, axis=3) #matrice (60 000 x 28 x 28 x 1)
        valid = np.ones((batch_size, 1)) # (128 x 1) de 1
        fake = np.zeros((batch_size, 1)) # (128 x 1) de 0

        for epoch in range(epochs):

            """
            Sur une epoch :
                    -on prend un echantillon dans la DB et 2 bruits.
                    - 1) on entraine le discriminateur
                    - 2) on entraine le générateur (modèle combiné)
            """

            #On prends 128 (batch size) images aléatoirement parmis les (60 000) images de X_train (la DB)
            idx = np.random.randint(0, X_train.shape[0], batch_size) #(128, ) tableau d'entiers aleatoires entre 0 et 60 000
            imgs = X_train[idx] #(128 x 28 x 28 x 1) le lot (batch)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) #(128 x 100) le bruit
            ## On utilise le générateur pour générer des imgages à partir de notre bruit
            gen_imgs = self.generator.predict(noise) #(128 x 28 x 28 x 1)

            """
            On utilise le discriminateur pour reconnaitre les images
            On veut l'entrainer a reconnaitre les vrai img (DB) et les fausses (reseau génératif)
                - les images du lot venant de la DB & on compare avec valid (matrice de 1)
                - les images générés par le discriminateur & on compare avec fake (matrice de 0)
            On calcul la perte du discriminateur
            """

            print(self.discriminator.predict(imgs))

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #perte du discriminateur

            """
            Simultanément, on entraine le générateur (modèle combiné) pour améliorer les images créées
            On calcul la perte du générateur (modèle combiné).
            """
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))#(128 x 100) un autre bruit
            g_loss = self.combined.train_on_batch(noise, valid)#perte du générateur (modele combiné)

            #on affiche un message et toutes les 10000 (sample_interval) itérations, on enregistre un sample d'images
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.sample_images(epoch)
    def sample_images(self, epoch):
        
        #Fonction pour générer et enregistrer un échantillon d'images

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("img/images"+str(epoch)+".png")
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1, batch_size=128, sample_interval=500)
    #settings : epochs=100000, batch_size=128, sample_interval=10000