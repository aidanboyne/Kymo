#Helper functions for Symposium Demo

from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Batched Comparison Losses")
    plt.show()

# plot the first two latent dimensions of the encoded kymographs
def plot_latent(model, data, exp = True):

    with torch.inference_mode():
        for (i, x) in enumerate(data):
            z = model.embed(x.unsqueeze(0).to('cpu'))
            z = z.numpy()
            if i <= 100:
                plt.scatter(z[0][0], z[0][1], c='r', alpha = 0.8, label = ('sin' if i == 1 else None))
            elif i <= 200:
                plt.scatter(z[0][0], z[0][1], c='g', alpha = 0.8, label = ('pol' if i == 101 else None))
            elif i <= 300:
                plt.scatter(z[0][0], z[0][1], c='b', alpha = 0.8, label = ('rand' if i == 201 else None))
            else:
                plt.scatter(z[0][0], z[0][1], c='orange', alpha = (0.8 if exp == True else 0), label = ('exp' if (i == 301 and exp == True) else None))

        plt.xlabel(f'Latent Dim 1')
        plt.ylabel(f'Latent Dim 2')
        plt.legend(loc='upper right')
        plt.title('Validation Kymographs position in Latent Space')



#pick random kymographs to reconstruct and show
def plot_sample_im(model, data):
    plt.figure(figsize=(12,4))

    idx = np.random.randint(100, size=3)

    for i, id in enumerate(idx):
        x = data[id]

        with torch.inference_mode():
            _, _, x_hat = model(x.unsqueeze(0).to('cpu'))
            x_hat = x_hat.detach().to('cpu').numpy()

        plt.subplot(2,3,i+1)
        if i == 0:
            plt.ylabel('Input Kymographs\n Axis Position')
        plt.xlabel('Time')
        plt.imshow(x.squeeze(), cmap='plasma')

        # if i == 0:
        #     plt.title('Experimental Oscillatory')
        if i == 1:
            plt.title('Experimental Reconstructions')
        # if i == 2:
        #     plt.title('Artifical Negaive')
        # if i == 3:
        #     plt.title('Experimental')

        plt.subplot(2,3,i+4)
        if i == 0:
            plt.ylabel('Reconstructed Kymographs\n Axis Position')
        plt.xlabel('Time')
        plt.imshow(x_hat.squeeze(), cmap='plasma')

    print(f'ID: {idx}')

    plt.tight_layout()
    plt.show()