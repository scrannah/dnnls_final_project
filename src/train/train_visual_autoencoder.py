import torch

def train_visual_autoencoder(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        criterion_percep,
        criterion_ctx,
        beta,
        device,
        num_epochs
    ):

    epoch_losses = []
    model.train()


    for epoch in range(num_epochs):

        running_loss = 0.0
        running_kl = 0.0

        for images in train_dataloader:

            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, x_context, mu, logvar = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion_images(x_content, images)
            ctxloss = criterion_ctx(x_context, images)
            perceptual_loss = criterion_percep(x_content, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1).mean()


            backprop_loss = loss + beta*kl_loss # + perceptual_loss + ctxloss

            # Backpropagation
            optimizer.zero_grad()
            backprop_loss.backward()
            optimizer.step()

            running_loss += backprop_loss.item() * images.size(0)
            running_kl += kl_loss.item() * images.size(0)


        epoch_loss = running_loss / len(train_dataloader.dataset)
        kl_loss = running_kl / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}] AE Loss: {epoch_loss:.4f}")
        print(f"Recon: {loss.item():.4f} | KL: {kl_loss.item():.4f}")


    return epoch_losses
