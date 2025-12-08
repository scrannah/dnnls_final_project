
def train_visual_autoencoder(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        criterion_percep,
        device,
        num_epochs
    ):

    epoch_losses = []
    model.train()


    for epoch in range(num_epochs):

        running_loss = 0.0

        for images in train_dataloader:

            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, _ = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion_images(x_content, images)
            perceptual_loss = criterion_percep(x_content, images)# I ONLY WANT CONTENT FOR VISUAL TRAINING, THIS TAKES CONTENT FROM THE DECODER ONLY
            backprop_loss = loss + perceptual_loss
            # Backpropagation
            optimizer.zero_grad()
            backprop_loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)


        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}] AE Loss: {epoch_loss:.4f}")

    return epoch_losses
