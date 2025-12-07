
def train_visual_autoencoder(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        device,
        num_epochs
    ):

    epoch_losses = []
    epoch_counter = -1 # This is-1 because for epoch in range starts at 0 so this makes it equal to that don't think about it

    model.train()


    for epoch in range(num_epochs):

        running_loss = 0.0

        for images in train_dataloader:

            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, _ = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion_images(x_content, images) # I ONLY WANT CONTENT FOR VISUAL TRAINING, THIS TAKES CONTENT FROM THE DECODER ONLY

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)


        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)
        epoch_counter = epoch_counter + 1 # just to track epochs when printing
        print(f"[Epoch {epoch_counter}] AE Loss: {epoch_loss:.4f}")

        return epoch_losses
