def train_visual_autoencoder(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        num_epochs
    ):

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images in dataloader:

            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, _ = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion(x_content, images) # I ONLY WANT CONTENT FOR VISUAL TRAINING, THIS TAKES CONTENT FROM THE DECODER ONLY

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}] AE Loss: {epoch_loss:.4f}") # this needs fixing to display correct epoch

    return epoch_losses
