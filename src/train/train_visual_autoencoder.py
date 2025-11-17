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
            reconstructed = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion(reconstructed, images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)

    return epoch_losses
