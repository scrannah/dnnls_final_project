import torch


def train_visual_autoencoder(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        criterion_percep,
        criterion_ctx,
        beta,
        kl_anneal_epoch,
        lambda_percep,
        device,
        num_epochs=1
):
    epoch_losses = []
    epoch_perceploss = []
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_percep = 0.0

        for images in train_dataloader:
            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, x_context = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            loss = criterion_images(x_content, images)
            ctxloss = criterion_ctx(x_context, images)
            perceptual_loss = criterion_percep(x_content, images)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1).mean()

            # kl_weight = min(beta, beta * global_step / kl_anneal_epoch)

            backprop_loss = loss + lambda_percep * perceptual_loss + ctxloss

            # Backpropagation
            optimizer.zero_grad()
            backprop_loss.backward()
            optimizer.step()

            running_loss += backprop_loss.item() * images.size(0)
            running_percep += perceptual_loss.item() * images.size(0)

            # running_kl += kl_loss * images.size(0)

            # print(f"KL weight: {kl_weight}")
            # print(f"Effective KL: {(kl_weight * kl_loss).item()}")

        epoch_loss = running_loss / len(train_dataloader.dataset)
        # kl_loss = running_kl / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch + 1}]  AE Loss: {epoch_loss:.4f}")
        avg_percep = running_percep / len(train_dataloader.dataset)
        epoch_perceploss.append(avg_percep)
        print(f"[Epoch {epoch + 1}] Perceptual Loss: {avg_percep:.4f}")
        # print(f"Recon: {loss.item():.4f} | KL: {kl_loss.item():.4f}")

        return epoch_losses, epoch_perceploss
