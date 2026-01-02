import torch


def train_visual_autoencoder(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        criterion_percep,
        criterion_ctx,
        lambda_percep,
        lambda_ctx,
        device,
        num_epochs=1
):
    epoch_losses = []
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_percep = 0.0
        running_ctx = 0.0 # CONTEXT IS MSE
        running_l1 = 0.0 # CONTENT IS L1

        for images in train_dataloader:
            # Move batch of images to device
            images = images.to(device)

            # Forward pass (autoencoder returns reconstructed images)
            x_content, x_context = model(images)

            # Compute reconstruction loss, how does it compare to the actual image
            l1loss = criterion_images(x_content, images)
            ctxloss = criterion_ctx(x_context, images)
            perceptual_loss = criterion_percep(x_content, images)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1).mean()

            # kl_weight = min(beta, beta * global_step / kl_anneal_epoch)

            backprop_loss = l1loss + lambda_percep * perceptual_loss + lambda_ctx * ctxloss

            # Backpropagation
            optimizer.zero_grad()
            backprop_loss.backward()
            optimizer.step()

            running_loss += backprop_loss.item() * images.size(0)
            running_percep += perceptual_loss.item() * images.size(0)
            running_l1 += l1loss.item() * images.size(0)
            running_ctx += ctxloss.item() * images.size(0)

            # running_kl += kl_loss * images.size(0)

            # print(f"KL weight: {kl_weight}")
            # print(f"Effective KL: {(kl_weight * kl_loss).item()}")

        combinedepoch_loss = running_loss / len(train_dataloader.dataset)
        percep_lossavg = running_percep / len(train_dataloader.dataset)
        l1_lossavg = running_l1 / len(train_dataloader.dataset)
        ctx_lossavg = running_ctx / len(train_dataloader.dataset)
        epoch_losses.append(combinedepoch_loss)
        print(f"[Epoch {epoch + 1}]  AE combined Loss: {combinedepoch_loss:.4f} l1 (content) Loss: {l1_lossavg:.4f} mse (context) Loss: {ctx_lossavg:.4f} Perceptual Loss: {percep_lossavg:.4f}")



        return epoch_losses
