from src.utils.training_utils import validation

def train_sequence_predictor(
        model,
        train_dataloader,
        optimizer,
        criterion_images,
        criterion_ctx,
        criterion_text,
        tokenizer,
        device,
        num_epochs
    ):

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):

        running_loss = 0.0
        for frames, descriptions, image_target, text_target  in train_dataloader:

            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            frames = frames.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)
            # Predictions from our model
            pred_image_content, pred_image_context, predicted_text_logits_k, _, _ = model(frames, descriptions, text_target)
            # Computing losses
            # Loss for image reconstruction
            loss_im = criterion_images(pred_image_content, image_target) # image loss
            # Loss for the average pattern the images contain
            mu_global = frames.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global) # context loss
            # Loss function for the text prediction
            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:] # Slice to get [8, 119]
            target_flat = target_labels.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)
            # Combining the losses
            loss = loss_im + loss_text + 0.2*loss_context
            # Optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        model.eval()
        print("Validation on training dataset")
        print( "----------------")
        validation( model, train_dataloader, device )
        print("Validation on validation dataset")
        print( "----------------")
        validation( model, val_dataloader, device)
        model.train()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)

    return epoch_losses

