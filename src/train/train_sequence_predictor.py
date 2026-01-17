from src.utils.training_utils import validation, show_image
import torch.nn.functional as F


def train_sequence_predictor(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion_images,
        criterion_ctx,
        criterion_text,
        tokenizer,
        device,
        num_epochs,
        lambda_cm
):
    model.train()
    epoch_losses = []
    train_mse_values = []
    train_perplexity_values = []
    train_bleu_values = []
    train_crossmodal_values = []
    train_ssim_values = []

    val_mse_values = []
    val_perplexity_values = []
    val_bleu_values = []
    val_crossmodal_values = []
    val_ssim_values = []
    for epoch in range(num_epochs):

        running_loss = 0.0
        for frames, descriptions, image_target, text_target in train_dataloader:
            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            frames = frames.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)
            # Predictions from our model
            pred_image_content, pred_image_context, predicted_text_logits_k, _, _, z_t_flat, z_v_flat = model(frames,
                                                                                                              descriptions,
                                                                                                              text_target)
            # Computing losses
            # Loss for image reconstruction
            loss_im = criterion_images(pred_image_content, image_target)  # image loss
            # Loss for the average pattern the images contain
            mu_global = frames.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global)  # context loss
            # Loss function for the text prediction
            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:]  # Slice to get [8, 119]
            target_flat = target_labels.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)
            loss_align = 1 - F.cosine_similarity(z_v_flat, z_t_flat, dim=1).mean()
            # Combining the losses
            loss = loss_im + loss_text + 0.2 * loss_context + lambda_cm * loss_align
            # added a cm alignment loss to push them together

            # Optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        model.eval()
        print("Validation on training dataset")
        print("----------------")
        mse_values, perplexity_values, bleu_values, crossmodal_values, ssim_values = validation(model, train_dataloader,
                                                                                               device, tokenizer,
                                                                                               criterion_text,
                                                                                               criterion_ctx)
        train_mse_values.append(mse_values)
        train_perplexity_values.append(perplexity_values)
        train_bleu_values.append(bleu_values)
        train_crossmodal_values.append(crossmodal_values)
        train_ssim_values.append(ssim_values)
        print("Validation on validation dataset")
        print("----------------")
        val_mse, val_perp, val_bleu, val_crossmodal, val_ssim = validation(model, val_dataloader, device, tokenizer,
                                                                          criterion_text, criterion_ctx)
        val_mse_values.append(val_mse)
        val_perplexity_values.append(val_perp)
        val_bleu_values.append(val_bleu)
        val_crossmodal_values.append(val_crossmodal)
        val_ssim_values.append(val_ssim)
        model.train()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch + 1}] AE Loss: {epoch_loss:.4f}")

    return {
        "epoch_losses": epoch_losses,

        "train_mse": train_mse_values,
        "train_perplexity": train_perplexity_values,
        "train_bleu": train_bleu_values,
        "train_crossmodal": train_crossmodal_values,
        "train_ssim": train_ssim_values,

        "val_mse": val_mse_values,
        "val_perplexity": val_perplexity_values,
        "val_bleu": val_bleu_values,
        "val_crossmodal": val_crossmodal_values,
        "val_ssim": val_ssim_values
    }
