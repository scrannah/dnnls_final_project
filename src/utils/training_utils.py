import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import textwrap
import torchvision.transforms as transforms
import torch.nn.functional as F
from .token_generate import generate
from torchmetrics.image import StructuralSimilarityIndexMeasure
from nltk.translate.bleu_score import sentence_bleu


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


# Plots four images and their reconstructions
def validation(model, data_loader, device, tokenizer, criterion_text, criterion_ctx):
    model.eval()
    with torch.no_grad():
        frames, descriptions, image_target, text_target = next(iter(data_loader)) # this will be the same if val dataloader is NOT shuffled

        descriptions = descriptions.to(device)
        frames = frames.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)

        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # METRICS

        predicted_image_content, context_image, predicted_text_logits_k, hidden, cell, _, _ = model(frames, descriptions,
                                                                                        text_target)  # need all these for validation metrics

        val_image_mse = criterion_ctx(context_image, image_target).item()
        print(f"Validation Image MSE: {val_image_mse:.4f}")

        # flatten and remove teacher forcing

        prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)

        # flatten and remove teacher forcing
        target_tokens = text_target.squeeze(1)[:, 1:]  # remove the dimension (1) and BOS
        target_flat = target_tokens.reshape(-1)  # make it a 1d vector

        # calculating los for text
        val_loss_text = criterion_text(prediction_flat, target_flat)

        val_perplexity = torch.exp(val_loss_text).item()
        print(f"Validation Perplexity: {val_perplexity:.2f}")

        # SSIM

        predicted_img = torch.clamp(predicted_image_content, 0, 1)  # SSIM needs values between 0,1 so we clamp for it here
        target_img = torch.clamp(image_target, 0, 1)
        ssim_val = ssim_fn(predicted_img, target_img).item()
        print(f"Validation SSIM: {ssim_val:.4f}")

        pred_ids = torch.argmax(predicted_text_logits_k, dim=-1)[0]
        pred_ids2 = torch.argmax(text_target, dim=-1)


        # Ground truth IDs
        gt_ids = text_target.squeeze(1)[0]  # get rid of middle dimension, pass first example for visualisation

        # Decode tokens
        pred_sentence = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
        gt_sentence = tokenizer.decode(gt_ids.tolist(), skip_special_tokens=True)

        val_bleu = sentence_bleu([gt_sentence.split()], pred_sentence.split())  # pass ground truth and predicted
        print(f"Validation BLEU: {val_bleu:.4f}")

        img_emb = model.image_encoder(image_target)  # GT instead of predicted cms
        txt_emb = model.text_encoder(pred_ids2)
        img_emb = F.normalize(img_emb, p=2, dim=1) # Normalising embeddings for cosine similarity
        txt_emb = F.normalize(txt_emb, p=2, dim=1)

        # if text encoder returns embeddings for each token. average them
        if isinstance(txt_emb, tuple):  # if text encoder returns tuple not tensor, take 1st value only
            txt_emb = txt_emb[0]
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(dim=1)  # mean pool sequence

        val_cross_modal = F.cosine_similarity(img_emb, txt_emb, dim=1).mean().item()

        # perm = torch.randperm(txt_emb.size(0), device=txt_emb.device)
        # cms_shuffled = F.cosine_similarity(img_emb, txt_emb[perm], dim=1).mean().item()

        # print("CMS shuffled:", cms_shuffled)
        print(f"Validation Cross-modal Similarity: {val_cross_modal:.4f}")

        figure, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={'height_ratios': [2, 1.5]})

        for i in range(4):
            im = frames[0, i, :, :, :].cpu()
            show_image(ax[0, i], im)
            ax[0, i].set_aspect('auto')
            ax[0, i].axis('off')
            wrapped_text = textwrap.fill(tokenizer.decode(descriptions[0, i, :], skip_special_tokens=True), width=40)

            ax[1, i].text(
                0.5, 0.99,
                wrapped_text,
                ha='center',
                va='top',
                fontsize=10,
                wrap=True
            )

            ax[1, i].axis('off')  # Hide axes for the text subplot

        show_image(ax[0, 4], image_target[0].cpu())
        ax[0, 4].set_title('Target')
        ax[0, 4].set_aspect('auto')
        ax[0, 4].axis('off')
        text_target = text_target.squeeze(1)

        wrapped_text = textwrap.fill(tokenizer.decode(text_target[0], skip_special_tokens=True), width=40)
        ax[1, 4].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 4].axis('off')
        output = predicted_image_content[0, :, :, :].cpu()  # this was context image I changed to content
        show_image(ax[0, 5], output)
        ax[0, 5].set_title('Predicted')
        ax[0, 5].set_aspect('auto')
        ax[0, 5].axis('off')

        generated_tokens = generate(model.text_decoder,
                                    hidden[:, 0, :].unsqueeze(1),
                                    cell[:, 0, :].unsqueeze(1),
                                    max_len=150,
                                    sos_token_id=tokenizer.cls_token_id,
                                    eos_token_id=tokenizer.sep_token_id,
                                    device=device)

        wrapped_text = textwrap.fill(tokenizer.decode(generated_tokens), width=40)

        ax[1, 5].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 5].axis('off')
        plt.tight_layout()
        plt.show()
        return (
            val_image_mse,
            val_perplexity,
            val_bleu,
            val_cross_modal,
            ssim_val
        )


def show_image(ax, image, de_normalize=False, img_mean=None, img_std=None):
    """
  De-normalize the image (if necessary) and show image
  """
    if de_normalize:
        new_mean = -img_mean / img_std
        new_std = 1 / img_std

        image = transforms.Normalize(
            mean=new_mean,
            std=new_std
        )(image)
    ax.imshow(image.permute(1, 2, 0))
