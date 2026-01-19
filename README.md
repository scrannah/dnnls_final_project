# DNNLS Final Assessment   
Author: Hannah Timms 
 
--- 
 
## Introduction and Problem Statement 
 
This repository contains the final assessment for the Deep Neural Networks and Learning Systems (DNNLS) course. The goal of this project is to extend a provided baseline storytelling model by implementing cross modal fusion and image autoencoder pretraining and evaluating their impact on reasoning quality and narrative coherence. 
 
The task focuses on story reasoning, where a model must understand causal and temporal relationships between events in short stories and generate consistent narrative outcomes. 
 
We use the StoryReasoning dataset for training and evaluation: 
 
- Dataset: https://huggingface.co/datasets/daniel3303/StoryReasoning 
 
## Problem Definition 
 

Given a sequence of K image–text pairs, the model must predict the (K+1)th image and corresponding textual description. 

This task requires the system to: 

Encode visual and textual information into a shared latent space 

Model temporal dependencies across sequences 

Generate multimodal outputs that are both visually plausible and narratively consistent 

The provided baseline architecture performs multimodal fusion via simple latent concatenation. This approach does not consider how useful each unimodal latent is to the fused representation. 

 
 
--- 
 
## Methods 
 
### Model Architecture Overview 
 
Starting from the baseline encoder–decoder architecture provided, we introduce a cross-modal fusion gate to explicitly control information flow between visual and textual representations. 

To support effective multimodal reasoning, we also strengthen the unimodal encoders by increasing their latent capacity and retraining them with task-specific objectives. 

The final architecture consists of: 

A ResNet18-style visual encoder with perceptual and gradient losses, and a decoder with additional layers 

A retrained text autoencoder with increased latent dimensionality (128) to match the visual autoencoder 

A learned cross-modal fusion gate replacing simple latent concatenation within the sequence model 

A high-level diagram of the modified architecture can be found here: 

docs/architecture_diagram.png 

 
 
## Results 
 
### Quantitative Evaluation 
 
| Metric                 | Baseline Average | Cross-Modal Gate | Cross-Modal Gate + Retrained Modalities |
|------------------------|------------------|------------------|-----------------------------------------|
| Image MSE              | 0.077745         | 0.07724          | 0.083735                                |
| Perplexity             | 65.6105          | 54.5915          | 36.939                                  |
| SSIM                   | 0.154985         | 0.16306          | 0.246495                                |
| BLEU                   | 0.013515         | 0.027            | 0.065295                                |
| Cross-Modal Similarity | -0.184515        | 0.23133          | 0.111565                                |

Cross-modal similarity improves substantially compared to the baseline when introducing a cross-modal fusion gate. However, when retrained modalities are added, cross-modal similarity decreases while unimodal generation metrics improve. This suggests a trade-off between representation alignment and generation quality, which is widely observed in multimodal learning systems, such as CLIP (Radford et al., 2021).

- Loss curves: `results/loss_curves.png` 
 
### Qualitative Analysis 
 
Example generated stories and comparisons are shown in: 
 
- `results/sample_generations.png` 
 
--- 
 
## Conclusions 

The proposed cross-modal fusion gate improves cross-modal similarity, suggesting more consistent alignment between visual and textual representations compared to the baseline mode. This indicates that controlling information flow during multimodal fusion is beneficial for representation alignment. However, improvements in visual reconstruction quality were limited despite architectural changes to the visual autoencoder. This suggests that stronger latent representations alone are insufficient for high-fidelity reconstruction, potentially due to the absence of skip connections in the decoder and the compression imposed by the latent bottleneck. The text autoencoder benefited from retraining, as reflected by reduced perplexity during training. However, when integrated into sequence prediction, the model exhibited signs of overfitting, limiting generalisation to unseen story sequences. These results together show that improved multimodal alignment does not translate into improved generation quality. Alignment and generation should be considered independent objectives to effectively predict the next step in the sequence. 

---

## Future Work 
-	 
- Full disentanglement of content and context heads with a visual autoencoder. 
- Using additional tags to extract information from the dataset. 
- Utilising skip connections to aid visual reconstructions in the decoder. 
- Multimodal contrastive learning. 
 
