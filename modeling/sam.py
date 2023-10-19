# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


# by LBK EDIT
from torchvision.ops.boxes import batched_nms
from utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    uncrop_masks,
)


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
    

    # Batch Individual Mask Generation by LBK
    @torch.no_grad()
    def individual_forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        is_low_resol: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        
        input_images = torch.stack([self.lbk_preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        refined_mask_outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
          if "point_coords" in image_record:
              points = (image_record["point_coords"], image_record["point_labels"])
          else:
              points = None
          sparse_embeddings, dense_embeddings = self.prompt_encoder(
              points=points,
              boxes=image_record.get("boxes", None),
              masks=image_record.get("mask_inputs", None),
          )
          low_res_masks, iou_predictions = self.mask_decoder(
              image_embeddings=curr_embedding.unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )

          # Progressing Intergraion.. by LBK
          refined_masks = self.postprocess_small_regions(low_res_masks, iou_predictions, *input_images.shape[2:], is_low_resol)
          if not is_low_resol:
            refined_masks = F.interpolate(
              refined_masks.unsqueeze(1).float(),
              input_images.shape[2:],
              mode="bilinear",
              align_corners=False,
            ).squeeze(1).bool()
          refined_mask_outputs.append(refined_masks)
          
        return refined_mask_outputs
    
    # PostProcess by LBK EDIT
    def postprocess_small_regions(self, masks, iou_predictions, orig_h, orig_w, is_low_resol):


      """
      Configuration
      """
      pred_iou_thresh = 0.7 #0.88
      stability_score_offset = 1.0
      stability_score_thresh = 0.7 #0.95
      box_nms_thresh = 0.7


      # Interpolation
      if not is_low_resol:
        masks = F.interpolate(
            masks,
            (orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
      else:
          orig_h, orig_w = masks.shape[2:]

      # Serialize predictions and store in MaskData
      data = MaskData(
          masks=masks.flatten(0, 1),
          iou_preds=iou_predictions.flatten(0, 1),          
      )

      # Filter by predicted IoU
      if pred_iou_thresh > 0.0:
          keep_mask = data["iou_preds"] > pred_iou_thresh
          data.filter(keep_mask)

      # Calculate stability score
      data["stability_score"] = calculate_stability_score(
          data["masks"], self.mask_threshold, stability_score_offset
      )
      if stability_score_thresh > 0.0:
          keep_mask = data["stability_score"] >= stability_score_thresh
          data.filter(keep_mask)

      # Threshold masks and calculate boxes
      data["masks"] = data["masks"] > self.mask_threshold
      data["boxes"] = batched_mask_to_box(data["masks"])

      # Filter boxes that touch crop boundaries
      keep_mask = ~is_box_near_crop_edge(data["boxes"], [0, 0, orig_w, orig_h], [0, 0, orig_w, orig_h])
      if not torch.all(keep_mask):
          data.filter(keep_mask)
      data['masks'] = uncrop_masks(data["masks"], [0, 0, orig_w, orig_h], orig_h, orig_w)

      # Remove duplicates within this crop.
      keep_by_nms = batched_nms(
          data["boxes"].float(),
          data["iou_preds"],
          torch.zeros_like(data["boxes"][:, 0]),  # categories
          iou_threshold=box_nms_thresh,
      )
      data.filter(keep_by_nms)

      # making masks
      return data['masks']

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    # by lbk edit
    def lbk_preprocess(self, x: torch.Tensor) -> torch.Tensor:
      """Normalize pixel values and pad to a square input."""
      # Normalize colors
      x = (x - self.pixel_mean) / self.pixel_std
      return x
