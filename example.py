# %%
""" Example 1: All mask generation """
# basic module
import numpy as np
from PIL import Image

from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_masks


from build_sam import sam_model_registry

sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth').cuda()
auto_to_mask = SamMaskGenerator(sam, stability_score_thresh=0.8)

# image upload
img = np.array(Image.open("figure/paris2.jpg"))
masks = auto_to_mask.generate(img)

# visualization
plt.figure(figsize=(20,20))
plt.imshow(img)
img = show_masks(masks, plt)
plt.axis('off')
plt.show()
""" End """
# %%
""" Example 2: Prompts -> one mask generation for one image """
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from build_sam import sam_model_registry
from predictor import SamPredictor
from utils.utils import show_mask, show_box, show_points
sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth').cuda()
prompt_to_mask = SamPredictor(sam)


# image upload
img = np.array(Image.open("figure/paris2.jpg"))
prompt_to_mask.set_image(img)


# prompt
# input_point = np.array([[500, 375], [370, 1200]])
# input_label = np.array([1, 1])
input_point = np.array([[370, 1200]])
input_label = np.array([1])

input_box = None #np.array([200, 600, 500, 1400])

# visualization
plt.figure(figsize=(10,10))
plt.imshow(img)
# show_box(input_box, plt.gca(), plt)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

# sam prediction
masks, scores, logits = prompt_to_mask.predict(
    point_coords=input_point,
    point_labels=input_label,
    box = input_box,
    multimask_output=True,
)


# image proposal
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
""" End """

# %%
""" Example 3: Individual prompt -> Multi-mask generation for one image """
import numpy as np
from PIL import Image

from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_mask, show_masks, show_points

from build_sam import sam_model_registry

sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth').cuda()
individual_prompt_to_mask = SamMaskGenerator(model=sam, stability_score_thresh=0.8)

# prompt
input_point = np.array([[500, 375], [600, 600], [370, 1200], [800, 1000]])
input_label = np.array([1, 1, 1, 1])
# input_point = np.array([[370, 1200]])
# input_label = np.array([1])

# image upload
img = np.array(Image.open("figure/paris2.jpg"))
masks = individual_prompt_to_mask.individual_generate(img, input_point)


# image mask generation visualization
for i, mask in enumerate(masks):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask['segmentation'], plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}", fontsize=18)
    plt.axis('on')
    plt.show()
""" End """


# %%
""" [Original SAM] Example 4: Batched Inputs -> Some Prompts -> Multiple Mask Generation """
import numpy as np
from PIL import Image
import torch
import torchvision

from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_mask, show_masks, show_points

from build_sam import sam_model_registry

sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth').cuda()

# prompt
input_point = torch.tensor([[200, 900], [150, 150], [100, 450], [600, 300], [370, 640], [800, 800]]).cuda()
input_label = torch.tensor([1, 1, 1, 1, 1, 1]).cuda()

def prepare_image(image, img_resolution=1024):
    trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(1024),
                                            torchvision.transforms.Resize(img_resolution)])
    image = torch.as_tensor(image).cuda()
    return trans(image.permute(2, 0, 1))

# image upload
img1 = np.array(Image.open("figure/paris.jpg"))
img2 = np.array(Image.open("figure/paris2.jpg"))
img1_tensor = prepare_image(img1)
img2_tensor = prepare_image(img2)
plt.figure(figsize=(5,5))
plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()


batched_input = [
     {
         'image': img1_tensor,
         'point_coords': input_point,
         'point_labels': input_label,
         'original_size': img1_tensor.shape[1:]
     },
     {
         'image': img2_tensor,
         'point_coords': input_point,
         'point_labels': input_label,
         'original_size': img2_tensor.shape[1:]
     }
]

batched_outputs = sam(batched_input, multimask_output=False)

# image mask generation visualization
for i, mask in enumerate(batched_outputs[0]['masks']):
    plt.figure(figsize=(5,5))
    plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
    show_mask(mask.cpu().numpy(), plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
    plt.title(f"Original SAM Mask {i+1}", fontsize=18)
    plt.axis('on')
    plt.show()


for i, mask in enumerate(batched_outputs[1]['masks']):
    plt.figure(figsize=(5,5))
    plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
    show_mask(mask.cpu().numpy(), plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
    plt.title(f"Original SAM Mask {i+1}", fontsize=18)
    plt.axis('on')
    plt.show()


""" End """

# %%
""" Example 5: [LBK SAM] Batched Inputs -> **Some Prompts** -> Multiple Mask Generation with filtering small and dulicated regions or holes [Very Hard] """
import numpy as np
from PIL import Image
import torch
import torchvision

from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_mask, show_masks, show_points

from build_sam import sam_model_registry

sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth').cuda()

# prompt
input_point = torch.tensor([[200, 900], [150, 150], [100, 450], [600, 300], [370, 640], [800, 800]]).cuda()
input_label = torch.tensor([1, 1, 1, 1, 1, 1]).cuda()

def prepare_image(image, img_resolution=1024):
    trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(1024),
                                            torchvision.transforms.Resize(img_resolution)])
    image = torch.as_tensor(image).cuda()
    return trans(image.permute(2, 0, 1))

# image upload
img1 = np.array(Image.open("figure/paris.jpg"))
img2 = np.array(Image.open("figure/paris2.jpg"))
img1_tensor = prepare_image(img1)
img2_tensor = prepare_image(img2)
plt.figure(figsize=(5,5))
plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()


batched_input = [
     {
         'image': img1_tensor,
         'point_coords': input_point,
         'point_labels': input_label,
         'original_size': img1_tensor.shape[1:]
     },
     {
         'image': img2_tensor,
         'point_coords': input_point,
         'point_labels': input_label,
         'original_size': img2_tensor.shape[1:]
     }
]


# LBK propagation
refined_masks = sam.individual_forward(batched_input, multimask_output=True)

# image mask generation visualization
for i, mask in enumerate(refined_masks[0]):
    plt.figure(figsize=(5,5))
    plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
    show_mask(mask.cpu().numpy(), plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
    plt.title(f"LBK Refined Mask {i+1}", fontsize=18)
    plt.axis('on')
    plt.show()

for i, mask in enumerate(refined_masks[1]):
    plt.figure(figsize=(5,5))
    plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
    show_mask(mask.cpu().numpy(), plt.gca())
    # show_box(input_box, plt.gca(), plt)
    show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
    plt.title(f"LBK Refined Mask {i+1}", fontsize=18)
    plt.axis('on')
    plt.show()

""" End """

# %%
""" Example 6: [LBK SAM] Batched Inputs -> **Full Grid Prompts** -> Multiple Mask Generation with filtering small and dulicated regions or holes [Very Hard] """
import numpy as np
from PIL import Image
import torch
import torchvision

from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_mask, show_points, show_lbk_masks

from build_sam import sam_model_registry


# img resolution
img_resolution = 1024
sam = sam_model_registry['vit_b'](checkpoint='ckpt/sam_vit_b_01ec64.pth', custom_img_size=img_resolution).cuda()

# prompt
from utils.amg import build_all_layer_point_grids
input_point = torch.as_tensor(build_all_layer_point_grids(16, 0, 1)[0] * img_resolution, dtype=torch.int64).cuda()
input_label = torch.tensor([1 for _ in range(input_point.shape[0])]).cuda()

def prepare_image(image, img_resolution=img_resolution):
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((img_resolution, img_resolution))])
    image = torch.as_tensor(image).cuda()
    return trans(image.permute(2, 0, 1))

# image upload
img1 = np.array(Image.open("figure/sam1.png"))[...,:3]
img2 = np.array(Image.open("figure/sam2.png"))[...,:3]
img3 = np.array(Image.open("figure/sam3.png"))[...,:3]
img4 = np.array(Image.open("figure/sam4.png"))[...,:3]
img1_tensor = prepare_image(img1)
img2_tensor = prepare_image(img2)
img3_tensor = prepare_image(img3)
img4_tensor = prepare_image(img4)
plt.figure(figsize=(5,5))
plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(img3_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(img4_tensor.permute(1,2,0).cpu().numpy())
plt.axis('on')
plt.show()

# batchify
batched_input = [
     {
         'image': x,
         'point_coords': input_point,
         'point_labels': input_label,
         'original_size': x.shape[1:]
     } for x in [img1_tensor, img2_tensor, img3_tensor, img4_tensor]
]

# LBK propagation
refined_masks = sam.individual_forward(batched_input, multimask_output=True)

# image mask generation visualization
plt.figure(figsize=(5,5))
plt.imshow(img1_tensor.permute(1,2,0).cpu().numpy())
show_lbk_masks(refined_masks[0].cpu().numpy(), plt)
show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
plt.title(f"[Full Grid] LBK Refined Mask", fontsize=18)
plt.axis('on')
plt.show()


plt.figure(figsize=(5,5))
plt.imshow(img2_tensor.permute(1,2,0).cpu().numpy())
show_lbk_masks(refined_masks[1].cpu().numpy(), plt)
show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
plt.title(f"[Full Grid] LBK Refined Mask", fontsize=18)
plt.axis('on')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(img3_tensor.permute(1,2,0).cpu().numpy())
show_lbk_masks(refined_masks[2].cpu().numpy(), plt)
show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
plt.title(f"[Full Grid] LBK Refined Mask", fontsize=18)
plt.axis('on')
plt.show()


plt.figure(figsize=(5,5))
plt.imshow(img4_tensor.permute(1,2,0).cpu().numpy())
show_lbk_masks(refined_masks[3].cpu().numpy(), plt)
show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
plt.title(f"[Full Grid] LBK Refined Mask", fontsize=18)
plt.axis('on')
plt.show()
# %%
