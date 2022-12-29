from vis_graph import draw_skel
from unipose.datasets import MPIIDataset, COCODataset, AnimalKingdomDataset

dataset = MPIIDataset("/home/dl2022/d3d/unipose/datasets/mpii")
# dataset = COCODataset("/home/dl2022/d3d/unipose/datasets/coco")
# dataset = AnimalKingdomDataset("/home/dl2022/d3d/unipose/datasets/animal_kingdom", sub_category="ak_P3_amphibian")
dataloader = dataset.make_dataloader(batch_size=1, shuffle=True)

# Get one item
for i, d in enumerate(dataloader):
    images = d["images"]
    kp_images = d["keypoint_images"]
    image_arr = d["images"].cpu().detach().numpy() # (1, 3, 256, 256)
    draw_skel(image_arr, kp_images, "output.png")
    break 
