# 2D-Human-Pose-Estimators-Strenghts-and-Caveats
Perform a deeper analysis of your model on the COCO validation set for human keypoint detection

To perform this deeper analysis the COCO validation set corrected the occlusion labelling and added new labels. The new labels added are:
- Occlusion types (per keypoint):
  - Self occlusion
  - Other person occlusion
  - Environment occlusion
- Truncation by the image border (per person)
- Person resolution (per person)

Occlusion
The occlusion labels are corrected, because many occlusion labels are inaccurate:
![image](https://user-images.githubusercontent.com/63635825/174550627-655c1f68-94ff-4082-8ecc-f2267b36bfba.png)

Occlusion types
There are different ways keypoints can be occluded. We separated these ways by three categories: self occlusion, other person occlusion and environment occlusion which are respectively shown below
