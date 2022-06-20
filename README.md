# 2D-Human-Pose-Estimators-Strenghts-and-Caveats
Perform a deeper analysis of your model on the COCO validation set for human keypoint detection

To perform this deeper analysis the COCO validation set corrected the occlusion labelling and added new labels. The new labels added are:
- Occlusion types (per keypoint):
  - Self occlusion
  - Other person occlusion
  - Environment occlusion
- Truncation by the image border (per person)
- Person resolution (per person)

