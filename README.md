# 2D-Human-Pose-Estimators-Strenghts-and-Caveats
Perform a deeper analysis of your model on the validation set of the COCO Keypoint Detection Task 2017.

To perform this deeper analysis the COCO validation set corrected the occlusion labelling and added new labels. The new labels added are:
- Occlusion types (per keypoint):
  - Self occlusion
  - Other person occlusion
  - Environment occlusion
- Truncation by the image border (per person)
- Person resolution (per person)
- Wrong annotations

Note: the example images contain the original labels.

**Occlusion**

The occlusion labels are corrected, because many occlusion labels are inaccurate:
![image](https://user-images.githubusercontent.com/63635825/174550627-655c1f68-94ff-4082-8ecc-f2267b36bfba.png)

**Occlusion types**

There are different ways keypoints can be occluded. We separated these ways by three categories: self occlusion, other person occlusion and environment occlusion which are respectively shown below. The COCO 

**Self occlusion**

The COCO dataset only counts other person occlusion and environment occlusion as occlusion, but self occlusion can also lead to worse performance. Hence, self occlusion is also added.
![image](https://user-images.githubusercontent.com/63635825/174561185-12ed0ce6-51e4-4aa7-99c1-a286bcf2954f.png)


**Other person occlusion**

Other person occlusion are keypoints occluded by other people.
![image](https://user-images.githubusercontent.com/63635825/174561216-06fe3630-ec6a-4169-acd5-a41e6497f833.png)


**Environment occlusion**

This type of occlusion is any type of occlusion which is not self- or other person-occlusion. Hence, any object, helmet, blanket, etc.
![image](https://user-images.githubusercontent.com/63635825/174561247-0d6bd9e7-4671-4105-bd4e-c93f30c36c6c.png)


**Truncation by the image border**

Truncation by the image border happens when the image is cutoff in such a way such that only a subset of the person is present in the image.
![image](https://user-images.githubusercontent.com/63635825/174556198-6838f68e-d00c-4dc4-8432-adb1065470bb.png)

**Person resolution examples**

These are calculated by taking the area of a bounding box generated by the minimum and maximum coordinates of the keypoints.
![image](https://user-images.githubusercontent.com/63635825/174557078-ce219797-2ff4-4cae-85c2-903dc7925292.png)

**Wrong annotations**

The COCO validation set contains a lot of wrong annotations. From the validation set 107 people are removed to get a more accurate performance measure. These mistakes consist of wrong keypoint mappings, labelling of objects and wrong people mapping. The latter contains the right keypoints, but leads to problems during the evaluation phase. Since, the Euclidian distance is used during the evaluation between the ground truth keypoints and the predicted keypoints on the segmented person.
![image](https://user-images.githubusercontent.com/63635825/174558124-f4b864c1-a65f-4be0-8b29-6445c80e2085.png)
