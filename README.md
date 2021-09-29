## Skin Tone Manipulation

A PyTorch implementation of Skin Tone Manipulation.

This repo takes 2 steps to achieve the goal of adjusting skin color.

1.Use a semantic segmentation model to extract skin pixels.

Due to few images with skin annotations, train the model with the method of transfer learning.

2.Adopt a skin color matching algorithm to process the skin pixels in the image.

## Requirements

- **Windows** or **Linux**, with **Python ≥ 3.6**

- **[PyTorch](https://pytorch.org/)**

- **matplotlib** - visualizing images and results

## Test

If you can use Jupyter, run ```eval.ipynb```.

Or

```python eval.py --image-path ./skin_tone_val/001.jpg --ckpt-path weights.pth```

## Reference

Skin Segmentation: https://github.com/WillBrennan/SemanticSegmentation

Skin Color Matching Algorithm: https://xie.infoq.cn/article/2bd6ac8b2e2c23a27ae85c316

