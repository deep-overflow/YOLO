# YOLO : ๐ฏYonsei gOnna Lose fOrever๐ฏ 
- YOLO V1 ๋ชจ๋ธ์ ๊ตฌํํ ๊ฒฐ๊ณผ๋ฌผ ๋ ํฌ์งํ ๋ฆฌ์๋๋ค.
  - Training ๊ณผ์ ์ ๋ํ์ผ์ ์ง์คํ์ต๋๋ค.
  - ๊ณ ์ฐ์  ์ค๋๋นต์ ๊ธฐ์ํฉ๋๋ค.

# ์ค๋ช
- ์  ์์๋ฌผ์ `./loggerJK` ํด๋ ์์ ์์ต๋๋ค.
  - `YOLO_singleLoss.ipynb`
    - ๋ฐฐ์น ๋จ์ ์ฒ๋ฆฌ๋ฅผ ์ง์ํ์ง ์๋ ๋ฒ์ ์ YOLO์๋๋ค.
  - `YOLO_batchLoss.ipynb`
    - ๋ฐฐ์น ๋จ์ ์ฒ๋ฆฌ๋ฅผ ์ง์ํ๋๋ก ๊ฐ์ ํ ๋ฒ์ ์ YOLO์๋๋ค.
  - `YOLO_batchLoss_trainval.ipynb`
    - Training Set๋ง์ผ๋ก๋ ํ์ต์ด ์ด๋ ค์ Training / Validation Set ๋ชจ๋ ํ์ต์ ์ด์ฉํ ๋ธํธ๋ถ์๋๋ค.
- ํ์ตํ ๋ชจ๋ธ์ Inference ๊ฒฐ๊ณผ๋ฌผ์ `./loggerJK/Model Test` ํด๋ ์์ ์์ต๋๋ค.
  - `./loggerJK/Model Test/model_test.ipynb`
  - Inference ๊ณผ์  ์ค mAP ๊ณ์ฐ, Non-Maximum Suppression๊ณผ ๊ฐ์ ๋ถ๋ถ๋ค์ ๊ตฌํ๋์ด ์์ง ์์์ต๋๋ค.

# ๋ชจ๋ธ ์ค๋ช
- Base Model : Vision Transformer
  - `vit_base_patch32_384` from `timm`
- `input_size` : $384 \times 384$
- `learning_rate` : 1e-5 (fixed)
- `epoch` : 70
  - ์ด ์ธ์ ๊ธฐํ Training์ ๊ด๋ จ๋ ์ํ์  ๋ํ์ผ๋ค์ ๋ผ๋ฌธ๊ณผ ๋์ผํ๊ฑฐ๋, ์ต๋ํ ์ ์ฌํ๋๋ก ๊ตฌํํ์ต๋๋ค.


@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}	
