## EcommerceLMMQwen2.5

基于电商数据微调的Qwen2.5系列的电商大模型，电商数据sft后电商大模型。是https://github.com/leeguandong/EcommerceLLMQwen2.5的升级版本。人设注入非常好，但是效果确实和之前的模型有所不同。

 <p align="center">
      <a href='https://github.com/leeguandong/EcommerceLLMQwen2.5'>
            <img src='https://img.shields.io/badge/Project-Page-Green'>
      </a>
      <a href='https://github.com/leeguandong/EcommerceLLMQwen2.5'>
            <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
      </a>
      </br>
      <a href="https://github.com/leeguandong/EcommerceLLMQwen2.5/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/leeguandong/EcommerceLLMQwen2.5" />
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLMQwen2.5/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/leeguandong/EcommerceLLMQwen2.5?color=0088ff" />
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLMQwen2.5/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/leeguandong/EcommerceLLMQwen2.5?color=0088ff" />
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLMQwen2.5/stargazers">
        <img src="https://img.shields.io/github/stars/leeguandong/EcommerceLLMQwen2.5?color=ccf">
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLMQwen2.5">
        <img src="https://img.shields.io/github/repo-size/leeguandong/EcommerceLLMQwen2.5.svg?style=flat-square">
      </a>
      </br>
      <a href=href="https://github.com/leeguandong/EcommerceLLMQwen2.5">
        <img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/leeguandong/EcommerceLLMQwen2.5">
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLMQwen2.5">
        <img src="https://img.shields.io/github/last-commit/leeguandong/EcommerceLLMQwen2.5">
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLMQwen2.5/blob/main/LICENSE">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" />
      </a>
  </p>

## 本文贡献

- 借助电商领域的开源数据集，包括alpaca的通用数据，直播带货文案，商品主题词抽取，场景问答，选择题，客服问答，标题扩展，商品介绍seo，写文章，短标题抽取，小红书文案，根据参数扩写，文章总结，人设，tiktok数据。基于Qwen2.5训练微调，2.5版本是之前EcommerceLLM的qwen1.5的升级版本，由于使用了升级之后的llamafactory 0.9.0版本训练，并且训练的方式产生了较大的变化，因此新开了一个repo来记录，权重coming soon。
## 数据集

- 数据集放开了很小一部分的开源tiktok收集的数据，我们的电商数据非常多样化，包括常见的电商领域的任务。
- 训练借助llama-factory 0.9.0，数据格式类似
```bash
  "alpaca_gpt4_data_zh": {
    "file_name": "alpaca_gpt4_data_zh.json",
    "file_sha1": ""
  },
  "tiktok_v1": {
    "file_name": "tiktok_v1.json",
    "file_sha1": ""
  },
```


## 快速上手

### 1.安装环境
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
### 2.模型推理

|模型权重|下载链接|微调方法|
|:-|:-|:-|
|ecommercellm-qwen2.5-7b-instruct|comming soon....|LoRA|


#### CLI推理

```python
yaml文件：
model_name_or_path: "/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2___5-7B-Instruct/"
adapter_name_or_path: /home/lgd/e_commerce_llm/llama_factory_0.9.0/saves/qwen2.5_7b/lora/sft
template: qwen
finetuning_type: lora

export export CUDA_VISIBLE_DEVICES=3,4,5,6
llamafactory-cli chat "/home/lgd/e_commerce_llm/llama_factory_0.9.0/examples/inference/llama3_lora_sft.yaml"
```
### 3.模型训练（复现CommerceLLM）

<details>
  <summary>硬件资源</summary>
  <p>* 实验在A800 (4X, 80GB)上进行</p>
</details>

- （1）准备数据，并且写入到dataset_info.json中；
- （2）开始训练：

```bash
yaml文件
### model
model_name_or_path: "/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2___5-7B-Instruct/"

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset_dir: "/home/lgd/e_commerce_llm/llama_factory_0.9.0/data/"
dataset: alpaca_gpt4_data_zh_retained,sft_gpt4_livestreaming_data_20240105,sft_product_extraction_train,sn_generate_gpt_long_title_part0,sn_generate_gpt_long_title_part1,sn_generate_gpt_long_title_part2,sn_generate_gpt_seo_cp,sn_generate_gpt_seo_other,sn_generate_gpt_seo_phb,sn_generate_gpt_seo_zc,sn_generate_gpt_short_title_part0,sn_generate_gpt_short_title_part1,sn_generate_gpt_train_gds_title,sn_generate_gpt_train_multigds_xhs,sn_generate_gpt_train_part0,sn_generate_gpt_train_part1,sn_generate_qwen72b_gds_tj,who_are_you,tiktok_v1
template: qwen
cutoff_len: 2048
max_samples: 3000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /home/lgd/e_commerce_llm/llama_factory_0.9.0/saves/qwen2.5_7b/lora/sft
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 1000

export export CUDA_VISIBLE_DEVICES=3,4,5,6
llamafactory-cli train "/home/lgd/e_commerce_llm/llama_factory_0.9.0/examples/train_lora/llama3_lora_sft.yaml"
```
这里的复现过程非常简单，主要是很多过程我们都为大家准备好了，大家可以随时复现一个自己的`EcommerceLLM`。

## 效果展示

*以下效果来自于**低质量**的数据训练和权重

影像诊断：
|instruction| qwen1.5-1.8b                                                 | qwen1.5-7b                                                   |qwen2.5-7b|
|:-|:-:|:-:|:-:|
| 你用了开源的数据吗？                                         | 是的，我使用了开源的数据。                                   | 是的，我使用了开源的数据。我使用了由xx开发的xx零售垂域大模型，它基于深度学习技术，使用了大规模的电商数据进行训练。这些数据包括商品信息、用户行为、评论等，用于生成回答。 | 没有，我们的模型训练没有使用开源数据。 |
| 请帮我写一篇小红书风格的电商营销文案                         | 小红书风格的电商营销文案：\n\n亲爱的朋友们，你们是否经常在小红书上看到各种各样的美妆、时尚、美食等推荐？这些推荐的背后，是无数个美妆博主和美食博主们的心血和努力。他们用自己的专业知识和经验，为消费者提供最真实、最实用的购物建议。\n\n今天，我要给大家推荐一款超级好用的电商产品，它将帮助你们在小红书上获得更多的曝光和流量。这款产品就是我们的XXX品牌，它是一款专为小红书用户设计的美妆工具。\n\n首先，这款美妆工具具有强大的数据分析功能，可以实时监测你的美妆和护肤效果，为你提供最精准的建议和指导。无论你是想提升你的妆容效果，还是想了解最新的护肤趋势，这款工具都能满足你的需求。\n\n其次，这款美妆工具还具有强大的社交分享功能。你可以将你的美妆和护肤心得分享到小红书上，让更多的人看到你的推荐和分享。同时，你也可以通过这款工具，与其他美妆博主和美食博主进行交流和互动，获取更多的灵感和建议。\n\n最后，我要强调的是，这款美妆工具的使用非常简单。只需要下载并安装，就可以开始使用。无论你是新手还是老手，都能轻松上手，享受美妆和护肤的乐趣。\n\n希望我的推荐能帮助到你们，让你们在小红书上获得更多的曝光和流量。谢谢大家的关注和支持！ | 标题：小红书风格的电商营销文案，让你的产品瞬间爆红！\n\n正文：大家好，有没有遇到过这样的情况，你想要购买一款产品，但是却不知道该如何选择？或者你想要推广一款产品，但是却不知道如何吸引更多的用户？今天，我就来给大家分享一下如何写出小红书风格的电商营销文案，让你的产品瞬间爆红！\n\n首先，我们要明确一点，小红书风格的电商营销文案是以用户为中心的。我们要从用户的角度出发，了解他们的需求和痛点，然后根据这些信息来撰写文案。例如，如果你的产品是一款面膜，你可以从用户的角度出发，写一篇关于如何选择适合自己的面膜的文章，告诉他们如何根据自己的肤质和需求来选择面膜。\n\n其次，小红书风格的电商营销文案要简洁明了，重点突出。我们要用简洁的语言来描述产品的特点和优势，让用户一眼就能看出产品的价值。例如，如果你的产品是一款口红，你可以写一篇关于如何选择适合自己肤色的口红的文章，告诉用户如何根据自己的肤色来选择口红的颜色。\n\n最后，小红书风格的电商营销文案要有趣味性和互动性。我们要通过一些有趣的内容和互动方式来吸引用户的注意力，让他们对产品产生兴趣。例如，你可以写一篇关于如何化妆的文章，然后在文章中加入一些互动环节，让用户参与进来，分享自己的化妆心得。\n\n总结一下，小红书风格的电商营销文案是以用户为中心的，简洁明了，重点突出，有趣味性和互动性。希望这些技巧能帮助到大家，让你的产品瞬间爆红！ | 【限时特惠】快来抢购！超值优惠，让你的购物体验更上一层楼！😍<br/><br/>🔥限时优惠🔥：现在购买即可享受超值折扣，数量有限，先到先得！快抓住机会，让购物更划算！💰<br/><br/>🎁限时特惠🎁：为了感谢大家的支持，我们特地推出了限时特惠活动，只需在今天下单，即可享受超值优惠！别错过这个绝佳的机会，快来抢购吧！🏃‍♀️🏃‍♂️<br/><br/>🎉限时特惠🎉：为了庆祝新店开业，我们特地推出了限时特惠活动，现在购买即可享受超值折扣！赶紧下单，让你的购物体验更上一层楼！🛍️<br/><br/>🛍️限时特惠🛍️：为了回馈新老顾客，我们特地推出了限时特惠活动，现在购买即可享受超值折扣！别犹豫，快来抢购吧！💨<br/><br/>🛒限时特惠🛒：为了让更多人享受到优质的购物体验，我们特地推出了限时特惠活动，现在购买即可享受超值折扣！快来抢购，让你的购物更愉快！🎉<br/><br/>【限时特惠】快来抢购！超值优惠，让你的购物体验更上一层楼！😍<br/><br/>【限时特惠】快来抢购！超值优惠，让你的购物体验更上一层楼！😍<br/><br/>【限时特惠】快来抢购！超值优惠，让你的购物体验更上一层楼！😍<br/><br/>【限时特惠】快来抢购！超值优惠，让你的购物体验更上一层楼！😍 |
| 请帮我撰写一份该品牌下这类商品如何选择的文章。\n模块必须按这个顺序输出：标题、正文；下面是这两个模块的要求：\n【标题要求】\n1、标题要体现出文章在描述该品类商品应该怎么选的主题；\n2、标题风格要看上去专业，让人信服；\n2、标题字数在30字内；\n【正文要求】\n1、字数在500字左右；\n【品牌】：美素佳儿（Friso）\n【商品类型】：婴幼儿奶粉\n | 标题：如何选择适合宝宝的婴幼儿奶粉品牌？\n\n正文：\n随着宝宝的成长，选择适合宝宝的奶粉品牌变得尤为重要。美素佳儿（Friso）作为婴幼儿奶粉的知名品牌，为家长们提供了多种选择。那么，如何选择适合宝宝的美素佳儿婴幼儿奶粉呢？\n\n首先，我们需要考虑宝宝的年龄和需求。美素佳儿的婴幼儿奶粉分为不同阶段，从0-6个月的婴儿奶粉到12个月以上的幼儿奶粉。根据宝宝的年龄和需求，选择适合的奶粉品牌和型号。\n\n其次，我们需要考虑奶粉的营养成分。美素佳儿的婴幼儿奶粉含有丰富的营养成分，如蛋白质、脂肪、维生素和矿物质等。选择奶粉时，要根据宝宝的年龄和需求，选择适合的营养成分。\n\n再次，我们需要考虑奶粉的口感和口感。美素佳儿的婴幼儿奶粉口感柔和，易于消化，适合宝宝的口感。选择奶粉时，要根据宝宝的口感，选择适合的奶粉。\n\n最后，我们需要考虑奶粉的价格和品牌信誉。美素佳儿的婴幼儿奶粉价格适中，品牌信誉良好。选择奶粉时，要根据自己的经济实力和对奶粉品牌的信任度，选择适合的奶粉。\n\n综上所述，选择适合宝宝的美素佳儿婴幼儿奶粉，需要考虑宝宝的年龄和需求、奶粉的营养成分、口感和价格等因素。希望以上建议能帮助到有需要的家长们。 | 标题】：美素佳儿婴幼儿奶粉选购指南：如何选择适合宝宝的奶粉\n\n【正文】：\n\n美素佳儿（Friso）作为一家知名的婴幼儿奶粉品牌，其产品深受消费者的喜爱。然而，面对众多的美素佳儿婴幼儿奶粉，如何选择适合宝宝的奶粉呢？以下是一些选购指南，希望能帮助到您。\n\n首先，我们需要了解宝宝的年龄和生长阶段。不同阶段的宝宝对营养的需求是不同的，因此选择适合宝宝的奶粉非常重要。一般来说，0-6个月的宝宝可以选择美素佳儿的婴儿配方奶粉，6个月以上的宝宝可以选择美素佳儿的幼儿配方奶粉。\n\n其次，我们需要关注奶粉的配方。美素佳儿的奶粉采用了科学的配方，含有丰富的营养成分，如蛋白质、脂肪、碳水化合物、维生素和矿物质等。在选择奶粉时，我们可以根据宝宝的年龄和生长阶段，选择适合的配方。\n\n再次，我们需要关注奶粉的成分。美素佳儿的奶粉采用了优质的原料，如脱脂乳粉、乳清蛋白、植物油等。这些成分能够提供宝宝所需的营养，同时避免了过多的糖分和脂肪。\n\n此外，我们还需要关注奶粉的口感和质地。美素佳儿的奶粉口感细腻，质地均匀，易于宝宝吞咽。这有助于宝宝更好地吸收奶粉中的营养成分。\n\n最后，我们还需要关注奶粉的价格。美素佳儿的奶粉价格相对较高，但其质量和口感都是有保证的。在选择奶粉时，我们可以根据自己的经济能力，选择适合自己的奶粉。\n\n总的来说，选择美素佳儿婴幼儿奶粉时，我们需要考虑宝宝的年龄和生长阶段，关注奶粉的配方、成分、口感和质地，以及价格等因素。只有这样，我们才能选择到最适合宝宝的奶粉。 | 作为父母，我们都希望给自己的宝宝最好的，尤其是在宝宝的营养补充上。而在众多婴幼儿奶粉品牌中，美素佳儿（Friso）以其高品质和专业性成为了许多家庭的首选。那么，如何选择适合宝宝的美素佳儿婴幼儿奶粉呢？以下是一些选购指南。<br/><br/>首先，要根据宝宝的年龄和阶段来选择奶粉。美素佳儿的婴幼儿奶粉分为不同阶段，分别适用于0-6个月、6-12个月、1-3岁等不同年龄段的宝宝。选择适合自己宝宝年龄阶段的奶粉，可以更好地满足宝宝的营养需求。<br/><br/>其次，要关注奶粉的成分。美素佳儿的婴幼儿奶粉采用了先进的配方，添加了DHA、ARA等营养成分，有助于宝宝大脑和视力的发育。同时，美素佳儿的奶粉还采用了易于消化吸收的配方，让宝宝更易吸收营养。<br/><br/>再次，要选择信誉良好的品牌。美素佳儿作为荷兰皇家菲仕兰旗下的婴幼儿奶粉品牌，以其高品质和专业性赢得了消费者的信赖。选择美素佳儿的婴幼儿奶粉，可以确保宝宝的营养摄入安全可靠。<br/><br/>最后，要根据宝宝的口味和偏好来选择奶粉。美素佳儿的婴幼儿奶粉有多种口味可供选择，包括原味、巧克力味、草莓味等。让宝宝选择自己喜欢的口味，可以提高宝宝喝奶粉的积极性。<br/><br/>总的来说，选择美素佳儿的婴幼儿奶粉，要根据宝宝的年龄、奶粉的成分、品牌信誉和口味偏好来综合考虑。只有这样，才能为宝宝提供最适合的营养补充，让宝宝健康成长。 |
| 在给定的笔记本电脑上撰写一篇评论,联想Yoga 920                | 作为AI语言模型，我无法直接在笔记本电脑上撰写评论。但是，我可以根据您提供的信息，为您提供一些关于笔记本电脑的评论。 | 作为一个人工智能助手，我无法在给定的笔记本电脑上撰写评论，因为我没有实际的体验和感受。然而，我可以为您提供一些撰写笔记本电脑评论的建议。\n\n首先，您需要考虑笔记本电脑的性能和功能。您可以评估其处理器、内存、硬盘容量、显卡和电池续航能力。您还可以考虑其屏幕分辨率、显示效果和触摸屏功能。\n\n其次，您需要考虑笔记本电脑的外观和设计。您可以评估其重量、尺寸、颜色和材质。您还可以考虑其键盘和触摸板的舒适度和响应速度。\n\n最后，您需要考虑笔记本电脑的价格和性价比。您可以评估其价格与性能的平衡，以及与其他同类产品相比的竞争力。\n\n综上所述，撰写一篇笔记本电脑评论需要考虑多个方面，包括性能、功能、外观、设计、价格和性价比。希望这些建议能帮助您撰写一篇有价值的评论。 | 联想Yoga 920是一款出色的笔记本电脑，它拥有出色的性能和令人印象深刻的屏幕。这款设备搭载了第8代Intel Core i7处理器，配备16GB内存和512GB固态硬盘，能够满足大多数用户的日常需求。它还配备了NVIDIA GeForce GTX 1050 Ti独立显卡，能够流畅运行各种图形密集型应用和游戏。<br/><br/>Yoga 920的13.9英寸PixelSense触摸屏采用了InfinityEdge设计，拥有1080p分辨率和16:10的宽高比，能够呈现出清晰细腻的画面。它的键盘采用背光设计，能够在黑暗环境中提供更好的输入体验。<br/><br/>此外，Yoga 920还具备出色的续航能力，电池续航时间可达到12小时以上。它还支持快速充电技术，能够在短时间内为设备充电。<br/><br/>总体而言，联想Yoga 920是一款性能出色、设计精美的笔记本电脑，适合那些需要高效处理任务和享受出色视觉体验的用户。 |


## 项目致谢

1. [LLAMA-Factory](https://github.com/hiyouga/LLaMA-Factory)为我们提供了训练框架；
1. [Qwen2.5-7b-instruct](https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct/summary)在modelscope上下载；

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。

## 使用许可

此存储库遵循[CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) ，请参阅许可条款。

