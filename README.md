# Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis
This repo contains the core codes for the paper "[Deep Reinforcement Learning for Cost-Effective Medical Diagnosis](https://openreview.net/forum?id=0WVNuEnqVu)". Due to the data privacy policy, we are not allowed to share the datasets. For further information on the data format, please see [here](https://github.com/Zheng321/Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis/blob/e7365e563b32e812d5a38c9a907fa682c4972a92/blood_panel_data_preprocessing.py#L6). If you have any questions or concerns, please don't hesitate to get in touch with us.

The end-to-end semi-model-based RL training framework illustrated below contains three core modules: Posterior State Encoder via Imputer (E), Classifier (C), and Panel/Predictor Selector (S). The final state embedding of RL contains the observation 0-1 indicator, embedding output by (E) and (C).
<p align="center">
<img width="668" alt="image" src="https://user-images.githubusercontent.com/41489420/221870344-4b573367-0801-47f3-a644-f537f7d78271.png">
</p>

Here are the core codes and their functionality. Note that the required medical datasets are missing due to privacy policies.

- baselines.ipynb: multiple basic baseline algorithms reported in the paper.
- blood_panel_data_preprocessing.py: example data preprocessing script, needs to be replaced by your own datasets.
- classifier.py: (C) module
- data_loader.py: helper to construct random observation pattern for training (C) module 
- flow_models.py, nflow.py, imputation.py: (E) module, where a flow-based deep imputer named [EMFlow](https://github.com/guipenaufv/EMFlow) is used.
- rl.py: (S) module
- util.py, my_result_writer.py: some helper functions
- train.py: example training script
- meta_train.py: example tuning script

Please contact <zhengy@princeton.edu> or leave an issue if you have any questions.

If you find our implementation helps, please consider citing this:)
```
@article{yu2023deep,
  title={Deep Reinforcement Learning for Cost-Effective Medical Diagnosis},
  author={Yu, Zheng and Li, Yikuan and Kim, Joseph and Huang, Kaixuan and Luo, Yuan and Wang, Mengdi},
  journal={arXiv preprint arXiv:2302.10261},
  year={2023}
}
```
