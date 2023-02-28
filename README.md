# Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis
This repo contains the core codes for the paper "[Deep Reinforcement Learning for Cost-Effective Medical Diagnosis](https://openreview.net/forum?id=0WVNuEnqVu)". Due to the data privacy policy, we are not allowed to share the datasets. Therefore, we only share the core codes of our framework. 

The end-to-end semi-model-based RL training framework illustrated below contains three core modules: Posterior State Encoder via Imputer (E), Classifier (C), and Panel/Predictor Selector (S). The final state embedding of RL contains the observation 0-1 indicator, embedding output by (E) and (C).
<p align="center">
<img width="668" alt="image" src="https://user-images.githubusercontent.com/41489420/221870344-4b573367-0801-47f3-a644-f537f7d78271.png">
</p>

