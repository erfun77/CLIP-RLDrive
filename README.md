# CLIP-RLDrive: Enhancing Autonomous Vehicle Decision-Making with Vision-Language Models  

## Overview  
CLIP-RLDrive is a reinforcement learning (RL)-based framework designed to improve the decision-making of autonomous vehicles (AVs) in complex urban driving scenarios, particularly at **unsignalized intersections**. By integrating **Contrastive Language-Image Pretraining (CLIP)** into the reward shaping process, our approach aligns AV decisions with **human-like driving preferences**.  

## Key Features  
- ✅ **CLIP-Based Reward Shaping**: Uses Vision-Language Models (VLMs) to extract visual and textual cues for guiding AV decisions.  
- ✅ **Human-Like Driving Behavior**: Leverages CLIP’s image-text alignment to translate natural language instructions into RL rewards.  
- ✅ **Comparison of RL Algorithms**: Implements **Proximal Policy Optimization (PPO)** and **Deep Q-Network (DQN)** to evaluate the effectiveness of CLIP-based reward modeling.  

## Methodology  
- Traditional RL **struggles with designing reward models** for complex driving scenarios.  
- We incorporate **CLIP** to **automatically generate reward signals** based on visual and textual inputs.  
- The framework trains AVs using **PPO and DQN** in a simulated unsignalized intersection environment.  


#### Citation

Our paper has been pre-printed! If you find our work helpful, please consider citing us using the following reference:

```bibtex
@article{doroudian2024clip,
  title={CLIP-RLDrive: Human-Aligned Autonomous Driving via CLIP-Based Reward Shaping in Reinforcement Learning},
  author={Doroudian, Erfan and Taghavifar, Hamid},
  journal={arXiv preprint arXiv:2412.16201},
  year={2024}
}
