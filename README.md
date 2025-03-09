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

## Results  
| Algorithm   | Success Rate | Collision Rate | Timeout Rate |  
|------------|-------------|---------------|--------------|  
| **CLIP-DQN** | **96%** | **4%** | **-** |  
| **CLIP-PPO** | **38%** | **-** | **54%** |  

### Key Findings  
- **CLIP-DQN significantly outperforms CLIP-PPO**, achieving a **96% success rate** with only a **4% collision rate**.  
- The results demonstrate that **CLIP-based reward shaping enhances RL training efficiency**, leading to safer and more human-like AV behavior.  

## Installation & Usage  
### Prerequisites  
Ensure you have the following dependencies installed:  
```bash
pip install torch torchvision transformers stable-baselines3 highway-env



@article{doroudian2024cliprldrive,
  title={CLIP-RLDrive: Integrating Contrastive Language-Image Pretraining for Autonomous Vehicle Decision-Making at Unsignalized Intersections},
  author={Doroudian, Erfan and others},
  journal={IEEE ITSC 2024},
  year={2024}
}
