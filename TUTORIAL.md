# STRIDE Tutorial: Evolution and Technical Deep Dive

This document outlines the transition from the conceptual STRIDE framework to its current implementation and provides a detailed breakdown of its core synthesis and optimization mechanisms.

## Version Comparison Summary

| Feature | Original Concept (`context.md`) | Current Implementation (`README.md`) |
| :--- | :--- | :--- |
| **Editing Objective** | **Corrective Direction:** Minimizes MSE between edited actions and a weighted sum of influential neighbor actions ($\Delta a^{\text{target}}$). | **Direct Preference Optimization (DPO):** Uses max/min influence neighbors as winner/loser pairs to train the editor via preference-based log-sigmoid loss. |
| **Influence Calculation**| **Theoretical:** Hessian-based influence $I = -\nabla J^T H^{-1} \nabla \ell$. | **Practical (TRAK):** Dot product of random gradient projections $P \in \mathbb{R}^{d \times p}$ to approximate influence. |
| **Refinement Strategy** | **Residual Editing:** Directly predicts $\delta z$ to compute edited action $a'$. | **Two-Stage Synthesis:** Combines convex blending $(1-\alpha)a_{\text{orig}} + \alpha a'$ with latent-space noise augmentation ($n_{\text{aug}}$). |
| **Loss Function** | Qualitative description of alignment with $\Delta a^{\text{target}}$. | Quantitative objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \lambda_{\text{cos}}\text{Align} + \lambda_{\text{reg}}\|\delta z\|^2$. |
| **Empirical Validation** | Proposed baselines for evaluation. | Concrete result table for `AdroitHandPen-v1` showing 70% success rate (15% improvement over BC). |
| **Completeness** | Task description and conceptual formulation. | Added project structure, explicit mathematical hyperparameters ($\beta, \alpha, \lambda$), and Quick Start CLI instructions. |

---

## Technical Deep Dive

### 1. Two-Stage Synthesis Pipeline

The current implementation uses a two-stage process to transform the original dataset into a "high-utility" dataset. This goes beyond simple editing by combining conservative updates with distribution expansion.

*   **Stage 1: Convex Blending (Conservative Refinement)**
    Instead of replacing the original action $a_{\text{orig}}$ entirely with the editor's output $a'$, we use a blending factor $\alpha$:
    $$a_{\text{corrected}} = (1 - \alpha) a_{\text{orig}} + \alpha a'$$
    This ensures that even if the editor suggests a significant change, the trajectory remains anchored to the original human demonstration. This is critical for maintaining "reachability" and physical plausibility in complex environments like Adroit.

*   **Stage 2: Latent-Space Augmentation (Support Expansion)**
    Once we have the corrected actions, we further enrich the dataset by sampling around the new points in the VAE's latent space:
    $$\mathcal{D}' = \mathcal{D}_{\text{corrected}} \cup \{ (s, D_\phi(\mu + \epsilon, s)) \}_{n_{aug}}$$
    By adding controlled noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ to the latent means $\mu$ and decoding them, we create multiple variations of the "high-influence" behavior. This helps the downstream BC policy generalize better to small perturbations around the optimized trajectory.

### 2. The Multi-Objective Loss Function

The Editor $g_\psi$ is trained using a composite loss function that balances preference learning, geometric alignment, and regularization.

$$ \mathcal{L}_{total} = \mathcal{L}_{DPO} + \lambda_{cos} \mathcal{L}_{align} + \lambda_{reg} \mathcal{L}_{reg} $$

*   **DPO Preference Loss ($\mathcal{L}_{DPO}$):**
    This uses the Direct Preference Optimization framework. For every sample, we identify a "winner" (the neighbor with the highest positive influence) and a "loser" (the neighbor with the lowest/most-negative influence). The loss encourages the model to increase the relative log-probability of the edited action being closer to the winner than the loser:
    $$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \left( \|a' - a_{loser}\|^2 - \|a' - a_{winner}\|^2 \right) \right)$$
    The parameter $\beta$ acts as a temperature, controlling how strictly the model adheres to these preferences.

*   **Directional Alignment ($\lambda_{cos}$):**
    While DPO looks at endpoints, this term ensures the *delta* vector (how we changed the action) points in the right direction. It minimizes the cosine distance between the edit vector $(a' - a_{\text{orig}})$ and the target correction vector $\Delta a_{\text{target}}$ derived from the influence gradient.

*   **Latent Regularization ($\lambda_{reg}$):**
    We penalize the magnitude of the latent residual $\|\delta z\|^2$. This keeps the editor's modifications within the well-modeled regions of the VAE's latent space, preventing "hallucinated" actions that the decoder cannot realistically map back to the environment.
