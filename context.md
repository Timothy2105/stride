# STRIDE: Strategic Trajectory Refinement via Influence-guided Data Editing

## Task
Your task is to implement STRIDE and apply it to D4RL/pen/human-v2/ dataset.
## STRIDE

We propose STRIDE, a trajectory refinement framework that converts suboptimal demonstrations into high-utility training data for imitation learning. 


## Problem Formulation and Notation

A policy $\pi_\theta(a|s)$ is trained via behavior cloning:

$$
\mathcal{L}_{\text{BC}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^N
\ell(\pi_\theta(s_i), a_i),
$$

typically using mean squared error. BC treats all demonstrations as optimal, causing suboptimal behaviors to propagate into the learned policy. We instead learn an editor

$$g_\psi(s_i,a_i)\rightarrow a_i'$$

that produces an edited dataset $\mathcal{D}'=\{(s_i,a_i')\}$ to improve validation performance.


## Influence Estimation via TRAK

To measure the utility of each sample, we approximate its influence on validation loss.

Let $\theta^{(0)}$ be parameters trained on $\mathcal{D}_{\text{demo}}$.  
The influence of sample $(s_i, a_i)$ is defined as
$$
I(s_i,a_i)
=
- \nabla_\theta J(\theta^{(0)})^\top
H_\theta^{-1}
\nabla_\theta \ell\big(\pi_{\theta^{(0)}}(s_i), a_i\big),
$$
where $J(\theta)$ is validation loss and $H_\theta$ is the Hessian of the training loss.

We approximate this equation using TRAK-style feature projections, avoiding explicit Hessian inversion. These influence scores guide both sample selection and editing direction.


## Latent-Space Residual Editing

Direct action perturbations can produce unrealistic motions; therefore, we operate in a learned latent space using a conditional VAE.

We train a VAE on demonstrations:
$$
\mathcal{L}_{\text{VAE}}
=
\|a_i-\hat a_i\|^2
+
\beta_{\text{KL}}\,\mathrm{KL}(q(z_i|s_i,a_i)\|\mathcal N(0,I))$$.

The decoder predicts a residual in latent space:
$$
z_i' = z_i + \delta z_i,
\qquad
\delta z_i = g_\psi(s_i, a_i, \xi),
$$
where $\xi$ is noise.

The edited action is $a_i' = D_\phi(z_i', s_i)$.


## Influence-Guided Corrective Editing

Latent editing ensures smooth perturbations, but we must determine directions that improve training utility.

For each latent embedding $z_i$, we identify $k$-nearest neighbors $\mathcal N(i)$ in latent space. Each neighbor $j$ receives an influence weight
$$
\tilde I_j = (I_j-\mu)/\sigma,\;
w_j=\max(0,\tilde I_j),
$$
discarding negatively influential samples.

We compute a normalized corrective direction
$$
\Delta a_i^{\text{target}}
=
\sum_{j\in\mathcal N(i)} w_j (a_j-a_i),
\quad
\Delta a_i^{\text{target}}
\leftarrow
\frac{\Delta a_i^{\text{target}}}{\|\Delta a_i^{\text{target}}\|+\epsilon}.
$$

The editor predicts a latent residual $\delta z_i$, whose decoded action correction is trained to align with $\Delta a_i^{\text{target}}$.


## Baselines
We want to evaluate the following methods and compare their performance on the pen-v2 dataset
1. Vanilla BC: no augmentation to the dataset
2. Gaussian Filtering: applies temporal smoothing to action trajectories prior to training to test whether simple denoising improves performance
3. Vanilla BC + Influence Weighting: adjusts the training loss using influence scores without modifying the data itself
4. Random Latent Editing: applies random noise to the latent space of the VAE to test whether random perturbations improve performance
5. STRIDE: our proposed method