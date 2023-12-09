# Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models

Implementation of the Paper
  - B. Fesl, N. Turan, B. Böck, and W. Utschick, "Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models," 2023. <br>
  Link to paper: https://arxiv.org/abs/2305.03427

## Abstract

This work introduces a novel class of channel estimators tailored for coarse quantization systems. 
The proposed estimators are founded on conditionally Gaussian latent generative models, specifically Gaussian mixture models (GMMs), mixture of factor analyzers (MFAs), and variational autoencoders (VAEs). 
These models effectively learn the unknown channel distribution inherent in radio propagation scenarios, providing valuable prior information. 
Conditioning on the latent variable of these generative models yields a locally Gaussian channel distribution, thus enabling the application of the well-known Bussgang decomposition. 
By exploiting the resulting conditional Bussgang decomposition, we derive parameterized linear minimum mean square error (MMSE) estimators for the considered generative latent variable models. 
In this context, we explore leveraging model-based structural features to reduce memory and complexity overhead associated with the proposed estimators. 
Furthermore, we devise necessary training adaptations, enabling direct learning of the generative models from quantized pilot observations without requiring ground-truth channel samples during the training phase. 
Through extensive simulations, we demonstrate the superiority of our introduced estimators over existing state-of-the-art methods for coarsely quantized systems, 
as evidenced by significant improvements in mean square error (MSE) and achievable rate metrics.

## Scripts
  - **Bussgang_GMM.py** <br>
  Implementation of the Bussgang-GMM for different covariance structures. This script also includes the implementation of Buss-Scov, Buss-genie, and BLS.
  - **Bussgang_GMM_quant.py** <br>
  Implementation of the Bussgang-GMM to learn from quantized training data via the proposed covariance recovery scheme.
  - **Bussgang_MFA.py** <br>
  Implementation of the Bussgang-MFA.
  - **Bussgang_VAE.py** <br>
  Implementation of all Bussgang-VAE variants to learn from either perfect CSI data (VAE-noisy) or from quantized training data (VAE-real).
  - **Covariance_recovery.py** <br>
  Script to reproduce Fig. 2 in the paper.

## Possible GMM Covariance Structures

The following covariance structures are supported for the GMM variant:
  - 'full' (full covariance matrix with no structural constraints for each GMM component)
  - 'circulant' (Circulant covariance matrix for each GMM component
  - 'block-circulant' (Block-circulant covariance matrix with circulant blocks for each GMM component, use keyword 'blocks' in 'fit')
  - 'toeplitz' (Toeplitz covariance matrix for each GMM component)
  - 'block-toeplitz' (Block-Toeplitz covariance matrix with Toeplitz blocks for each GMM component, use keyword 'blocks' in 'fit')

## Related Repositories

  - Complex-valued implementation the expectation-maximization (EM) algorithm for Gaussian mixture models (GMMs): <br>
  https://github.com/benediktfesl/GMM_cplx
  - Complex-valued implementation the expectation-maximization (EM) algorithm for Mixtures of Factor Analyzers (MFAs): <br>
  https://github.com/benediktfesl/MFA_cplx
  - Implementation of the GMM channel estimator for high-resolution systems: <br>
  https://github.com/michael-koller-91/gmm-estimator
  - Implementation of the MFA channel estimator for high-resolution systems: <br>
  https://github.com/benediktfesl/MFA_estimator
  - Implementation of the VAE channel estimator for high-resolution systems: <br>
  https://github.com/tum-msv/vae-estimator

## Related Works

  - M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models,” IEEE Trans. Signal Process., vol. 70, pp. 4109–4123, 2022. <br>
  https://ieeexplore.ieee.org/abstract/document/9842343
  - N. Turan, B. Fesl, M. Grundei, M. Koller, and W. Utschick, “Evaluation of a Gaussian Mixture Model-based Channel Estimator using Measurement Data,” in Int. Symp. Wireless Commun. Syst. (ISWCS), 2022. <br>
  https://ieeexplore.ieee.org/abstract/document/9940363
  - B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” in 56th Asilomar Conf. Signals, Syst., Comput., 2022, pp. 533–537. <br>
  https://ieeexplore.ieee.org/abstract/document/10051921
  - B. Fesl, N. Turan, M. Joham, and W. Utschick, “Learning a Gaussian Mixture Model from Imperfect Training Data for Robust Channel Estimation,” IEEE Wireless Commun. Lett., 2023. <br>
  https://ieeexplore.ieee.org/abstract/document/10078293
  - M. Koller, B. Fesl, N. Turan and W. Utschick, "An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models," IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2022, pp. 5268-5272. <br>
  https://ieeexplore.ieee.org/abstract/document/9747226
  - B. Fesl, A. Faika, N. Turan, M. Joham, and W. Utschick, “Channel Estimation with Reduced Phase Allocations in RIS-Aided Systems,” in IEEE 24th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC), 2023, pp. 161-165. <br>
  https://ieeexplore.ieee.org/document/10304464
  - N. Turan, B. Fesl, M. Koller, M. Joham, and W. Utschick, “A Versatile Low-Complexity Feedback Scheme for FDD Systems via Generative Modeling,” in IEEE Transactions on Wireless Communications, 2023. <br>
  https://ieeexplore.ieee.org/document/10318056
  - N. Turan, B. Fesl, and W. Utschick, "Enhanced Low-Complexity FDD System Feedback with Variable Bit Lengths via Generative Modeling," in 57th Asilomar Conf. Signals, Syst., Comput., 2023. <br>
  https://arxiv.org/abs/2305.03427
  - N. Turan, M. Koller, B. Fesl, S. Bazzi, W. Xu and W. Utschick, "GMM-based Codebook Construction and Feedback Encoding in FDD Systems,"in 56th Asilomar Conf. Signals, Syst., Comput., 2022, pp. 37-42. <br>
  https://ieeexplore.ieee.org/abstract/document/10052020
  - M. Baur, B. Fesl, and W. Utschick, "Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation," 2023. <br>
  https://arxiv.org/abs/2307.05352
  - M. Baur, B. Fesl, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," in 56th Asilomar Conf. Signals, Syst., Comput., 2022, pp. 527-532. <br>
  https://ieeexplore.ieee.org/abstract/document/10051858

## Original License
The original code from https://scikit-learn.org/stable/modules/mixture.html is covered by the following license:

> BSD 3-Clause License
>
> Copyright (c) 2007-2023 The scikit-learn developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
## Licence of Contributions
The contributions and extensions are also covered by the BSD 3-Clause License:

> BSD 3-Clause License
>
> Copyright (c) 2023 Benedikt Fesl.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
