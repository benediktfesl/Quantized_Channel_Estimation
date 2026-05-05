# Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models

Implementation of the paper
>B. Fesl, N. Turan, B. Böck, and W. Utschick, "Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models," in *IEEE Transactions on Signal Processing*, 2024. <br>

[[IEEE](https://ieeexplore.ieee.org/abstract/document/10454252)] [[arXiv](https://arxiv.org/abs/2309.04014)]

## Overview

This repository contains implementations of channel estimators for coarsely quantized systems based on conditionally Gaussian latent generative models.

The implemented variants include:

- Bussgang-GMM
- Bussgang-MFA
- Bussgang-VAE
- covariance recovery from quantized training data

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
  - `full` (full covariance matrix with no structural constraints for each GMM component)
  - `circulant` (Circulant covariance matrix for each GMM component)
  - `block-circulant` (Block-circulant covariance matrix with circulant blocks for each GMM component, use keyword `blocks` in `fit`)
  - `toeplitz` (Toeplitz covariance matrix for each GMM component)
  - `block-toeplitz` (Block-Toeplitz covariance matrix with Toeplitz blocks for each GMM component, use keyword `blocks` in `fit`)

## Related Repositories

  - Complex-valued implementation of the expectation-maximization (EM) algorithm for Gaussian mixture models (GMMs): <br>
  [[GitHub](https://github.com/benediktfesl/cplx-gmm)] [[PyPI](https://pypi.org/project/cplx-gmm/)]
  - Complex-valued implementation of the expectation-maximization (EM) algorithm for Mixtures of Factor Analyzers (MFAs): <br>
  [[GitHub](https://github.com/benediktfesl/cplx-mfa)]
  - Implementation of the GMM channel estimator for high-resolution systems: <br>
  [[GitHub](https://github.com/michael-koller-91/gmm-estimator)]
  - Implementation of the MFA channel estimator for high-resolution systems: <br>
  [[GitHub](https://github.com/benediktfesl/MFA_estimator)]
  - Implementation of the VAE channel estimator for high-resolution systems: <br>
  [[GitHub](https://github.com/tum-msv/vae-estimator)]

## Related Works

  - M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models,” *IEEE Transactions on Signal Processing*, vol. 70, pp. 4109–4123, 2022. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9842343)] [[arXiv](https://arxiv.org/abs/2112.12499v2)]
  - N. Turan, B. Fesl, M. Grundei, M. Koller, and W. Utschick, “Evaluation of a Gaussian Mixture Model-based Channel Estimator using Measurement Data,” in *International Symposium on Wireless Communication Systems (ISWCS)*, 2022. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9940363)] [[arXiv](https://arxiv.org/abs/2207.14150)]
  - B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” in *56th Asilomar Conference on Signals, Systems, and Computers*, 2022, pp. 533–537. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10051921)] [[arXiv](https://arxiv.org/abs/2205.03634)]
  - B. Fesl, N. Turan, M. Joham, and W. Utschick, “Learning a Gaussian Mixture Model from Imperfect Training Data for Robust Channel Estimation,” *IEEE Wireless Communication Letters*, 2023. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10078293)] [[arXiv](https://arxiv.org/abs/2301.06488)]
  - M. Koller, B. Fesl, N. Turan and W. Utschick, "An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models," *IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, 2022, pp. 5268-5272. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9747226)] [[arXiv](https://arxiv.org/abs/2111.11064)]
  - B. Fesl, A. Faika, N. Turan, M. Joham, and W. Utschick, “Channel Estimation with Reduced Phase Allocations in RIS-Aided Systems,” in *IEEE 24th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, 2023, pp. 161-165. <br>
  [[IEEE](https://ieeexplore.ieee.org/document/10304464)] [[arXiv](https://arxiv.org/abs/2211.07552)]
  - N. Turan, B. Fesl, M. Koller, M. Joham, and W. Utschick, “A Versatile Low-Complexity Feedback Scheme for FDD Systems via Generative Modeling,” in *IEEE Transactions on Wireless Communications*, 2023. <br>
  [[IEEE](https://ieeexplore.ieee.org/document/10318056)] [[arXiv](https://arxiv.org/abs/2304.14373)]
  - N. Turan, B. Fesl, and W. Utschick, "Enhanced Low-Complexity FDD System Feedback with Variable Bit Lengths via Generative Modeling," in *57th Asilomar Conference on Signals, Systems, and Computers*, 2023. <br>
    [[IEEE](https://ieeexplore.ieee.org/document/10477075)] [[arXiv](https://arxiv.org/abs/2305.03427)]
  - N. Turan, M. Koller, B. Fesl, S. Bazzi, W. Xu and W. Utschick, "GMM-based Codebook Construction and Feedback Encoding in FDD Systems,"in *56th Asilomar Conference on Signals, Systems, and Computers*, 2022, pp. 37-42. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10052020)] [[arXiv](https://arxiv.org/abs/2205.12002)]
  - M. Baur, B. Fesl, and W. Utschick, "Leveraging Variational Autoencoders for Parameterized MMSE Estimation," in *IEEE Transactions on Signal Processing*, vol. 72, pp. 3731-3744, 2024. <br>
  [[IEEE](https://ieeexplore.ieee.org/document/10629241)] [[arXiv](https://arxiv.org/abs/2307.05352)]
  - M. Baur, B. Fesl, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," in *56th Asilomar Conference on Signals, Systems, and Computers*, 2022, pp. 527-532. <br>
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10051858)] [[arXiv](https://arxiv.org/abs/2205.05345)]

## License

This repository is distributed under the BSD 3-Clause License.

Parts of the implementation are derived from scikit-learn's mixture module, which is also licensed under the BSD 3-Clause License.

See [`LICENSE`](LICENSE) for details.
