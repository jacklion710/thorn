# Audio Cloak Project

## Overview

The Audio Cloak Project is an open-source initiative aimed at developing a system for cloaking audio recordings to protect against unauthorized use in training deep learning models. Inspired by the groundbreaking work of the Fawkes project on image cloaking, our goal is to extend similar protections to the audio domain. This endeavor seeks to empower individuals with the ability to shield their music and other kinds of audio content from AI models trained without their consent, thus preserving ones unique style from generative AI platforms.

## Inspiration

Our work draws inspiration from the [Fawkes image cloaking software](https://github.com/Shawn-Shan/fawkes) developed by researchers from the University of Chicago. The Fawkes project demonstrated a novel approach to protecting personal images from unauthorized facial recognition models. By making subtle alterations to digital photos before sharing them online, Fawkes effectively "cloaks" an individual's identity, causing any derived facial recognition model to misidentify them.

The concept of audio cloaking was further encouraged by a [technical review](https://repository.fit.edu/cgi/viewcontent.cgi?article=1793&context=etd) and the collective efforts of the [University of Chicago's SAND Lab team](https://sandlab.cs.uchicago.edu/fawkes/). Their scientific contributions, outlined in their [published academic paper](https://arxiv.org/abs/2002.08327), and the broader implications reported in [University of Chicago News](https://news.uchicago.edu/story/uchicago-scientists-develop-new-tool-protect-artists-ai-mimicry), have laid the foundation for our project.

## Project Status

The Audio Cloak Project is currently in development and is a work in progress. Our team is actively working on algorithms and methods to cloak audio efficiently without compromising the original audio's perceptual quality. While the project is nascent, we are making strides toward a prototype that demonstrates the feasibility and effectiveness of audio cloaking.

## Open Source and Collaboration

This project is open source and we warmly invite collaboration from interested individuals and organizations. Whether you are a researcher, developer, privacy advocate, or just passionate about the ethical implications of AI, your contributions are welcome. We believe in the power of community-driven development and the importance of open collaboration to address the challenges posed by unauthorized use of AI technologies.

## Technical Details

Some of the features & techniques to consider along the way are listed below.  

## Special Features

* Spectral Features Manipulation: Altering the spectral content of the audio subtly, such as the spectral centroid, spectral bandwidth, and spectral contrast, to mislead audio recognition models without significantly affecting the perceived audio quality.
* Temporal Features Distortion: Temporal features, like zero-crossing rate and tempo, could be slightly modified. These changes would need to be imperceptible or minimally perceptible to human ears but significant enough to confuse AI models.
* Mel-Frequency Cepstral Coefficients (MFCCs) Alteration: MFCCs are widely used in voice recognition. Subtle perturbations in MFCCs could lead to misclassification by AI systems while being less noticeable to humans.
* Harmonic and Percussive Sound Perturbation: Separating audio into harmonic (tonal sound) and percussive (rhythm) components and introducing minor perturbations can mislead AI without drastically affecting human perception.
* Embedding Sub-audible Noise: Injecting noise that is sub-audible or at the edge of human hearing thresholds can act as a form of perturbation to mislead AI models without affecting audio quality for listeners.
* Voiceprint Disguising: For voice recognition systems, subtly altering pitch, timbre, and other voiceprint characteristics to prevent the AI from matching the cloaked audio to the original speaker's voiceprint.
* Rhythm and Beat Synchronization Errors: Introducing slight timing errors or shifts in the rhythm and beats of music could confuse AI models designed for genre recognition or music analysis.
* Dynamic Range Compression/Expansion: Slightly altering the dynamic range of the audio can affect AI recognition patterns related to loudness and energy without majorly impacting listener experience.
* Phase Distortion: Introducing small phase shifts in the audio signal can confuse AI models, especially those relying on phase information for localization or identification, with minimal perceptual changes for humans.
* Reverberation and Echo Manipulation: Adding or altering reverberation and echo in a controlled manner could serve as a technique to cloak audio signatures from AI detection systems, provided the changes remain subtle to the human ear.

## Perturbations

1. **Spectral Masking:**

Spectral masking involves adding noise in specific frequency bands where the original signal has lower energy. This technique exploits the psychoacoustic property of masking, where louder sounds can make it harder to hear quieter sounds at nearby frequencies. The initial masking can be applied without a model, with optimization used to refine the masking noise to be minimally perceptible while still effective at confusing AI models.

2. **Temporal Masking:**

Similar to spectral masking but applied over time, temporal masking adds noise or alterations before or after certain sounds. This can make it harder for models to identify the temporal features of the audio signal. The initial application of temporal masking does not require a model, but optimization can help adjust the timing and intensity of the masking to maximize cloaking effectiveness.

3. **Phase Perturbation:**

Perturbing the phase of the audio signal while keeping its magnitude spectrum intact can significantly alter the signal's waveform without changing its perceived sound quality. Phase perturbations can be calculated by manipulating the phase of the Fourier transform of the audio signal. Optimization can then ensure that these perturbations do not lead to perceptible artifacts.

4. **Psychoacoustic Hiding:**

This technique involves embedding the perturbation noise into parts of the audio signal that are less audible to humans due to psychoacoustic phenomena such as auditory masking. Initial perturbations can be designed based on generic psychoacoustic models and then optimized to ensure they are effectively hiding relevant audio features from AI detection.

5. **Random Noise Injection:**

Injecting carefully calibrated random noise into the audio can serve as an initial perturbation method. The characteristics of the noise (e.g., amplitude, frequency range) can be chosen based on general principles to minimize human perceptibility. Optimization can then be employed to adjust the noise properties to specifically target weaknesses in AI recognition systems.

6. **Time Stretching and Compression:**

Slightly altering the speed of the audio playback without changing its pitch can modify the temporal characteristics of the signal. These changes can be initially applied in a subtle manner and later refined through optimization to find the best balance between cloaking effectiveness and audio quality.

### Optimization Phase:

In all these techniques, the calculated perturbations serve as a starting point. The pretrained model comes into play during the optimization phase, where the initial perturbations are adjusted to specifically target the model's vulnerabilities. This iterative process involves evaluating the cloaked audio against the pretrained model, assessing the impact on model recognition accuracy, and fine-tuning the perturbations to maximize cloaking while maintaining audio fidelity from a human listener's perspective.

## Contributing

If you are interested in contributing to the Audio Cloak Project, please feel free to fork the repository, make your changes, and submit a pull request. We also appreciate any feedback, suggestions, or ideas that can help improve the project. For more detailed information on how to contribute, please refer to our CONTRIBUTING.md file. If you wish to be added as a collaborator please [email me](mailto:jack.lion710@gmail.com)

## Acknowledgments

We extend our gratitude to the Fawkes team and the SAND Lab at the University of Chicago for their pioneering work and inspiration. Their dedication to privacy and the ethical use of AI has paved the way for projects like ours.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
