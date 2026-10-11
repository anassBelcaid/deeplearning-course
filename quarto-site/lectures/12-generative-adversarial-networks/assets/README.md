# GAN chapter evidence

## Opening

`gan-hero.svg` is an original course-made schematic of the GAN setup (Goodfellow et al., 2014, https://arxiv.org/abs/1406.2661). The garment silhouettes are conceptual, not model outputs.

## Section 1: learning a source classifier

`build_section1.py` produces every measured image, score, and HTML gallery in this section. Run explicitly with `COURSE_DATA_DIR=/path/to/data python build_section1.py`. Requires the VAE homework's saved `beta1.pt`, torch, torchvision, numpy, and Pillow. Quarto reads stored assets and never trains or downloads data.

The generator is the frozen two-dimensional VAE decoder from the 90-minute homework. **This preliminary experiment is not GAN training.** All output images are decoder means for prior latent draws. No generator parameter changes; only a separate binary source classifier learns.

Protocol and per-checkpoint results are in `evidence/section1/manifest.json`, including the exact generator checkpoint SHA-256. Training: 6,000 dataset and 6,000 generated examples; validation: 1,000 of each source. The real validation partition is disjoint from both the VAE and classifier training data; generated validation examples use a separate latent stream. Garment categories are not classifier targets. The classifier is a 784–128–64–1 MLP trained for eight epochs with BCEWithLogitsLoss; visible scores are sigmoid outputs and the decision threshold is 0.5. They are not calibrated universal realism estimates.

The source exercise uses the first three examples per held-out source, interleaved in the declared order R/G/G/R/G/R. It is illustrative, not a representative quality benchmark. All six gallery examples are kept regardless of correctness or confidence. Labels and solution reveal together.

The unpaired-target experiment keeps the first generated output fixed and changes only the candidate target. Targets are the first held-out examples of class 0 (T-shirt), 9 (ankle boot), and 8 (bag), without selection by MSE. Metrics use original floating-point tensors; displayed PNGs are rounded to eight bits. Difference images use a common absolute-difference color scale, black=0 and orange=1; the displayed scalar is mean squared difference over pixels.

The final feedback diagram is explicitly schematic. Its actual image tiles reuse the earlier experiment as tokens; it does not imply the shown image has improved through adversarial training.

All observed garment images: Fashion-MNIST, Xiao, Rasul, and Vollgraf (2017), https://arxiv.org/abs/1708.07747. Models, figures, and interface: created for this course. Architecture, losses, and alternating GAN updates are reserved for later sections.

## Section 2: executable module contracts

`gan_modules.py` defines the exact MLP generator and discriminator displayed in the slides. Run it with the existing course Python environment and `COURSE_DATA_DIR` pointing to the cached Fashion-MNIST root. It records four real images and four outputs of the **untrained** generator, plus `evidence/section2/contracts.json`. There is no optimizer and no training. Model initialization seed is 1202; latent seed is 1203. Real images use the first four held-out examples in the split defined by permutation seed 41 (positions 12000–12003), scaled to [0,1].

Checks cover shapes for batches 1, 4, and 7, finite values, generator range, separate G/D parameter ownership, deterministic generation for fixed inputs, changing outputs for new codes, and the same unchanged D applied to both sources. Batch concatenation equivalence is tested only for this deterministic MLP without batch-dependent layers. Outputs demonstrate programming contracts, not GAN generation quality.

`build_section2_diagrams.py` produces the original SVG node diagrams and extracts both code editors from marked regions of `gan_modules.py`, keeping visible code synchronized with its executable source. The comparison uses symbols for reconstruction and generated images; these are schematic, not claimed model results. Actual images beside the code come from the recorded contract check. Fashion-MNIST attribution: https://github.com/zalandoresearch/fashion-mnist . The architecture comparison follows the original VAE and GAN papers linked on the slide.

The routing buttons in `section2-controls.html` highlight forward paths; they do not run inference, train models, or imply which parameters receive optimizer updates. Shared network-node styles live in the course `01-what-is-learning/slides.css` component library; section-specific geometry remains in `week12.css`.

## Section 3: objectives and gradients

`build_section3.py` reproduces the loss graph, analytic gradient plot, code exercise, and `evidence/section3/loss-checks.json`. It needs PyTorch but performs no training and requires no data download. The loss-code HTML is extracted from an executed source block. The arithmetic uses natural logs and the sum of the real-source and generated-source means for D, with no half factor. G uses the non-saturating target-one objective.

The opening p=0.8/0.2 predictions and score slider are explicitly hypothetical scalar examples, not model measurements. The last slide instead applies the displayed loss code to the actual archived section 2 untrained logits: D loss 1.4043543, G loss 0.7014364. The gradients are analytic derivatives with respect to the discriminator logit, independently checked by float64 autograd at p=0.01, 0.2, 0.5, 0.8, and 0.99. They are not full parameter-gradient norms or measured training trajectories.

`section3-controls.html` updates both losses on a shared meter scale and selects points on the two logit-gradient curves. The slider changes no images or model weights. The gradient plot and objectives follow §3 of Goodfellow et al. (2014), https://arxiv.org/abs/1406.2661; the loss API follows the official PyTorch BCEWithLogitsLoss documentation linked on the slide. Optimizer ownership and separate forward passes are deliberately developed in section 4.

## Section 4: alternating updates and a real gradient failure

`build_section4.py` initializes the exact section 2 MLPs with seed 1204, selects the first four training examples from the Fashion-MNIST permutation with seed 41, and runs one D step followed by one G step. Both independent Adam optimizers use lr=0.001. It extracts the displayed code from the blocks it actually executes and records `evidence/section4/update-checks.json`.

The audit compares all parameter tensors bit for bit before and after each step: D turn changes 6/6 D tensors and 0/6 G tensors; G turn changes 6/6 G tensors and 0/6 D tensors. Each model has three linear layers, hence six weight/bias tensors. It checks gradient production separately, clearing old gradients before measurement. The generator pass uses fresh forward computations through the updated D. The same latent batch is reused. This is a correctness check, not evidence of image quality or convergence.

The script also executes the no_grad bug from the final exercise, verifies loss.requires_grad=False, captures the RuntimeError at backward, and checks that G has no gradients. `build_section4_diagrams.py` reproduces the original node diagrams. `section4-controls.html` switches the schematic update paths without model execution. The example's MLP discriminator has no BatchNorm or dropout; freezing trainable parameters should not be interpreted as a general policy for mutable buffers.

## Section 5: convolutional architecture and recorded training

`conv_gan.py` defines the exact G and D displayed in the code toggle and trains a small DCGAN-inspired adaptation on the first 12,000 Fashion-MNIST training examples in permutation seed 41. It uses 16 latent channels at 1×1, a 7×7 ConvTranspose projection, two stride-two expansions, and a Tanh image output; D uses strided convolution and emits raw logits. It is not a reproduction of the DCGAN paper. Real pixels are scaled by x/127.5−1.

Both independent Adam optimizers use lr=0.0002 and betas=(0.5,0.999), with one D turn then one non-saturating G turn per batch. G uses BatchNorm: its statistics update only on G turns; D turns and recorded samples use G.eval(). D has no BatchNorm. This mode policy is explicitly documented on the slide and in speaker notes.

The initial 20-epoch run was extended to 60 after inspecting its rough outputs. Its original checkpoint images remain visible at epochs 0,1,5,20 alongside epoch 60. Six fixed normal draws (seed 1206) are used throughout; none is ranked, removed, or replaced. Initialization seed is 1205, minibatch-order seed 1207, batch size 128 with the final batch retained (94 updates/epoch). `evidence/section5/run.json` records the full protocol, losses, runtime, checks, and final checkpoint hash. `final-checkpoint.pt` contains G, D, and the fixed codes. The viewer uses local PNGs and never trains or runs inference in the browser. No MLP baseline, diversity metric, or general quality claim is implied.

The builder verifies batch shapes at sizes 1,4,128 and the exact toy transposed convolution [1,2] with kernel [1,2,1], stride 2, giving [1,2,3,4,2]. `build_section5.py` creates the original layer-bar diagrams and extracts code editors from the training module definitions. `section5-controls.html` switches module views and all recorded checkpoints. References: Radford et al., https://arxiv.org/abs/1511.06434; official PyTorch ConvTranspose2d documentation; Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist.

### Paired spatial architecture overview

`build_spatial_architecture.py` generates `spatial-architecture.html`, an original paper-style feature-volume view of the exact convolutional G and D. Feature-map faces grow through G and shrink through D; stack depth suggests channels and exact C×H×W labels remain authoritative. Geometry is schematic, not measured activation evidence. The repeated image tile is actual epoch-60 draw 1, connecting G output to a possible D input; D also accepts dataset images. G appears first and D is revealed on advancement. Kernel/stride/padding labels match `conv_gan.py`.

## Section 6: evaluation and synthesis

`build_section6.py` loads the section 5 final checkpoint without training. It generates 36 fresh outputs from seed 1208 in eval mode and records three consecutive 12-image batches alongside the first 36 held-out Fashion-MNIST reference examples (split seed 41, indices 12000–12035). Neither gallery is ranked or filtered. The repetition control is explicitly constructed by copying the first output of the current batch twelve times; it is not an observed collapse of the trained model.

For fresh outputs 1,2,3, the builder performs exhaustive pixel-MSE nearest-neighbor search against all 12,000 training images, using floating-point generated outputs in [0,1]. `evaluation.json` stores exact image IDs, distances, selection protocol, limitations, and the generator checkpoint hash. This is a visual sanity check, not a memorization test or novelty guarantee.

The training-loss plot uses every recorded epoch mean from section 5, on a shared 0–4 vertical axis, without smoothing. Checkpoint buttons connect epochs 5,20,60 to the same fixed draws 1 and 5 from the earlier viewer. The training losses were recorded on alternating minibatches, not recomputed on the displayed images. `section6-controls.html` handles gallery batches, the labeled repetition control, and the loss-marker/sample updates. The final exercise synthesizes the previously taught five-part pipeline and sampling path without adding a new model family or metric.
