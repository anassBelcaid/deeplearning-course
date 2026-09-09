# Chapter 6 · Building Training Systems with PyTorch

## Agreed level-1 table of contents

1. **From NumPy to PyTorch**  
   Revisit the same Fashion-MNIST classification problem and identify what PyTorch adds to the mathematical machinery students already know.

2. **Tensors, Shapes, and Devices**  
   Creating tensors, indexing, broadcasting, dtypes, CPU/GPU movement, and NumPy interoperability.

3. **Autograd: Backpropagation Becomes Automatic**  
   `requires_grad`, dynamic computation graphs, `backward()`, `.grad`, and disabling gradient tracking.

4. **Models as `nn.Module` Objects**  
   Layers, registered parameters, `forward`, model composition, and architecture inspection.

5. **Datasets, Transforms, and DataLoaders**  
   Turning individual examples into transformed, shuffled mini-batches suitable for training.

6. **The Training and Validation Loops**  
   Forward pass, loss, zeroing gradients, backward pass, optimizer step, metrics, and `train()`/`eval()`.

7. **A Complete Reusable Training System**  
   Device-safe execution, checkpoints, reproducibility, inference, and the final Fashion-MNIST pipeline.

## Chapter flow

```text
data representation
→ automatic differentiation
→ model representation
→ data pipeline
→ training loop
→ deployable checkpoint
```

## Agreed teaching decisions

- Use one Fashion-MNIST classifier as the concrete thread through the chapter.
- Present PyTorch as an organization of concepts students have already learned, not as a new collection of disconnected syntax.
- Keep the autograd section concise because the mathematical mechanism of backpropagation is already known.
- Give most attention to how tensors, modules, data loaders, optimizers, modes, and checkpoints cooperate in a reliable training system.
- Connect `model.train()` and `model.eval()` directly to the BatchNorm and dropout behavior established in Chapter 5.
- Preserve the five-part frame where useful: data, hypothesis, loss, optimization, and evaluation.
- End with a complete pipeline that students can reuse in the following computer-vision chapters.

## Continuation point

The next authoring step is to design the section progression and slide sequence for **Section 1 · From NumPy to PyTorch**.
