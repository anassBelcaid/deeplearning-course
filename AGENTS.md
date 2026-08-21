# Deep Learning Course Authoring Guide

These instructions apply to the whole repository. Use them whenever creating or revising course notes, Reveal.js lectures, exercises, schedule entries, or supporting assets.

## Teaching approach

- Teach through one concrete problem before abstracting. Prefer a worked image-classification example over a list of definitions.
- Make formulas answer a question raised by an example. Do not present long sequences of equations without interpretation, a visual, or a decision for the student.
- Keep the five-part frame visible where useful: data, hypothesis, loss, optimization, and evaluation.
- Notes are the durable technical reference; slides are the visual, conversational teaching path. They should agree, but the notes should not merely transcribe the slides.

## Slides

- Prefer illustrations, real image examples, diagrams, score bars, decision regions, and error galleries over prose-heavy slides.
- Use progressive Reveal.js fragments when introducing alternatives, model families, calculations, or conclusions. Do not reveal the conclusion before students have seen the evidence.
- Keep one principal idea per slide. Use large visuals and text readable at 1440×900.
- Use Mermaid for workflows and relationships, but size diagrams to use the available canvas. If a diagram remains thin, use taller nodes, line breaks, or a more suitable layout.
- Reuse the established visual language and assets in `quarto-site/lectures/01-what-is-learning/` when appropriate.

## Exercise convention

Every in-class exercise should follow this sequence:

1. State the givens and the exact task.
2. Display a conspicuous `PAUSE` panel with a suggested duration and one useful instruction.
3. Keep the solution hidden in a Reveal.js fragment.
4. Reveal a clearly labeled worked solution on the next advancement.
5. When useful, reveal a transfer or extension question after the solution.

In HTML notes, use a prompt followed by a `<details>` solution so readers choose when to reveal it. In PDF notes, keep the prompt before the worked solution and preserve enough visual separation to allow an honest pause.

## Images and attribution

- Use as many pedagogically useful images as the explanation benefits from; do not add decorative images that carry no teaching information.
- Prefer primary or authoritative sources: original papers, official datasets, authors' project pages, and official course materials.
- CS231n illustrations may be reused when they are the clearest explanation. Attribute them visibly to the Stanford CS231n course notes and cite the notes in the bibliography.
- Store fetched or generated assets locally beside the lecture. Give every meaningful image useful alt text.
- For a borrowed image, include a visible source in the caption or immediately below it. For a course-made diagram derived from a paper, say that the diagram was created for the course and cite the paper.
- When generating a multi-stage technical figure, keep the source or generation script when practical so the figure is reproducible.

## Citations

- Cite frequently enough that students can trace important claims, methods, datasets, and architectures.
- Prefer original work for algorithms and architectures: for example, Canny for edge detection, Cover and Hart for nearest neighbors, Cortes and Vapnik for SVMs, and Simonyan and Zisserman for VGG.
- Use reviews or textbooks for synthesis, not as a substitute for primary citations when an original source is available.
- Add reusable entries to `quarto-site/references.bib`; do not scatter ad hoc bibliographies across chapters.
- Ensure all citation keys resolve in both HTML and PDF builds.

## Quarto HTML/PDF compatibility

- When an HTML-only custom layout needs `content-visible`, nest the layout inside the conditional container:

  ```markdown
  :::{.content-visible when-format="html"}
  :::{.custom-layout}
  ...
  :::
  :::
  ```

  Do not put `.custom-layout` and `.content-visible` on the same fenced div. Quarto unwraps the conditional container and the layout class will be lost.

- Provide a PDF-safe alternative for complex HTML-only grids when necessary.
- After substantive note changes, render both HTML and PDF. Check for unresolved citations, warnings, malformed conditional blocks, missing images, and whitespace errors.
- Visually inspect representative pages/slides rather than relying only on a successful render.

## Synchronization workflow

When a lecture section is approved:

1. Update the corresponding notes with the same examples, images, terminology, and references.
2. Add the deeper derivation and caveats that do not fit on slides.
3. Confirm schedule links still point to the rendered notes, PDF, and lecture.
4. Render and verify both formats before declaring the section complete.
