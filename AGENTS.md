# Deep Learning Course Authoring Guide

These instructions apply to the whole repository. Use them whenever creating or revising course notes, Reveal.js lectures, exercises, schedule entries, or supporting assets.

## Teaching approach

- Teach through one concrete problem before abstracting. Prefer a worked image-classification example over a list of definitions.
- Make formulas answer a question raised by an example. Do not present long sequences of equations without interpretation, a visual, or a decision for the student.
- Keep the five-part frame visible where useful: data, hypothesis, loss, optimization, and evaluation.
- Notes are the durable technical reference; slides are the visual, conversational teaching path. They should agree, but the notes should not merely transcribe the slides.
- Give every notes chapter an original, purpose-made visual cover that previews its central idea without embedded text. Use that cover as the opening visual and as the dominant entry on the Notes index; keep only the title and navigation label as supporting index text.

## Slides

- Prefer illustrations, real image examples, diagrams, score bars, decision regions, and error galleries over prose-heavy slides.
- Use progressive Reveal.js fragments when introducing alternatives, model families, calculations, or conclusions. Do not reveal the conclusion before students have seen the evidence.
- Build explanations as an ascending sequence of evidence, student prediction or interpretation, and then revealed conclusion. Prefer moments where students actively judge a result before the slide names the principle.
- Use each lecture's section-divider bar as semantic progress through that lecture. Span the available slide width, divide it visibly into one segment per section so completed and remaining sections are countable, fill it according to the current section number, and show the section fraction rather than using a short decorative underline beneath the title.
- Keep one principal idea per slide. Use large visuals and text readable at 1440×900.
- Keep text inside boxes and callouts at a stable, readable font size. Let copy wrap across two or more balanced lines when needed; never force it onto one long line or shrink the type merely to keep it on one line.
- Scale arrowheads and line weights to the diagram, and leave enough clearance at node boundaries for both a visible shaft and the complete arrowhead between adjacent nodes. Never let an edge appear stuck inside its source or merged into its destination. Arrows should clarify flow without competing with the nodes.
- Keep node labels moderate in size relative to the node's width and height. When a full technical term would crowd a node, use a standard acronym in the node and expand it nearby or in the narration.
- For computation graphs, distinguish identity from operation semantics: keep short mathematical identifiers such as $x$, $W_1$, and $h$, but represent recurring operations with consistent shape-and-icon symbols explained once in a legend. Do not repeat operation names or full formulas inside every node when the graph may grow.
- Use Mermaid for workflows and relationships, but size diagrams to use the available canvas. If a diagram remains thin, use taller nodes, line breaks, or a more suitable layout.
- Reuse the established visual language and assets in `quarto-site/lectures/01-what-is-learning/` when appropriate.
- When a diagram is the principal content of a slide, scale it to occupy most of the available canvas. Override thin default Mermaid sizing or redesign the layout rather than leaving large unused areas.
- When nodes or small multiples are the principal evidence, distribute them evenly across the available width and height with consistent sizing and non-overlapping labels; do not leave large unused regions while compressing the diagram into one corner.
- On a student's first exposure to an abstract claim, pair the formula or geometric statement with a concrete visual and walk through the mechanism before summarizing it.
- Introduce an activation function with its formula, output curve, derivative curve, principal strengths, and principal weaknesses. Follow the definition immediately with any exercise that directly practices it, then compare activations through a controlled interactive experiment when useful.

## Exercise convention

Every in-class exercise should follow this sequence:

1. State the givens and the exact task.
2. Display a conspicuous `PAUSE` panel with a suggested duration and one useful instruction.
3. Keep the solution hidden in a Reveal.js fragment.
4. Reveal a clearly labeled worked solution on the next advancement.
5. When useful, reveal a transfer or extension question after the solution.

In HTML notes, use a prompt followed by a `<details>` solution so readers choose when to reveal it. In PDF notes, keep the prompt before the worked solution and preserve enough visual separation to allow an honest pause.

## Project scope

- Assess only concepts that students have already encountered in the course sequence. A project may use machinery from a future chapter, but that machinery must be provided as clearly labeled infrastructure, treated as a temporary black box, and revisited when it is formally taught. Do not turn an upcoming topic such as backpropagation or optimization into a student TODO merely because the project needs it to run.
- When a project introduces a nontrivial class, object hierarchy, or hidden data structure, precede the implementation task with a diagram that makes object ownership, stored state, graph relationships, and information flow concrete. In particular, distinguish references to other objects from copied numerical values.
- When a project or homework becomes available, link it in both places on the schedule: the course-material button and the corresponding fourth-column project-track entry. The project-track title and description should form one clear clickable target; never leave an available project represented there as plain text.

## Publishing course material

- When a lecture, notes chapter, PDF, code session, project, or homework becomes available, update the corresponding schedule material button in the same change. Replace the muted placeholder with a working link, render the schedule, and verify that the generated HTML points to the rendered resource. Do not consider the material published or the task complete while its schedule entry still says that it is unavailable.

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

## Learning from review feedback

- Treat every instructor correction as either a local edit or a reusable course-authoring rule.
- Before applying a correction broadly, state the generalized rule in plain language so its intended scope is clear.
- When a correction expresses a reusable preference about pedagogy, sequencing, visual design, exercises, citations, navigation, or technical presentation, add the generalized rule to this file and apply it to future material.
- Do not turn a correction tied to one specific slide, example, or exceptional circumstance into a global constraint.
- Preserve previously accepted rules unless the instructor explicitly revises or replaces them.
