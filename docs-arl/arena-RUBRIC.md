# Qualitative tie-break rubric

Applies ONLY to artifacts whose objective score already ties. The hidden tests have
already decided correctness; this pass decides which correct answer is the better piece of
engineering. Score each dimension 0-3 and record a one-line justification.

1. **Failure handling** -- what happens on input the spec did not mention? Does it degrade
   sensibly or panic/silently corrupt?
2. **Fidelity to the spec's intent** -- did it implement what was asked, or something
   adjacent that happens to satisfy the assertions? Tests-shaped code is a negative.
3. **Clarity** -- would you accept this in review without rewriting it? Naming, structure,
   dead code, needless cleverness.
4. **Comment discipline** -- do comments explain *why* where the reason is non-obvious, and
   stay absent where the code already says it? Restating the code is a negative.
5. **Economy** -- did it solve the problem without inventing scaffolding nobody asked for?

Deliberately NOT scored: visual design of any HTML (unverifiable here), and raw length.

Record for each label: the five scores, total, and the single most telling detail.
