// Unified grid-wide subtree curriculum (SR + sexp, all backbones, seed 42).
// Geometric ramp 8->16->32->64->full: 4 warmup rungs (20 epochs total) then 40
// full-document epochs, so subtree and full-doc phases get equal gradient steps
// (~1,560 each at batch1/accum8 on 309-doc RST-DT) -- the full-doc phase, the only
// validated phase where accuracy is made, spans the back half of the run. This
// replaces the split recipe (SR 2-phase/60 vs sexp 4-phase/75), which (a) confounded
// the SR-vs-sexp contrast with a curriculum contrast and (b) on sexp starved the
// full-doc phase of learning rate. The rung at 16 targets the mid-width 5-16 EDU
// cascade band; the 64->full jump is mild (66% of train docs are <=64 EDUs whole)
// and gentler than the proven-safe sexp 60->full, so the decoder-only-sexp cold-start
// collapse stays guarded. expand_factor 2.0 gives four distinct seeded subsamples
// across the rungs (~2,470 distinct warmup examples) vs 618 ground repeatedly.
{ type: 'subtree_size', size_schedule: [8, 16, 32, 64, null], phase_epochs: [8, 4, 4, 4, 40], max_epoch_expand_factor: 2.0 }
