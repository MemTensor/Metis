# LoCoMo and NextMem

Inference and scoring code for the canonical LoCoMo evidence-session TPS16 and
NextMem official Task-2 STM inputs. Normalized payload paths, counts, bytes, and
hashes are centralized in `eval/data/manifest.json`. Data-building history is
not required to run the released evaluation and is intentionally excluded.

Memory-only methods receive only the question and internal state at query time;
full- and partial-context methods are explicitly labeled by the dispatcher.
