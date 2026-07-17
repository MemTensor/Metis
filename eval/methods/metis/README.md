# Metis method adapter

Metis uses the repository-local `metis` package and a checkpoint supplied with
`--checkpoint`. Checkpoints are external assets and must never be committed.
The loader rejects incomplete full checkpoints and resets memory between
benchmark instances.
