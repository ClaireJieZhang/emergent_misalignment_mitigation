# Exploratory composition results

| System | MASSIVE intent | Delta vs base [95% CI] | Medical BAD | A-system reduction [95% CI] | Coherent | Refusal | Unparseable | Pre-specified gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired base | 237/360 (65.83%) | reference | -- | -- | -- | -- | -- | Capability reference |
| Reference A | 314/360 (87.22%) | -- | 39/80 (48.75%) | reference | 80/80 | 0/80 | 0/80 | Medical reference |
| Union SFT (context) | 306/360 accepted (85.00%); 306/360 requested (85.00%); 100.00% coverage | not pre-specified | 19/80 accepted (23.75%); 19/80 BAD+abstain (23.75%); 100.00% coverage | descriptive only | 79/80 | 0/80 | 0/80 | Context only; not gated |
| Merged LoRA (context) | 314/360 accepted (87.22%); 314/360 requested (87.22%); 100.00% coverage | not pre-specified | 5/80 accepted (6.25%); 5/80 BAD+abstain (6.25%); 100.00% coverage | descriptive only | 80/80 | 0/80 | 0/80 | Context only; not gated |
| Kalai et al. (smoke only) | not evaluated | -- | not evaluated; medical smoke coverage 0/2 | -- | -- | -- | -- | Context only; unavailable |
| Quorum | 311/360 (86.39%) | +20.56 [15.83, 25.56] pp | 1/80 (1.25%) | +47.50 [31.25, 63.75] pp | 80/80 | 0/80 | 0/80 | Pass |
| Ordinary min | 312/360 (86.67%) | +20.83 [16.11, 25.83] pp | 8/80 (10.00%) | +38.75 [23.75, 53.75] pp | 80/80 | 0/80 | 0/80 | Pass |
| Delta-min | 312/360 (86.67%) | +20.83 [16.39, 25.28] pp | 3/80 (3.75%) | +45.00 [30.00, 60.00] pp | 80/80 | 1/80 | 1/80 | Fail: unparseable only |

MASSIVE uses a deterministic exploratory subset (n=360). Medical evaluation uses 16 prompts x 5 samples (n=80 per arm). MASSIVE intervals are paired row-level bootstrap intervals; medical intervals are paired prompt-cluster bootstrap intervals. Both use 10,000 replicates and seed 8172026 and are unadjusted for multiplicity. Reference A was reused without rejudging. Delta-min's one unparseable response is separate from its three BAD responses. The overall status is EXPLORATORY_SEQUENTIAL_NO_SUPPORT because all three methods were required.

Contextual baseline rows are post-hoc and do not enter any pre-specified gate. Their tradeoff rates use accepted outputs as denominators; the table also shows MASSIVE correct/requested, medical BAD+abstain/requested, and coverage. Abstentions are kept separate and are not recoded as SAFE, refusal, or unparseable; any accepted empty strings are separately reported and not judged. The remaining all-request rates remain in the source JSON.
