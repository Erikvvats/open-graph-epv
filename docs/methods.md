# What the model estimates

For a current state S, let J be the first shot strictly after the current frame
and within 15 seconds in the same period. Let q(J) be the transferred xG teacher's
estimate. Define Y_home = q(J) if J belongs to the home team and zero otherwise;
define Y_away analogously. No shot means both labels are zero. The full horizon
must be available. This uses elapsed time, including interruptions, not a fixed
possession or the first future shot separately for each team.

The outputs aim to estimate E[Y_home | S] and E[Y_away | S]. Both can be positive
before the future is observed. Their difference is a derived net value. Their
sum is expected first-shot xG, not shot probability or total expected goals.
Calling this EPV describes a short-horizon value proxy, with these qualifications.

## Why MSE?

For scalar Y, expected squared loss is minimized at E[Y | S]. This matches the
estimand. Smooth L1/Huber instead solves a clipped-residual estimating equation;
with zero-heavy, skewed targets it need not recover the conditional mean.
MSE is sensitive to rare high labels: inspect those labels, report concentration,
and evaluate uncertainty rather than silently changing the estimand. The actual
loss averages squared errors across both components, not only their difference.

## Contextual xG teacher

An L2 logistic regression learns actual goal/non-goal outcomes on StatsBomb shots.
Its 18 features describe geometry, goalkeeper position and defenders. Match splits
separate teacher training, model selection and evaluation. Our generated values
are not official StatsBomb xG. The frozen geometry model supplies 113 of the 526
SkillCorner shot labels where context is unavailable or unreliable; 413 use the
contextual model. This hybrid introduces teacher and cross-source transfer error.

For expected shot xG to equal a goal probability at an earlier state, the teacher
must adequately capture goal-relevant context conditional on that state. Aggregate
calibration alone is insufficient. First-shot scoring also differs from scoring
at least once over a window containing possible rebounds or later shots.

## Graph and controls

Up to 23 nodes represent the ball and 22 players. Each has 10 features: normalized
x/y, backward vx/vy, possession-team/opponent indicators, ball, goalkeeper,
detected and valid-velocity flags. Invalid/unobserved nodes are masked, not treated
as players at the centre. Player IDs are used to align frames, not as features.

Two message-passing layers combine node embeddings with relative positions,
distance and same-team flags on all valid non-self edges. Masked mean aggregation
and separate possession/opponent/ball pooling feed the output head. A no-message
model uses player features and pooling but disables inter-node messages. A ball
MLP and training-mean prior test whether added complexity contributes anything.

Three logits followed by softmax produce two nonnegative values with sum at most
one. The unused third coordinate is **not** interpreted as no-shot probability.
Internal components are possession-relative; metadata maps them to fixed home/away
identities. Channel swapping never negates a prediction.

## Symmetry and evaluation

Coordinates orient the possession team towards positive x. Reflection across the
pitch's longitudinal axis flips y and vy. Mirror augmentation and averaging the
two reflected predictions impose left/right invariance under the assumption that
these features contain no relevant side-specific information. This may remove real
asymmetry from footedness or player roles absent from the features. It is a modelling
assumption, not evidence of football understanding.

Match splitting prevents neighbouring windows crossing train/test boundaries;
windows inside a match remain dependent. Three seeds are averaged per family and
validation two-component MSE selects the family. The set model wins validation;
the GNN happens to score slightly better on the previously inspected test set.
No architecture superiority claim follows from this reversal. Seed dispersion is
not a confidence interval over future matches.

There is no arbitrary acceptance threshold that turns this into a validated
football model. Baseline improvement, label audits, independent matches and a
specified use case are required. Earlier direction-agreement thresholds and
net-value experiments are historical and do not validate this two-output model.
