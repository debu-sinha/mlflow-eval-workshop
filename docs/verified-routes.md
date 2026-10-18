# Verified workshop routes

The current revision uses MLflow 3.17.1, OpenAI 3.8.0 and the committed dependency lock. Local setup uses Python 3.12.13. The native Databricks route uses Standard v5 with Python 3.12.3 and `requirements-workshop.txt`.

The three primary notebooks completed with fresh local kernels and as native Databricks serverless notebook tasks on October 18, 2026. The first complete outcomes are preserved below. Execution completion and acceptance are separate results.

| Route | Application and judge | All required checks, deployed to repair | Controls | Recorded decision after model review |
| --- | --- | --- | --- | --- |
| Local | OpenAI `gpt-4o-mini` | 4/10 to 9/10, five gains, no losses | 20/20 | REVIEW_REQUIRED |
| Native Databricks | Configured Qwen serving endpoint | 2/10 to 10/10, eight gains, no losses | 20/20 | PASS under the workshop policy |

The local judge incorrectly attributed the customer's request for an already processed refund to the assistant. The assistant had stated correct eligibility and a permitted customer-to-support next step. That recorded zero remains unchanged, and its material review finding remains open. The native grades agreed with the reviewed business rubric. Two related rationale findings remain recorded as nonmaterial for that comparison. Passing the authored controls did not establish judge accuracy.

Both bundles identify the models that reviewed their answers and grades. These are model reviews, not independent human labels or production approval. Each route uses the same model for its application and judge. Open the review panel in notebook 02 to inspect the customer input, answer and rationale before deciding.

Saved mode contains dated, hash-verified outputs from those actual runs in `evidence/workshop/local/final` and `evidence/workshop/databricks/final`. Local and Databricks routes remain separate. Do not pool their scores or treat ten teaching cases as production coverage. Their manifest source identities describe the measured implementation. The final notebook adds a read-only review display without changing measurements.

Historical answers were also rescored with the frozen v2 judge, without regenerating answers or replacing v1 decisions. Local totals changed from 8/6/9 to 8/3/8 and still contained wrong rejections. Native totals changed from 10/3/10 to 10/2/10. A revised rubric is another evaluator to test, not proof that judging has been fixed.

The original optional integration checks exercised Phoenix Hallucination and TruLens Coherence on one answer per route. Guardrails AI and TruLens Groundedness were not qualified. External scores do not enter the repair policy.

Native notebook jobs and manual interactive setup are different checks. The interactive Git-folder and Environment-panel steps still need a manual walkthrough. Follow the [setup instructions](../README.md) and choose saved mode if you prefer to inspect results without credentials.
