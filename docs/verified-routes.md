# Verified workshop routes

The current revision uses MLflow 3.17.1, OpenAI 3.8.0 and the committed dependency lock. Local setup uses Python 3.12.13. The native Databricks route uses Standard v5 with Python 3.12.3 and `requirements-workshop.txt`.

The three primary notebooks completed with fresh local kernels and as native Databricks serverless notebook tasks on October 18, 2026. The first complete outcomes are preserved below. Execution completion and acceptance are separate results.

| Route | Application and judge | All required checks, deployed to repair | Controls | Recorded decision after model review |
| --- | --- | --- | --- | --- |
| Local | OpenAI `gpt-4o-mini` | 4/10 to 9/10, five gains, no losses | 20/20 | REVIEW_REQUIRED |
| Native Databricks | Configured Qwen serving endpoint | 2/10 to 10/10, eight gains, no losses | 20/20 | PASS under the workshop policy |

The local judge incorrectly attributed the customer's request for an already processed refund to the assistant. The assistant had stated correct eligibility and a permitted customer-to-support next step. That recorded zero remains unchanged, and its material review finding remains open. The native grades agreed with the reviewed business rubric. Two related rationale findings remain recorded as nonmaterial for that comparison. Passing the authored controls did not establish judge accuracy.

Both bundles identify the models that reviewed their answers and grades. These are model reviews, not independent human labels or production approval. Each route uses the same model for its application and judge. Open the review panel in notebook 02 to inspect the customer input, answer and rationale before deciding.

Saved mode contains dated, hash-verified outputs from those actual runs in `evidence/workshop/local/final` and `evidence/workshop/databricks/final`. Local and Databricks routes remain separate. Do not pool their scores or treat ten teaching cases as production coverage. Their manifest source identities describe the measured implementation. The final notebook adds read-only review and control displays without changing measurements.

Historical answers were also rescored with the frozen v2 judge, without regenerating answers or replacing v1 decisions. Local totals changed from 8/6/9 to 8/3/8 and still contained wrong rejections. Native totals changed from 10/3/10 to 10/2/10. A revised rubric is another evaluator to test, not proof that judging has been fixed.

The original optional integration checks exercised Phoenix Hallucination and TruLens Coherence on one answer per route. Guardrails AI and TruLens Groundedness were not qualified. External scores do not enter the repair policy.

The interactive attendee route was also exercised through Chrome in Databricks Free Edition on October 18, 2026. A fresh Git folder selected workshop revision `e253860`. Each notebook received Standard v5 and the committed requirements through the Environment panel. Browser **Run all** completed all 19 code cells across 00, 01 and 02 in saved mode, displaying the default local bundle and its unchanged REVIEW_REQUIRED result.

A fresh live comparison then completed through the same browser route, with one documented configuration cell added to each notebook. All 22 code cells succeeded. This separate run recorded 2/10 to 10/10, eight gains, no losses, two ties and 20/20 controls. Its decision remains REVIEW_REQUIRED because its review is pending. It does not replace the earlier reviewed native bundle above. The MLflow interface showed all three runs, and the day-45 trace exposed the obsolete 90-day source used by the deployed answer.

Selecting the bundled native Databricks evidence explicitly in all three notebooks also completed the 22-cell saved replay. It displayed the retained reviewed result, PASS under the workshop policy. Saved replay does not make model calls. Its completed review belongs to that saved bundle, not to the new live comparison.

The browser check verified collapsed helper source, visible material-review evidence, working secondary-finding disclosures, and hiding both final verdict outputs until the audience vote. In this browser, Databricks offered **Run in strict sandbox** when its usual HTML-output host was unreachable. That option and a page reload displayed the completed HTML outputs without another model run. The [setup instructions](../README.md) include those steps and explain which settings to repeat in every notebook.

These checks establish the tested execution and browser workflow. They do not measure presentation delivery or attendee understanding.
