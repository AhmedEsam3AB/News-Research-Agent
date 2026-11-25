# Evaluation results

- Agent model: `nvidia/nemotron-3.5-super-vl:free`
- Judge model: `baidu/ernie-5.5-flash-preview:free`
- Code: commit `06c7036`
- Cases: 10, one run per case per mode

| Mode | Pass rate | Errors | Avg sources | Avg % sources in window | Avg URLs cited | Avg drafts | Avg searches | Avg time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_check | 2/10 (20%) | 2 | 22.0 | 100.0 | 0.0 | 1.4 | 6.6 | 160.2 |
| without_check | 0/10 (0%) | 1 | 15.6 | 100.0 | 0.0 | 1.0 | 4.4 | 92.7 |

## Per case

| Case | Kind | with_check | without_check |
| --- | --- | --- | --- |
| boeing | company | ERROR | fail: grounded |
| tesla | company | fail: grounded | fail: grounded |
| nike | company | fail: grounded | fail: grounded |
| starbucks | company | ERROR | fail: grounded |
| ryanair | company | fail: grounded, in_window | fail: grounded |
| openai | company | fail: grounded | fail: grounded, in_window, covers_main_stories, neutral |
| nintendo | company | pass | fail: grounded, in_window, neutral |
| eu-ai-act | issue | pass | fail: grounded, in_window, covers_main_stories |
| apple-one-day | narrow_window | fail: grounded | fail: grounded, in_window |
| fictional-recall | no_coverage | fail: grounded, in_window | ERROR |

Pass = the run finished and the judge said yes to all five checks: on_topic, grounded, in_window, covers_main_stories, neutral. Full reports, sources and the judge's reasons are in `runs.jsonl`.
