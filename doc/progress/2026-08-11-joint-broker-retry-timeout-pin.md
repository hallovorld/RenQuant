# Joint Broker Retry And Timeout Pin Update (PR #583)

STATUS: delivered
WHAT: Update `subrepos.lock.json` to pin `renquant-pipeline` from
`e13cd3eba37856a43acb0cd16b147bf9a2cf452e` to
`4aec0e35e8200c623c5353c74bd175a0871d3a9d`, and `renquant-execution` from
`5724dc74ec2b020dac6f567d6e0d049b2c006b4e` to
`91c7bf8873fda9d2806963da7a23032a6e8fbdc4`.
WHY/DIR: Keep the umbrella on the reviewed retry-policy and bounded account-read
timeout pair as one integration candidate. This is a pin-only change and does
not authorize any production cutover.
EVIDENCE: n/a
NEXT: Wait for the umbrella pin checks on PR #583; if they fail, revert or
re-pin from the next reviewed subrepo heads.

## Pin Delta

| Subrepo | Previous pin | Candidate pin | Source PR | Change |
| --- | --- | --- | --- | --- |
| `renquant-pipeline` | `e13cd3eba37856a43acb0cd16b147bf9a2cf452e` | `4aec0e35e8200c623c5353c74bd175a0871d3a9d` | [renquant-pipeline#286](https://github.com/hallovorld/renquant-pipeline/pull/286) | Three broker-connect attempts with two-second backoff, shared by runtime and legacy entry points; hard failure remains after exhaustion. |
| `renquant-execution` | `5724dc74ec2b020dac6f567d6e0d049b2c006b4e` | `91c7bf8873fda9d2806963da7a23032a6e8fbdc4` | [renquant-execution#41](https://github.com/hallovorld/renquant-execution/pull/41) | Apply timeout defaults around account reads without replacing the SDK session or its transport state. |

## Notes

- `subrepos.lock.json` remains valid JSON and the diff is metadata-only.
- The umbrella `subrepo-pin-ci-green` workflow remains the merge gate for this
  candidate pair.
- Rollback is a single revert of this umbrella commit, which restores both
  previous pins together.
