# Examples

Sample request payloads for the firewall's batch endpoint.

## Files

- `batch_prompts.json` — A 3-prompt smoke test (mix of safe prompts and a known
  jailbreak attempt). Use this when you want quick feedback that the pipeline runs
  end-to-end.
- `batch_prompts_100.json` — A 100-prompt batch for load-style testing. Useful for
  exercising the `concurrency` parameter and verifying per-decision counts in
  the dashboard.

## Usage

Both files target `POST /v1/chat/completions/batch` (max 1000 prompts per request):

```bash
curl -X POST http://localhost:8000/v1/chat/completions/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @examples/batch_prompts.json
```

Each request body has the shape:

```json
{
  "model": "firewall-demo",
  "system_message": "You are a helpful assistant.",
  "concurrency": 20,
  "prompts": ["…", "…"]
}
```

The response includes a `summary` of `allowed` / `blocked` / `dropped` / `errors`
counts plus a per-prompt `results` array with the firewall decision, scores, and
latencies for each entry.
